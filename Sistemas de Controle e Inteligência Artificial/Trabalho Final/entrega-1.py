# Instalação do Pyoperon (biblioteca para Programação Genética)
# !pip install -qU pyoperon

# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from pyoperon.sklearn import SymbolicRegressor

# Carregando o dataset de diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

# Configurações
n_runs = 50  # Número de execuções para analisar a variância

# Função para avaliação usando train-test split
def evaluate_model_train_test(model, X, y):
    mse_scores_train, mse_scores_test = [], []
    r2_scores_train, r2_scores_test = [], []

    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Treinando o modelo
        model.fit(X_train, y_train)

        # Previsões
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Cálculo do MSE
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        mse_scores_train.append(mse_train)
        mse_scores_test.append(mse_test)
        r2_scores_train.append(r2_train)
        r2_scores_test.append(r2_test)

    return mse_scores_train, mse_scores_test, r2_scores_train, r2_scores_test

# Função para avaliação usando K-Fold Cross Validation
def evaluate_model_kfold(model, X, y):
    mse_scores, r2_scores = [], []

    for _ in range(n_runs):
        fold_mse_scores = []
        fold_r2_scores = []
        kf = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Treinando o modelo
            model.fit(X_train, y_train)

            # Previsão e cálculo do MSE
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            fold_mse_scores.append(mse)
            fold_r2_scores.append(r2)

        mse_scores.append(np.mean(fold_mse_scores))
        r2_scores.append(np.mean(fold_r2_scores))

    return mse_scores, r2_scores

# Função para plotar os gráficos
def plot_results(train_test_mse, kfold_mse, model_name):
    plt.figure(figsize=(12, 6))

    # Plot do Train-Test Split
    plt.subplot(1, 2, 1)
    plt.boxplot(train_test_mse, labels=["Treino", "Teste"])
    plt.title(f'{model_name} - Train-Test Split MSE')
    plt.ylabel("MSE")

    # Plot do K-Fold Cross-Validation
    plt.subplot(1, 2, 2)
    plt.boxplot([kfold_mse], labels=["K-Fold"])
    plt.title(f'{model_name} - K-Fold Cross-Validation MSE')

    plt.suptitle(f'Comparação de Variância do {model_name}')
    plt.show()

# Função para plotar os gráficos
def plot_results_r2(train_r2, test_r2, kfold_r2, model_name):
    plt.figure(figsize=(12, 6))

    # Primeiro gráfico: Treino vs Validação
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_r2)), train_r2, label="Treino")
    plt.plot(range(len(test_r2)), test_r2, label="Validação")
    plt.title(f'{model_name} - Train vs Validation R2 Score')
    plt.xlabel("Época")
    plt.ylabel("R2 Score")
    plt.legend()

    # Segundo gráfico: R2 Score da Validação Cruzada
    plt.subplot(1, 2, 2)
    plt.plot(range(len(kfold_r2)), kfold_r2)
    plt.title(f'{model_name} - K-Fold Cross-Validation R2 Score')
    plt.xlabel("Época")
    plt.ylabel("R2 Score")

    plt.suptitle(f'Comparação de Desempenho do {model_name}')
    plt.tight_layout()
    plt.show()

# Definindo os modelos
models = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Decisão": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(),
    "Random Forests": RandomForestRegressor(),
    "Programação Genética": SymbolicRegressor(
            allowed_symbols='add,sub,mul,div,pow,constant,variable',
            mutation_probability=.25,
            crossover_probability=.99,
            generations=5000,
            n_threads=1,
            time_limit=60 * 60,  # 1 hours
            max_evaluations=5000,
            population_size=5000,
            max_length=20,
            tournament_size=3,
            epsilon=1e-5,
            reinserter='keep-best',
            offspring_generator='basic',
            objectives=["r2"]
        )

}

# Avaliação de cada modelo
for model_name, model in models.items():
    # Train-Test Split
    mse_scores_train, mse_scores_test, r2_scores_train, r2_scores_test = evaluate_model_train_test(model, X, y)

    # K-Fold Cross-Validation
    mse_scores_kfold, r2_scores_kfold = evaluate_model_kfold(model, X, y)

    # Plotando resultados
    plot_results((mse_scores_train, mse_scores_test), mse_scores_kfold, model_name)
    plot_results_r2(r2_scores_train, r2_scores_test, r2_scores_kfold, model_name)

    # Exibindo métricas descritivas
    print(f'{model_name} - Train-Test Split Variância:')
    print(f'Treino MSE Média: {np.mean(mse_scores_train)}, Desvio Padrão: {np.std(mse_scores_train)}')
    print(f'Teste MSE Média: {np.mean(mse_scores_test)}, Desvio Padrão: {np.std(mse_scores_test)}')

    print(f'{model_name} - K-Fold Cross-Validation Variância:')
    print(f'MSE Média: {np.mean(mse_scores_kfold)}, Desvio Padrão: {np.std(mse_scores_kfold)}')
    print("\n")

"""**CONCLUSÂO:**

Ao aplicar e comparar os cinco modelos de regressão com e sem o uso do K-Fold Cross-Validation, podemos compreender mais profundamente a influência dessa técnica na estabilidade e confiabilidade dos resultados. A utilização do K-Fold oferece benefícios como maior confiabilidade nas métricas de desempenho, redução do overfitting e uma avaliação mais robusta dos modelos.

Embora o K-Fold não melhore diretamente a capacidade preditiva dos modelos, ele contribui para uma validação mais precisa e estável, permitindo uma melhor compreensão do desempenho dos modelos em dados não vistos. Ao comparar os resultados com e sem K-Fold, espera-se observar uma diferença significativa na confiabilidade das métricas, reforçando a importância dessa técnica na análise e seleção de modelos mais consistentes.

Por fim, embora os resultados deste exercicio prático não mostrem de forma clara uma melhora ao utilizar o K-Fold, a análise e compreensão dessa técnica permitiu entender sua importância na avaliação dos modelos. Mesmo sem uma diferença nos resultados obtidos, o estudo reforça que o uso do K-Fold é essencial para alcançar uma validação mais confiável, o que é muito relevante em contextos com dados mais complexos.
"""