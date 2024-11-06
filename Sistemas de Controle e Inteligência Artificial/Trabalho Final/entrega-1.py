# Passo 1 - Configuração do ambiente: instalando e importando libs
# !pip install pyoperon

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import pyoperon as po
from pyoperon.sklearn import SymbolicRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# Passo 2 - Carregando o Dataset e realizando o treinamento

diabetes = load_diabetes()
x = diabetes.data       # variáveis independentes (características)
y = diabetes.target     # variável dependente (classes)

# Dividindo os dados em conjunto de treino (80%) e teste (20%)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Escalonando os dados
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Passo 3 - Modelos Simples (Treino e Validação)
# 3.1 - Regressão Linear

# Criando o modelo de Regressão Linear
linear_model = LinearRegression()

# Número de execuções
n_runs = 20

# Listas para armazenar os valores médios das métricas
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

for run in range(1, n_runs + 1):
    # Treinamento do modelo
    linear_model.fit(x_train_scaled, y_train)

    # Predição no conjunto de treino
    y_pred_train_lin = linear_model.predict(x_train_scaled)
    # Predição no conjunto de teste
    y_pred_test_lin = linear_model.predict(x_test_scaled)

    # Calculando as métricas de treino e teste
    mse_train_lin = mean_squared_error(y_train, y_pred_train_lin)
    mse_test_lin = mean_squared_error(y_test, y_pred_test_lin)
    r2_train_lin = r2_score(y_train, y_pred_train_lin)
    r2_test_lin = r2_score(y_test, y_pred_test_lin)

    # Armazenando os resultados para calcular a média depois
    train_mse_list.append(mse_train_lin)
    test_mse_list.append(mse_test_lin)
    train_r2_list.append(r2_train_lin)
    test_r2_list.append(r2_test_lin)

# Calculando as médias das métricas ao final de todas as execuções
mean_mse_train_lin = np.mean(train_mse_list)
mean_mse_test_lin = np.mean(test_mse_list)
mean_r2_train_lin = np.mean(train_r2_list)
mean_r2_test_lin = np.mean(test_r2_list)

# Exibindo as médias das métricas
print("\nMétricas médias após todas as execuções:")
print(f"  - MSE Médio Treino: {mean_mse_train_lin:.4f}")
print(f"  - MSE Médio Teste: {mean_mse_test_lin:.4f}")
print(f"  - R2 Médio Treino: {mean_r2_train_lin:.4f}")
print(f"  - R2 Médio Teste: {mean_r2_test_lin:.4f}")

# Passo 3 - Modelos Simples (Treino e Validação)
# 3.2 - Árvore de Decisão

# Criando o modelo de Árvore de Decisão
decision_tree_model = DecisionTreeRegressor()

# Definindo os parâmetros para busca em grade
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Número de execuções
n_runs = 20

# Listas para armazenar as métricas de cada execução
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

# Normalizando os dados
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

for run in range(1, n_runs + 1):
    # Usando GridSearchCV para encontrar os melhores hiperparâmetros
    grid_search = GridSearchCV(decision_tree_model, param_grid, cv=5)
    grid_search.fit(x_train_scaled, y_train)

    # Treinamento do modelo com os melhores parâmetros encontrados
    best_model = grid_search.best_estimator_
    best_model.fit(x_train_scaled, y_train)

    # Predição no conjunto de treino
    y_pred_train_tree = best_model.predict(x_train_scaled)
    # Predição no conjunto de teste
    y_pred_test_tree = best_model.predict(x_test_scaled)

    # Calculando as métricas de treino e teste
    mse_train_tree = mean_squared_error(y_train, y_pred_train_tree)
    mse_test_tree = mean_squared_error(y_test, y_pred_test_tree)
    r2_train_tree = r2_score(y_train, y_pred_train_tree)
    r2_test_tree = r2_score(y_test, y_pred_test_tree)

    # Armazenando os resultados para calcular a média depois
    train_mse_list.append(mse_train_tree)
    test_mse_list.append(mse_test_tree)
    train_r2_list.append(r2_train_tree)
    test_r2_list.append(r2_test_tree)

# Calculando as médias das métricas
mean_mse_train_tree = np.mean(train_mse_list)
mean_mse_test_tree = np.mean(test_mse_list)
mean_r2_train_tree = np.mean(train_r2_list)
mean_r2_test_tree = np.mean(test_r2_list)

# Exibindo as médias das métricas
print(f"\nÁrvore de Decisão (média de {n_runs} execuções)")
print(f"  - MSE Médio Treino: {mean_mse_train_tree:.4f}")
print(f"  - MSE Médio Teste: {mean_mse_test_tree:.4f}")
print(f"  - R² Médio Treino: {mean_r2_train_tree:.4f}")
print(f"  - R² Médio Teste: {mean_r2_test_tree:.4f}")

# Passo 3 - Modelos Simples (Treino e Validação)
# 3.3 - KNN

# Criando o modelo KNN
knn_model = KNeighborsRegressor()

# Criando um pipeline
pipeline = Pipeline([
    ('scaler', scaler),
    ('knn', knn_model)
])

# Definindo os parâmetros para busca em grade
param_grid = {
    'knn__n_neighbors': [3, 5, 7, 9, 11],
    'knn__weights': ['uniform', 'distance']
}

# Número de execuções
n_runs = 20
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

for run in range(n_runs):
    # Usando GridSearchCV para encontrar os melhores hiperparâmetros
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(x_train_scaled, y_train)

    # Treinamento do modelo com os melhores parâmetros encontrados
    best_model = grid_search.best_estimator_
    best_model.fit(x_train_scaled, y_train)

    # Predição no conjunto de treino
    y_pred_train_knn = best_model.predict(x_train_scaled)
    # Predição no conjunto de teste
    y_pred_test_knn = best_model.predict(x_test_scaled)

    # Calculando as métricas de treino e teste
    mse_train_knn = mean_squared_error(y_train, y_pred_train_knn)
    mse_test_knn = mean_squared_error(y_test, y_pred_test_knn)
    r2_train_knn = r2_score(y_train, y_pred_train_knn)
    r2_test_knn = r2_score(y_test, y_pred_test_knn)

    # Armazenando os resultados para calcular a média depois
    train_mse_list.append(mse_train_knn)
    test_mse_list.append(mse_test_knn)
    train_r2_list.append(r2_train_knn)
    test_r2_list.append(r2_test_knn)

# Calculando as médias das métricas
mean_mse_train_knn = np.mean(train_mse_list)
mean_mse_test_knn = np.mean(test_mse_list)
mean_r2_train_knn = np.mean(train_r2_list)
mean_r2_test_knn = np.mean(test_r2_list)

# Exibindo as médias das métricas
print(f"\nKNN (média de {n_runs} execuções)")
print(f"  - MSE Médio Treino: {mean_mse_train_knn:.4f}")
print(f"  - MSE Médio Teste: {mean_mse_test_knn:.4f}")
print(f"  - R² Médio Treino: {mean_r2_train_knn:.4f}")
print(f"  - R² Médio Teste: {mean_r2_test_knn:.4f}")

# Passo 3 - Modelos Simples (Treino e Validação)
# 3.4 - Random Forests

# Criando o modelo Random Forests
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Criando um pipeline
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),  # Normalização dos dados
    ('rf', random_forest_model)
])

# Número de execuções
n_runs = 20
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

for _ in range(n_runs):
    # Treinamento do modelo
    pipeline_rf.fit(x_train, y_train)

    # Predição no conjunto de treino
    y_pred_train_rf = pipeline_rf.predict(x_train)
    # Predição no conjunto de teste
    y_pred_test_rf = pipeline_rf.predict(x_test)

    # Calculando as métricas de treino e teste
    mse_train_rf = mean_squared_error(y_train, y_pred_train_rf)
    mse_test_rf = mean_squared_error(y_test, y_pred_test_rf)
    r2_train_rf = r2_score(y_train, y_pred_train_rf)
    r2_test_rf = r2_score(y_test, y_pred_test_rf)

    # Armazenando os resultados para calcular a média depois
    train_mse_list.append(mse_train_rf)
    test_mse_list.append(mse_test_rf)
    train_r2_list.append(r2_train_rf)
    test_r2_list.append(r2_test_rf)

# Calculando as médias das métricas
mean_mse_train_rf = np.mean(train_mse_list)
mean_mse_test_rf = np.mean(test_mse_list)
mean_r2_train_rf = np.mean(train_r2_list)
mean_r2_test_rf = np.mean(test_r2_list)

# Exibindo as médias das métricas
print(f"\nRandom Forest (média de {n_runs} execuções)")
print(f"  - MSE Médio Treino: {mean_mse_train_rf:.4f}")
print(f"  - MSE Médio Teste: {mean_mse_test_rf:.4f}")
print(f"  - R² Médio Treino: {mean_r2_train_rf:.4f}")
print(f"  - R² Médio Teste: {mean_r2_test_rf:.4f}")

# Passo 3 - Modelos Simples (Treino e Validação)
# 3.5 - Programação Genética (Pyoperon)

# Função para gerar os dados (com ruído) para a simulação
def f_real(x):
    return 3*x**2*np.cos(x) + 2*x

# Geração dos dados
x = np.linspace(0, 10, 100)
y = f_real(x)
y_ruido = np.random.normal(y, 15)  # Adiciona ruído aos dados

# Divisão dos dados
x_train, x_test, y_train, y_test = train_test_split(x, y_ruido, test_size=0.3, random_state=42)

# Configuração do modelo SymbolicRegressor
symbolic_model = SymbolicRegressor(
    crossover_probability=0.95,
    mutation_probability=0.25,
    max_evaluations=1000,
    allowed_symbols='add,sub,mul,div,pow,sin,cos,constant,variable',
    objectives=['r2']
)

# Avaliação do modelo SymbolicRegressor com múltiplas execuções
n_runs = 20
train_mse_list = []
test_mse_list = []
train_r2_list = []
test_r2_list = []

for _ in range(n_runs):
    # Treinamento do modelo
    symbolic_model.fit(x_train.reshape(-1, 1), y_train)

    # Predição no conjunto de treino e teste
    y_pred_train_sym = symbolic_model.predict(x_train.reshape(-1, 1))
    y_pred_test_sym = symbolic_model.predict(x_test.reshape(-1, 1))

    # Calculando as métricas de treino e teste
    mse_train_sym = mean_squared_error(y_train, y_pred_train_sym)
    mse_test_sym = mean_squared_error(y_test, y_pred_test_sym)
    r2_train_sym = r2_score(y_train, y_pred_train_sym)
    r2_test_sym = r2_score(y_test, y_pred_test_sym)

    # Armazenando os resultados para calcular a média depois
    train_mse_list.append(mse_train_sym)
    test_mse_list.append(mse_test_sym)
    train_r2_list.append(r2_train_sym)
    test_r2_list.append(r2_test_sym)

# Calculando as médias das métricas
mean_mse_train_sym = np.mean(train_mse_list)
mean_mse_test_sym = np.mean(test_mse_list)
mean_r2_train_sym = np.mean(train_r2_list)
mean_r2_test_sym = np.mean(test_r2_list)

# Exibindo as médias das métricas
print(f"\nRegressão Simbólica (média de {n_runs} execuções)")
print(f"  - MSE Médio Treino: {mean_mse_train_sym:.4f}")
print(f"  - MSE Médio Teste: {mean_mse_test_sym:.4f}")
print(f"  - R² Médio Treino: {mean_r2_train_sym:.4f}")
print(f"  - R² Médio Teste: {mean_r2_test_sym:.4f}")

# Passo 4 - Validação Cruzada com K-Fold

# Definindo o número de folds
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# Função para treinar e avaliar o modelo com K-Fold, retornando as métricas médias de MSE e R²
def evaluate_model_kfold(model, X, y):
    mse_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Treinando o modelo
        model.fit(X_train, y_train)

        # Fazendo previsões e calculando o MSE e R²
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mse_scores.append(mse)
        r2_scores.append(r2)

    # Retornando a média e o desvio padrão das métricas
    return np.mean(mse_scores), np.std(mse_scores), np.mean(r2_scores), np.std(r2_scores)

# Modelos a serem aplicados no K-Fold
models = {
    "Regressão Linear": LinearRegression(),
    "Árvore de Decisão": DecisionTreeRegressor(),
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "Random Forests": RandomForestRegressor(),
    "Programação Genética": SymbolicRegressor(
        crossover_probability=0.95,
        mutation_probability=0.25,
        max_evaluations=1000,
        allowed_symbols="add,sub,mul,div,pow,sin,cos,constant,variable",
        objectives=["r2"]
    )
}

# Certificando-se de que os dados estão no formato correto
X = x.reshape(-1, 1)  # Redimensionando os dados

# Aplicando K-Fold em cada modelo e imprimindo as métricas médias
for model_name, model in models.items():
    mean_mse, std_mse, mean_r2, std_r2 = evaluate_model_kfold(model, X, y)
    print(f"{model_name}:")
    print(f"  - MSE Médio: {mean_mse:.4f} ± {std_mse:.4f}")
    print(f"  - R² Médio: {mean_r2:.4f} ± {std_r2:.4f}\n")

# Função para exibir gráficos de comparação
def plot_results(mse_train, mse_test, mse_kfold_mean, mse_kfold_std, model_name):
    plt.figure(figsize=(10, 6))
    labels = ['Treino (Sem K-Fold)', 'Teste (Sem K-Fold)', 'Treino (K-Fold)', 'Teste (K-Fold)']
    mse_values = [mse_train, mse_test, mse_kfold_mean, mse_kfold_mean + mse_kfold_std]

    plt.bar(labels, mse_values, color=['blue', 'green', 'blue', 'green'], alpha=0.8)
    plt.title(f'Erro Quadrático Médio - {model_name}')
    plt.ylabel('MSE')
    plt.ylim(0, max(mse_values) * 1.2)
    plt.xticks(rotation=45)
    plt.show()

# Para cada modelo
for model_name, model in models.items():
    # Ajuste do formato para 2D para o modelo
    x_train_reshaped = x_train.reshape(-1, 1)
    x_test_reshaped = x_test.reshape(-1, 1)

    # Cálculo do MSE sem K-Fold
    model.fit(x_train_reshaped, y_train)
    y_pred_train = model.predict(x_train_reshaped)
    y_pred_test = model.predict(x_test_reshaped)

    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)

    # Cálculo do MSE com K-Fold
    mse_kfold_mean, mse_kfold_std, _, _ = evaluate_model_kfold(model, x.reshape(-1, 1), y)

    # Exibir gráfico de comparação
    plot_results(mse_train, mse_test, mse_kfold_mean, mse_kfold_std, model_name)

"""**CONCLUSÂO:**

Ao aplicar e comparar os cinco modelos de regressão com e sem o uso do K-Fold Cross-Validation, podemos compreender mais profundamente a influência dessa técnica na estabilidade e confiabilidade dos resultados. A utilização do K-Fold oferece benefícios como maior confiabilidade nas métricas de desempenho, redução do overfitting e uma avaliação mais robusta dos modelos.

Embora o K-Fold não melhore diretamente a capacidade preditiva dos modelos, ele contribui para uma validação mais precisa e estável, permitindo uma melhor compreensão do desempenho dos modelos em dados não vistos. Ao comparar os resultados com e sem K-Fold, espera-se observar uma diferença significativa na confiabilidade das métricas, reforçando a importância dessa técnica na análise e seleção de modelos mais consistentes.

Por fim, embora os resultados deste exercicio prático não mostrem de forma clara uma melhora ao utilizar o K-Fold, a análise e compreensão dessa técnica permitiu entender sua importância na avaliação dos modelos. Mesmo sem uma diferença nos resultados obtidos, o estudo reforça que o uso do K-Fold é essencial para alcançar uma validação mais confiável, o que é muito relevante em contextos com dados mais complexos.
"""