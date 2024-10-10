# Importação das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregamento dos dados de câncer de mama da sklearn
data = load_breast_cancer()
X = data.data  # Variáveis independentes (características)
y = data.target  # Variável dependente (classes)

# Função para plotar os resultados
def plot_results(models_results):
    """Função para plotar a acurácia média dos modelos no treino e teste"""
    labels = list(models_results.keys())
    train_accuracies = [results['train'] for results in models_results.values()]
    test_accuracies = [results['test'] for results in models_results.values()]

    x = np.arange(len(labels))  # Posições para as labels
    width = 0.35  # Largura das barras

    # Criação das barras
    fig, ax = plt.subplots()
    bars_train = ax.bar(x - width/2, train_accuracies, width, label='Treino')
    bars_test = ax.bar(x + width/2, test_accuracies, width, label='Teste')

    # Adiciona rótulos e título
    ax.set_ylabel('Acurácia média')
    ax.set_title('Acurácia média por modelo')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Exibe os valores acima das barras
    for bar in bars_train:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for bar in bars_test:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Mostra o gráfico
    plt.tight_layout()
    plt.show()

# Divisão dos dados entre treino (80%) e teste (20%)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Dicionário para armazenar os resultados dos modelos
models_results = {}

# Definição dos modelos a serem utilizados
models = {
    'Decision Tree': DecisionTreeClassifier(max_depth=3),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=5000)
}

# Treinamento e avaliação de cada modelo
for model_name, model in models.items():
    """Função para treinar e avaliar o modelo várias vezes e retornar as acurácias médias"""
    n_runs = 20  # Número de execuções para obter médias
    train_accuracies = []
    test_accuracies = []

    for _ in range(n_runs):
        # Treinamento do modelo
        model.fit(x_train, y_train)
        # Predição no conjunto de treino
        y_pred_train = model.predict(x_train)
        # Verifique a acurácia dos dados de treino
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_accuracies.append(train_accuracy)

        # Predição no conjunto de teste
        y_pred_test = model.predict(x_test)
        # Verifique a acurácia dos dados de Teste
        test_accuracy = accuracy_score(y_test, y_pred_test)
        test_accuracies.append(test_accuracy)

    # Retorna as acurácias médias de treino e teste
    train_acc, test_acc = np.mean(train_accuracies), np.mean(test_accuracies)
    models_results[model_name] = {'train': train_acc, 'test': test_acc}

# Plotagem dos resultados
plot_results(models_results)