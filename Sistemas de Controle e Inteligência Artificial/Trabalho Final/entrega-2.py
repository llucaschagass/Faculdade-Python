# Importação das bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.datasets import load_sample_image
import scipy.spatial.distance

# Função que calcula as distâncias de cada elemento em relação aos centróides
def calculate_distance(X, centroids):
    return scipy.spatial.distance.cdist(X, centroids)

# Função que distribui cada ponto para seu devido cluster
def assign_clusters(distances):
    return np.argmin(distances, axis=1)

# Função que atualiza a posição dos centróides
def update_centroids(X, group_idx, k):
    return np.array([X[group_idx == i].mean(axis=0) for i in range(k)])

# Função de clusterização K-means
def kmeans(X, k, epsilon=1e-5, max_iter=50):
    """
    Algoritmo de clusterização K-means.

    Parâmetros:
        X (numpy.ndarray): Dados de entrada com formato (n_amostras, n_características).
        k (int): Número de clusters.
        epsilon (float): Limite de convergência.
        max_iter (int): Número máximo de iterações.

    Retornos:
        centroids (numpy.ndarray): Centróides finais de cada cluster.
        group_idx (numpy.ndarray): Atribuição de cluster para cada amostra.
        J (list): Valores da função objetivo por iteração.
    """
    # Inicialização dos centróides aleatórios
    np.random.seed(42)
    random_idx = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[random_idx]

    J = []  # Lista para armazenar os valores da função objetivo

    for iteration in range(max_iter):
        # Passo 1: Calcular distâncias
        distances = calculate_distance(X, centroids)

        # Passo 2: Atribuir clusters
        group_idx = assign_clusters(distances)

        # Passo 3: Atualizar centróides
        new_centroids = update_centroids(X, group_idx, k)

        # Passo 4: Calcular função objetivo (soma das distâncias mínimas)
        J_current = np.sum(np.min(distances, axis=1))
        J.append(J_current)

        # Critério de parada
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break

        centroids = new_centroids

    return centroids, group_idx, J

# Teste com dados simulados
X, _ = make_classification(
    n_samples=150,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    scale=7
)

# Número de clusters
k = 3
centroids, group_idx, J = kmeans(X, k=k)

# Visualização dos resultados
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=group_idx, cmap='viridis', marker='o', edgecolor='k', s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroides')
plt.title(f'K-means clustering (K={k})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Visualização da convergência
plt.figure(figsize=(8, 6))
plt.plot(range(len(J)), J, marker='o', linestyle='-', color='b')
plt.title('Convergência do K-means')
plt.xlabel('Iteração')
plt.ylabel('Função Objetivo (J)')
plt.grid()
plt.show()

# Teste com imagem real
k = 8

# Passo 1 - Carrega a imagem original
I = load_sample_image("flower.jpg")

# Passo 2 - Normalizar os valores para [0,1]
I = np.array(I, dtype=np.float64) / 255

# Passo 3 - Converter a imagem em um array de pixels
X = I.reshape(I.shape[0] * I.shape[1], 3)

# Passo 4 - Aplicar K-means na imagem
centroids, group_idx, J = kmeans(X, k)

# Passo 5 - Quantizar a imagem substituindo os valores RGB pelos centróides
quantized_image = np.zeros_like(X)
for i in range(k):
    quantized_image[group_idx == i] = centroids[i]

quantized_image = quantized_image.reshape(I.shape)

# Visualizar imagem original
plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Imagem Original')
plt.imshow(I)

# Visualizar imagem quantizada
plt.figure(3)
plt.clf()
plt.axis('off')
plt.title(f'Imagem Quantizada ({k} cores, K-means)')
plt.imshow(quantized_image)
plt.show()

"""Conclusão:

**Explicação do código:**

O código implementa o algoritmo K-means para agrupar dados e realizar quantização de imagens. Primeiro ele é aplicado em um conjunto de dados bidimensionais gerados artificialmente com três clusters. O algoritmo identifica os grupos, posicionando os centróides próximos ao centro de cada cluster. A convergência é evidenciada pela redução progressiva da função objetivo (J), que mede a soma das distâncias mínimas entre os pontos e seus centróides.

No segundo teste, o K-means foi utilizado para a quantização de uma imagem. Aqui, os valores RGB dos pixels foram agrupados em 8 clusters, reduzindo a paleta de cores da imagem original para oito tons representados pelos centróides calculados. O resultado preservou as características visuais principais da imagem, apesar da redução de complexidade nas cores. Esse processo de quantização é útil em aplicações como compressão de imagens e redução de custos computacionais em análises visuais.

**Explicação de cada plot:**

Temos a visualização dos dados simulados bidimensionais. Os pontos estão distribuídos em três clusters, indicados por cores diferentes, e os centróides finais estão destacados com marcadores em formato de "X". É possível observar que os centróides estão bem posicionados, próximos ao centro de cada grupo, demonstrando que o algoritmo conseguiu identificar corretamente os padrões de agrupamento nos dados.

Em seguida temos o gráfico que mostra a convergência do K-means ao longo das iterações. A curva representa os valores da função objetivo (J), que mede a soma das distâncias mínimas entre os pontos e os centróides.

Por fim temos a aplicação do K-means para a quantização de cores em uma imagem. Primeiro temos a versão original, rica em detalhes e variações de cores e depois temos uma imagem após a redução para 8 cores (K=8). Apesar da simplificação, a imagem quantizada mantém as características principais da cena, como o formato da flor e o contraste com o fundo, demonstrando que o algoritmo preserva a essência visual enquanto reduz a complexidade da paleta de cores.

Após a análise do código e dos resultados obtivos podemos concluir que o algoritmo K-means é eficaz tanto para agrupamento de dados quanto para aplicações práticas como compressão de imagens. Os resultados dependem de escolhas como o valor de K e a inicialização dos centróides, no caso de agrupamento o interessante seria testar diferentes valores de K até encontrar a melhor separação. Já na quantização quanto maior o K melhores seriam os detalhes da imagem.

De pontos positivos consigo destacar:

A clareza nos resultados obtidos, tanto nos dados simulados quanto na

*   A clareza nos resultados obtidos, tanto nos dados simulados quanto na quantização de cores
*   A flexibilidade do K-means para aplicações diversas

E negativos:

*   A sensibilidade ao número de clusters é uma limitação
*   No caso da quantização, com valores menores de K, alguns nuances importantes da imagem original podem ser perdidas



"""