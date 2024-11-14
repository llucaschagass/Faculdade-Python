from sklearn.datasets import load_digits
digits = load_digits()

numero = 49

# Exibe a primeira imagem da base de dados
print(digits.images[numero])

import matplotlib.pyplot as plt
plt.figure(figsize=(2, 2))
plt.axis('off')
plt.imshow(digits.images[numero], cmap=plt.cm.gray_r, interpolation='nearest')

# Exibe as dimensões dos dados de características da primeira imagem
print(digits.data[numero].shape)
print(digits.data[numero])

# Carrega os dados de características (X) e alvos (y)
X = digits.data
print(X.shape)  # (1797, 64)

print(digits.target[:5])
y = digits.target % 2  # Transformação do alvo para indicar se o número é par ou ímpar
print(y[:5])

numero_selecionado = digits.target[numero]  # Obtém o dígito correspondente ao índice

if numero_selecionado % 2 == 0:
    print(f"O número {numero_selecionado} é par.")
else:
    print(f"O número {numero_selecionado} é ímpar.")

# Divide os dados em conjunto de treinamento e teste
n = X.shape[0]
n_train = int(n * 0.8)

X_train = X[:n_train, :]
y_train = y[:n_train]

X_test = X[n_train:, :]
y_test = y[n_train:]

# Treina um classificador de rede neural
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(10,))
clf.fit(X_train, y_train)
y_hat = clf.predict(X_test)

# Avaliação do modelo
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_hat)
print(acc)

from sklearn.metrics import confusion_matrix
print("Confusion matrix:\n%s" % confusion_matrix(y_test, y_hat))

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Carrega a base de dados de dígitos
digits = load_digits()

plt.figure(figsize=(2, 2))
plt.axis('off')
plt.title(f'Imagem da base de dados - Índice {numero}')
plt.imshow(digits.images[numero], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

# Carrega os dados de características (X) e alvos (y)
X = digits.data
y = digits.target

# Divide os dados em conjunto de treinamento e teste
n = X.shape[0]
n_train = int(n * 0.8)

X_train = X[:n_train, :]
y_train = y[:n_train]
X_test = X[n_train:, :]
y_test = y[n_train:]

# Treina um classificador de rede neural
clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=500)
clf.fit(X_train, y_train)

# Faz a previsão na imagem selecionada
sample_image = X[numero].reshape(1, -1)  # Usa a variável 'numero' para selecionar a imagem
predicted_digit = clf.predict(sample_image)

# Mostra a previsão
print(f"\nA previsão do modelo para a imagem é: {predicted_digit[0]}")