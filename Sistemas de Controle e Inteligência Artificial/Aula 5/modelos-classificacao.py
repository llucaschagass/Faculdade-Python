from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x, y = make_blobs(n_samples = 200, n_features = 2, centers = 2, cluster_std = 2.0)

plt.scatter(x[:, 0], x[:,1], c = y)

x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=0.3)

log_reg = LogisticRegression()
dec_tree = DecisionTreeClassifier()
knn_class = KNeighborsClassifier(n_neighbors=3)

# Treinando os modelos
log_reg.fit(x_t, y_t)
dec_tree.fit(x_t, y_t)
knn_class.fit(x_t, y_t)

# Fazendo previsões
y_hat_reglog = log_reg.predict(x_v)
y_hat_dectree = dec_tree.predict(x_v)
y_hat_knn = knn_class.predict(x_v)

print("Logistic Regressor:")
print("Accuracy:", accuracy_score(y_v, y_hat_reglog))
print()
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_v, y_hat_dectree))
print()
print("KNN  Classifier:")
print("Accuracy:", accuracy_score(y_v, y_hat_knn))
print()