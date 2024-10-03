from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor


x_reg, y_reg = make_regression(n_samples=100, n_features=1, n_targets=1, noise=4)


plt.scatter(x_reg, y_reg)
plt.show()


x_t, x_v, y_t, y_v = train_test_split(x_reg, y_reg, test_size=0.3)


reg_lin = LinearRegression()
dec_tree = DecisionTreeRegressor()
knn = KNeighborsRegressor(n_neighbors=3)

# Treinando os modelos
reg_lin.fit(x_t, y_t)
dec_tree.fit(x_t, y_t)
knn.fit(x_t, y_t)

# Fazendo previsões
y_hat_reglin = reg_lin.predict(x_v)
y_hat_dectree = dec_tree.predict(x_v)
y_hat_knn = knn.predict(x_v)

print("Linear Regressor:")
print("MSE:", mean_squared_error(y_v, y_hat_reglin))
print("R²:", r2_score(y_v, y_hat_reglin))
print()
print("Decision Tree Regressor:")
print("MSE:", mean_squared_error(y_v, y_hat_dectree))
print("R²:", r2_score(y_v, y_hat_dectree))
print()
print("KNN Regressor:")
print("MSE:", mean_squared_error(y_v, y_hat_knn))
print("R²:", r2_score(y_v, y_hat_knn))