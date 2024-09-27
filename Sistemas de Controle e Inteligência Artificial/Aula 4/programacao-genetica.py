#necessário: !pip install -Uq pydotplus graphviz pyoperon

import numpy as np
import matplotlib.pyplot as plt
from pyoperon.sklearn import SymbolicRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr
from graphviz import Source
from IPython.display import Image

def f_real(x):
  return 3*x**2*np.cos(x) + 2*x

x = np.linspace(0, 10, 100)
y = f_real(x)
y_ruido = np.random.normal(y, 15)

# plt.plot(x, y)
plt.figure(figsize=(6, 4))
plt.scatter(x, y_ruido)
plt.show()

x_treino, x_val, y_treino, y_val = train_test_split(x, y_ruido, test_size=0.3)

# configurando regressor simbólico

pg = SymbolicRegressor(crossover_probability=0.95,
                       mutation_probability=0.25,
                       max_evaluations=1000,
                       allowed_symbols = 'add,sub,mul,div,pow,sin,cos,constant,variable',
                       objectives=['r2'])

# treinando o regressor
pg.fit(x_treino.reshape(-1, 1), y_treino)

# obtendo fórmula matemática simbólica
pg.get_model_string(pg.model_)

# usando o modelo treinado
y_predito = pg.predict(x_val.reshape(-1, 1))

# visualizando o desempenho do modelo treinado

idx = np.argsort(x_val)

plt.figure(figsize=(6, 4))
plt.plot(x_val[idx], y_predito[idx], color="orange", label="f_predito")
plt.scatter(x_val[idx], y_val[idx], color="blue", label="dados")
plt.plot(x_val[idx], f_real(x_val[idx]), color="blue", label="f_real")
plt.grid()
plt.legend()
plt.show()

