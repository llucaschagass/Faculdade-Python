import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

x = np.linspace(0, 5, 20)
y = x ** 2
y_real = np.random.normal(y, 5)

dt1 = DecisionTreeRegressor(max_depth=3) #f1
dt2 = DecisionTreeRegressor(max_depth=3) #f2
dt3 = DecisionTreeRegressor(max_depth=3) #f3
dt4 = DecisionTreeRegressor(max_depth=3) #f4
dt5 = DecisionTreeRegressor(max_depth=3) #f5

#Sem Learn Rate

f_0 = np.ones(20) * np.mean(y)
r_0 = y_real - f_0

# Treinando a primeira árvore
dt1.fit(x.reshape(-1, 1), r_0)
y1 = dt1.predict(x.reshape(-1, 1))

f_1 = f_0 + y1
r_1 = y_real - f_1

# Treinando a segunda árvore
dt2.fit(x.reshape(-1, 1), r_1)
y2 = dt2.predict(x.reshape(-1, 1))

f_2 = f_1 + y2
r_2 = y_real - f_2 

# Treinando a terceira árvore
dt3.fit(x.reshape(-1, 1), r_2)
y3 = dt3.predict(x.reshape(-1, 1))

f_3 = f_2 + y3
r_3 = y_real - f_3

# Treinando a quarta árvore
dt4.fit(x.reshape(-1, 1), r_3)
y4 = dt4.predict(x.reshape(-1, 1))

f_4 = f_3 + y4
r_4 = y_real - f_4

# Treinando a quinta árvore
dt5.fit(x.reshape(-1, 1), r_4)
y5 = dt5.predict(x.reshape(-1, 1))

f_5 = f_4 + y5 
r_5 = y_real - f_5

# Visualização
plt.scatter(x, y_real)
#plt.plot(x, y)
plt.plot(x, f_0 + y1 + y2 + y3 + y4 + y5) 
plt.show()

#Com Learn Rate
f_0 = np.ones(20) * np.mean(y)
r_0 = y_real - f_0
learn_rate = 0.25

# Treinando a primeira árvore
dt1.fit(x.reshape(-1, 1), r_0)
y1 = learn_rate * dt1.predict(x.reshape(-1, 1))

f_1 = f_0 + y1
r_1 = y_real - f_1

# Treinando a segunda árvore
dt2.fit(x.reshape(-1, 1), r_1)
y2 = learn_rate * dt2.predict(x.reshape(-1, 1))

f_2 = f_1 + y2
r_2 = y_real - f_2 

# Treinando a terceira árvore
dt3.fit(x.reshape(-1, 1), r_2)
y3 = learn_rate * dt3.predict(x.reshape(-1, 1))

f_3 = f_2 + y3
r_3 = y_real - f_3

# Treinando a quarta árvore
dt4.fit(x.reshape(-1, 1), r_3)
y4 = learn_rate * dt4.predict(x.reshape(-1, 1))

f_4 = f_3 + y4
r_4 = y_real - f_4

# Treinando a quinta árvore
dt5.fit(x.reshape(-1, 1), r_4)
y5 = learn_rate * dt5.predict(x.reshape(-1, 1))

f_5 = f_4 + y5 
r_5 = y_real - f_5

# Visualização
plt.scatter(x, y_real)
#plt.plot(x, y)
plt.plot(x, f_0 + y1 + y2 + y3 + y4 + y5) 
plt.show()