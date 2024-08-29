idades = {}

idades["Lucas"] = 23
idades["Daniel"] = 20
idades["Paulo"] = 26
idades["Gustavo"] = 21
idades["Salvador"] = 22
indice = 0
total = 0

for idade in idades:
  total =  total + idades[idade]
  indice = indice + 1


print(total/indice)