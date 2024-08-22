# Crie uma lista de 1 até 100 com somente números impares

lista = []
contador = 1

while contador <= 100:
    if contador % 2 != 0:
        lista.append(contador)
    contador += 1

print(lista)