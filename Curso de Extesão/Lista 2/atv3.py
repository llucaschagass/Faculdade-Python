# Faça um algoritmo/programa que receba os lados de um retângulo, calcule a área e perímetro do mesmo e mostre o resultado na tela.  

lados = float(input("Digite o tamanho dos lados do triângulo: "))
unidade = str(input("Digite a unidade de medida: "))

altura = lados * 0,8660 # altura retirada da seguinte formula: h = x√3/2
area = (lados * altura)/2
perimetro = lados * 3

print(f"O seu triângulo tem a área de {area}{unidade}² e um perímetro de {perimetro}{unidade}")