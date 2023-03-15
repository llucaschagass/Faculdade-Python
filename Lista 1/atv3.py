# Faça um algoritmo/programa que leia dois valores inteiros A e B se os valores forem iguais deverá se somar os dois, caso contrário multiplique A por B. Ao final de qualquer um dos cálculos deve-se atribuir o resultado para uma variável C e mostrar seu conteúdo na tela.

numA = int(input("Digite um valor para A:")) 
numB = int(input("Digite um valor para B:")) 

if numA == numB: 
    numC = numA + numB 
    print(numC) 

else: 
    numC = numA * numB 
    print(numC)  