# Faça um algoritmo/programa que leia os valores A, B, C e imprima na tela se a soma de A + B é menor que C.  

num1 = int(input("Digite um valor para A:")) 
num2 = int(input("Digite um valor para B:")) 
num3 = int(input("Digite um valor para C:")) 
soma = num1+num2 

if soma < num3: 
    print("A soma de A+B é menor que C!") 

else: 
    print("A soma de A+B é maior que C!") 