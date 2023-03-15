# Faça um algoritmo/programa que imprima o dobro de um número caso ele seja positivo e o seu triplo caso seja negativo, imprimindo o resultado. 

num = int(input("Digite um número:")) 

if num > 0: 
    num = num * 2 
    print(num) 

elif num < 0: 
    num = num * 3 
    print(num) 

else: 
    print("Número inválido, digite um valor maior ou menor que 0") 