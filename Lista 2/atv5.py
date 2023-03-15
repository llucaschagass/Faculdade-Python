# Faça um programa que receba o salário de um funcionário e o percentual de aumento, calcule e mostre o valor do aumento e o novo salário. 

salario = float(input("Digite seu salário: ")) 
percentual = int(input("Digite o percentual de aumento: (Sem o %): ")) 
aumento = (percentual/100) * salario 

novosalario = salario + aumento 
print(f"Você recebeu um aumento de: R${aumento} e seu novo salário é {novosalario}") 