# O proprietário da empresa ABC LTDA precisa de um programa de computador para calcular o novo salário que seus funcionários irão receber a partir do mês que vem. Sabendo que o aumento de salário para todos os funcionários será de 25%, faça um programa que leia o valor do salário atual do funcionário e informe o seu novo salário acrescido de 25%. 

salarioatual = float(input("Digite o seu salário atual: "))
aumento = salarioatual * 0.25
novosalario = salarioatual + aumento

print(f"Olá! Seu salário teve um aumento de R${aumento} e seu novo salário a partir do próximo mês será de: R${novosalario}")