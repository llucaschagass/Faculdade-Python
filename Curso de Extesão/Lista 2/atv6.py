# Faça um programa que receba o salário base de um funcionário, calcule e mostre seu salário a receber, sabendo-se que o funcionário tem gratificação de R$ 50 e paga imposto de 10% sobre o salário base. 

salario = float(input("Digite seu salário base: ")) 
salariofinal = (salario + 50) - (salario * 0.10)  

print(f"Seu salário base é de: R${salario} e você irá receber R${salariofinal}") 