#Calcule o IMC

altura = float(input("Digite sua altura em metros: "))
peso = float(input("Digite seu peso em quilogramas: "))

imc = peso / (altura ** 2)

print(f"Seu IMC é: {imc:.2f}")