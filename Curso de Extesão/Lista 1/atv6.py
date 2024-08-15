# Construa um algoritmo/programa para determinar se o indivíduo está com um peso favorável. Essa situação é determinada através do IMC (Índice de Massa Corpórea), que é definida como sendo a relação entre o peso (PESO – em kg) e o quadrado da Altura (ALTURA – em m) do indivíduo. Ou seja, 

# Fórmula: IMC= PESO/ALTURA2 

peso = float(input("Digite seu peso:")) 
altura = float(input("Digite sua altura:")) 
imc = peso/(altura ** 2) 

print("O seu IMC é de {:.1f}".format(imc)) 

if imc < 20: 
    print("Você está abaixo do peso!") 

elif imc >= 20 or imc <25: 
    print("Peso normal, parabéns!") 

elif imc >= 25 or imc <30: 
    print("Você está com sobrepeso") 

elif imc >=30 or imc <40: 
    print("Você está obeso!") 

elif imc >=40: 
    print("Você está na obesidade morbida!") 

else: 
    print("Ops... parece que você digitou algo errado") 