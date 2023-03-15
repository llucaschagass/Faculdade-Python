# Altere o programa para que o usuário digite os 4 boletos que ele deve pagar, e o salário bruto que ele tem disponível e lhe retorne: 

# O Salário líquido (desconto de 14% em folha);   
# O total das contas 
# O restante do salário. 

salariobruto = float(input("Digite seu salário bruto:")) 
boleto1 = float(input("Digite o valor do primeiro boleto:")) 
boleto2 = float(input("Digite o valor do segundo boleto:")) 
boleto3 = float(input("Digite o valor do terceiro boleto:")) 
boleto4 = float(input("Digite o valor do quarto boleto:")) 
desconto = salariobruto * 0.14 
salarioliquido = salariobruto - desconto 

valorfacada = boleto1 + boleto2 + boleto3 + boleto4 
sobra = salarioliquido- valorfacada 

print(f"Seu salário é de: R${salariobruto}. Há um desconto de R${desconto} fiquando líquido: R$ {salarioliquido} ") 

print(f"Suas despesas somam R${valorfacada} e irá lhe sobrar: R${sobra}") 
