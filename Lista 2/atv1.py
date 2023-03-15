# Na aula passada foi solicitado para executar um programa. Altere o programa para que o usuário digite os 4 boletos que ele deve pagar, e o salário líquido que ele tem disponível e imprima: 

# • o total de suas contas; 
# • o valor que irá sobrar (ou faltar). 

salario = float(input("Digite seu salário:")) 
boleto1 = float(input("Digite o valor do primeiro boleto:")) 
boleto2 = float(input("Digite o valor do segundo boleto:")) 
boleto3 = float(input("Digite o valor do terceiro boleto:")) 
boleto4 = float(input("Digite o valor do quarto boleto:")) 

valorfacada = boleto1 + boleto2 + boleto3 + boleto4 
sobra = salario - valorfacada 

print(f"Seu salário é de: R${salario}. Suas despesas somam R${valorfacada} e irá lhe sobrar: R${sobra}") 