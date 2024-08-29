nome = input("Digite o seu nome: ")
protocolo = input("Digite o número do protocolo: ")
data_protocolo = input("Digite a data do protocolo: ")

nome_separado = nome.split()
ultimo_nome = nome_separado[-1]

ano_protocolo = int(data_protocolo[6:])
ano_atual = 2025
anos_dif = ano_atual - ano_protocolo

ultimas_letras = ''.join([nome[-1] for nome in nome_separado])

identificador = f"{protocolo}-{ultimas_letras}-{ano_protocolo}"

print(f"Sr(a). {ultimo_nome}, o seu identificador é: {identificador}.")
print(f"Seu protocolo irá perder a validade em {anos_dif} anos")