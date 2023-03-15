# Faça um algoritmo/programa  que receba (leia) 3 notas de um aluno, calcule e mostre uma mensagem de acordo com sua média: 

nota1 = float(input("Digite a primeira nota:")) 
nota2 = float(input("Digite a segunda nota:")) 
nota3 = float(input("Digite a terceira nota:")) 

media = (nota1+nota2+nota3)/3 

if media >= 0 and media <3: 
    print(f"Sua média foi de {media}, você foi REPROVADO!") 

elif media >= 3 and media <7: 
    print(f"Sua média foi de {media}, você irá fazer o EXAME") 

else: 
    print(f"Sua média foi de {media}, você foi APROVADO") 