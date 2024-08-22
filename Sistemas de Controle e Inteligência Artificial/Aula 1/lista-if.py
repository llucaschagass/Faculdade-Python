# Este código verifica se o aluno inputado pelo usuario está na lista 'alunos'.

alunos = [
    "Lucas", "Salvador", "Paulo", "Jones", "Gustavo",
    "Ana", "Bruna", "Carlos", "Daniel", "Eduarda",
    "Fernanda", "Gabriel", "Helena", "Isabela", "João",
    "Karen", "Leonardo", "Mariana", "Nathalia", "Otávio",
    "Pedro", "Quintino", "Renata", "Sofia", "Thiago",
    "Ursula", "Vinícius", "Wagner", "Xavier", "Yasmin",
    "Zuleica", "André", "Beatriz", "Caio", "Diana",
    "Erick", "Fabiana", "Henrique", "Iara", "Juliana",
    "Lara", "Maurício", "Nicolas", "Olívia", "Patrícia",
    "Quésia", "Raquel", "Samira", "Tatiane", "Valéria"
]

request = input("Qual aluno você deseja saber se estava na aula: ")
encontrado = False

for aluno in alunos:
    if aluno == request:
        encontrado = True
        break

if encontrado:
    print("O aluno estava na aula")
else:
    print("O aluno não estava na aula")