import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# Adicionando as cidades e as distâncias entre elas
edges = [
    ('Arad', 'Zerind', 75),
    ('Arad', 'Sibiu', 140),
    ('Arad', 'Timisoara', 118),
    ('Zerind', 'Oradea', 71),
    ('Oradea', 'Sibiu', 151),
    ('Timisoara', 'Lugoj', 111),
    ('Lugoj', 'Mehadia', 70),
    ('Mehadia', 'Drobeta', 75),
    ('Drobeta', 'Craiova', 120),
    ('Craiova', 'Pitesti', 138),
    ('Craiova', 'Rimnicu Vilcea', 146),
    ('Rimnicu Vilcea', 'Sibiu', 80),
    ('Rimnicu Vilcea', 'Pitesti', 97),
    ('Sibiu', 'Fagaras', 99),
    ('Fagaras', 'Bucareste', 211),
    ('Pitesti', 'Bucareste', 101),
    ('Bucareste', 'Giurgiu', 90),
    ('Bucareste', 'Urziceni', 85),
    ('Urziceni', 'Hirsova', 98),
    ('Hirsova', 'Eforie', 86),
    ('Urziceni', 'Vaslui', 142),
    ('Vaslui', 'Iasi', 92),
    ('Iasi', 'Neamt', 87)
]

for edge in edges:
  G.add_edge(edge[0], edge[1], weight=edge[2])

# Configurando a posição das cidades para visualização
pos = {
    'Arad': (1, 4),
    'Zerind': (0, 5),
    'Oradea': (0, 6),
    'Sibiu': (2, 5),
    'Timisoara': (1, 3),
    'Lugoj': (2, 2),
    'Mehadia': (2, 1),
    'Drobeta': (1, 0),
    'Craiova': (3, 0),
    'Pitesti': (4, 1),
    'Rimnicu Vilcea': (3, 2),
    'Fagaras': (4, 4),
    'Bucareste': (5, 3),
    'Giurgiu': (5, 2),
    'Urziceni': (6, 3),
    'Hirsova': (7, 3),
    'Eforie': (8, 3),
    'Vaslui': (7, 4),
    'Iasi': (7, 5),
    'Neamt': (6, 5)
}

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

plt.show()

def heuristic(node1, node2):
    x1, y1 = pos[node1]
    x2, y2 = pos[node2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


  
path = nx.astar_path(G, 'Arad', 'Bucareste', heuristic=heuristic, weight='weight')

print("Menhor caminho encontrado:", path)