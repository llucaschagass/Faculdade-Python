import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

# (v, u, w)
edges = [
    ('A', 'B', 5),
    ('B', 'C', 6),
    ('A', 'D', 7),
    ('C', 'E', 3),
    ('D','B', 3),
    ('D', 'E', 1)
]

for edge in edges:
  G.add_edge(edge[0], edge[1], weight=edge[2])

pos ={
    'A': (0, 1),
    'B': (2, 1),
    'C': (4, 1),
    'D': (1, 0),
    'E': (3, 0)
}

plt.figure(figsize=(10, 8))
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, font_weight='bold')

labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

plt.show()

bfs = nx.bfs_edges(G,'A')

for edge in bfs:
  print(edge)