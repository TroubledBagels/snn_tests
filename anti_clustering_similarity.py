import numpy as np
import networkx as nx

similarities = {
    (0, 1): 0.2,
    (0, 2): 0.46,
    (0, 3): 0.28,
    (0, 4): 0.39,
    (0, 5): 0.28,
    (0, 6): 0.24,
    (0, 7): 0.32,
    (0, 8): 0.58,
    (0, 9): 0.22,
    (1, 2): 0.0,
    (1, 3): 0.08,
    (1, 4): 0.01,
    (1, 5): 0.04,
    (1, 6): 0.05,
    (1, 7): 0.11,
    (1, 8): 0.15,
    (1, 9): 0.74,
    (2, 3): 0.74,
    (2, 4): 0.82,
    (2, 5): 0.74,
    (2, 6): 0.74,
    (2, 7): 0.63,
    (2, 8): 0.26,
    (2, 9): 0.09,
    (3, 4): 0.77,
    (3, 5): 0.87,
    (3, 6): 0.70,
    (3, 7): 0.72,
    (3, 8): 0.15,
    (3, 9): 0.21,
    (4, 5): 0.77,
    (4, 6): 0.77,
    (4, 7): 0.80,
    (4, 8): 0.20,
    (4, 9): 0.17,
    (5, 6): 0.62,
    (5, 7): 0.79,
    (5, 8): 0.14,
    (5, 9): 0.20,
    (6, 7): 0.43,
    (6, 8): 0.11,
    (6, 9): 0.11,
    (7, 8): 0.13,
    (7, 9): 0.37,
    (8, 9): 0.17
}

class_size = 5

G = nx.Graph()
for (i, j), sim in similarities.items():
    G.add_edge(i, j, weight=sim)

all_sets = []

for node in G.nodes():
    current_set = []
    current_set.append(node)
    for i in range(class_size-1):
        neighbours = [nbr for nbr in G.neighbors(node) if nbr not in current_set]
        if not neighbours:
            break
        best_nbr = None
        lowest_sim = np.inf
        for nbr in neighbours:
            nbr_weight = 0
            for other in current_set:
                if G.has_edge(nbr, other):
                    if nbr > other:
                        edge = (other, nbr)
                    else:
                        edge = (nbr, other)
                    nbr_weight += G[edge[0]][edge[1]]['weight']
            if nbr_weight < lowest_sim:
                lowest_sim = nbr_weight
                best_nbr = nbr
        if best_nbr is not None:
            current_set.append(best_nbr)
    print(f"Class starting with node {node}: {current_set}")
    all_sets.append(current_set)

# Find distribution of nodes in sets
node_count = {i: 0 for i in G.nodes()}
for s in all_sets:
    for node in s:
        node_count[node] += 1
print("Node distribution across sets:")
for node, count in node_count.items():
    print(f"Node {node}: {count} times")


class_sims = [
    [0, 0.2, 0.46, 0.28, 0.39, 0.28, 0.24, 0.32, 0.58, 0.22],
    [0.2, 0, 0.0, 0.08, 0.01, 0.04, 0.05, 0.11, 0.15, 0.74],
    [0.46, 0.0, 0, 0.74, 0.82, 0.74, 0.74, 0.63, 0.26, 0.09],
    [0.28, 0.08, 0.74, 0, 0.77, 0.87, 0.70, 0.72, 0.15, 0.21],
    [0.39, 0.01, 0.82, 0.77, 0, 0.77, 0.77, 0.80, 0.20, 0.17],
    [0.28, 0.04, 0.74, 0.87, 0.77, 0, 0.62, 0.79, 0.14, 0.20],
    [0.24, 0.05, 0.74, 0.70, 0.77, 0.62, 0, 0.43, 0.11, 0.11],
    [0.32, 0.11, 0.63, 0.72, 0.80, 0.79, 0.43, 0, 0.13, 0.37],
    [0.58, 0.15, 0.26, 0.15, 0.20, 0.14, 0.11, 0.13, 0, 0.17],
    [0.22, 0.74, 0.09, 0.21, 0.17, 0.20, 0.11, 0.37, 0.17, 0]
]

class_usage = [0 for _ in range(len(class_sims))]

k = 5
all_classes = []
for start_node in range(len(class_sims)):
    current_set = [start_node]
    for _ in range(k-1):
        lowest_sim = np.inf
        best_node = None
        for candidate in range(len(class_sims)):
            if candidate in current_set:
                continue
            sim_sum = sum(class_sims[candidate][other] for other in current_set) * (class_usage[candidate] + 1)
            if sim_sum < lowest_sim:
                lowest_sim = sim_sum
                best_node = candidate
        if best_node is not None:
            current_set.append(best_node)
            class_usage[best_node] += 1
    print(f"Class starting with node {start_node}: {current_set}")
    all_classes.append(set(current_set))

pruned_classes = []
for cls in all_classes:
    if cls not in pruned_classes:
        pruned_classes.append(cls)

print("Pruned Classes:")
for cls in pruned_classes:
    # Calculate intra-class similarity
    intra_sim = 0
    count = 0
    cls_list = list(cls)
    for i in range(len(cls_list)):
        for j in range(i + 1, len(cls_list)):
            intra_sim += class_sims[cls_list[i]][cls_list[j]]
            count += 1
    intra_sim /= count
    print(f"Class: {cls}, Intra-class similarity: {intra_sim:.4f}")
# Find distribution of nodes in pruned classes
node_count = {i: 0 for i in range(len(class_sims))}
for s in pruned_classes:
    for node in s:
        node_count[node] += 1
print("Node distribution across pruned classes:")
for node, count in node_count.items():
    print(f"Node {node}: {count} times")
