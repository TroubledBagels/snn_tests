import networkx as nx

def create_dependency_graph(vote_dict):
    # Vote dict of form (c_1, c_2): [c_1_vote, c_2_vote]
    G = nx.DiGraph()
    for key, value in vote_dict.items():
        c1, c2 = key
        c1_vote, c2_vote = value
        if c1_vote > c2_vote:
            G.add_edge(c1, c2)
        elif c2_vote > c1_vote:
            G.add_edge(c2, c1)
        # If votes are equal, no edge is added
    return G