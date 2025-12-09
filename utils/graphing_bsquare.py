import networkx as nx
import pyvis
from pyvis.network import Network
import webbrowser

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
    # Label nodes with number:
    for node in G.nodes():
        G.nodes[node]['label'] = str(node)
    return G

def visualise_graph(G, title="B-Square Dependency Graph"):
    net = Network(directed=True, notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.write_html("./graphs/cifar10_bsquare_dependency_graph.html")
    webbrowser.open("./graphs/cifar10_bsquare_dependency_graph.html")
    print(f"Graph visualized with title: {title}")

def save_graph(G, filepath):
    net = Network(directed=True, notebook=False)
    net.from_nx(G)
    net.show_buttons(filter_=['physics'])
    net.write_html(filepath)
    print(f"Graph saved to {filepath}")
