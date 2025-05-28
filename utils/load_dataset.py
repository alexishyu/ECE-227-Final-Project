import networkx as nx

def load_epinions(path, directed=True):
    """
    Load the SNAP soc-sign-epinions dataset into a directed or undirected NetworkX graph.
    Each edge gets a 'sign' attribute of +1 (trust) or -1 (distrust).
    """
    G = nx.DiGraph() if directed else nx.Graph()
    with open(path, 'r') as f:
        for line in f:
            # skip comment or header lines
            if line.startswith('#') or line.strip() == "":
                continue
            src, dst, sign = line.split()
            src, dst, sign = int(src), int(dst), int(sign)
            if directed:
                G.add_edge(src, dst, sign=sign)
            else:
                G.add_edge(src, dst) # Not saving sign for undirected graph
    return G

# Example usage:
if __name__ == "__main__":
    dataset_path = "data/epinion/soc-sign-epinions.txt"

    G = load_epinions(dataset_path, directed=True)
    print(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    # Inspect a few edges:
    for u, v, data in list(G.edges(data=True))[:5]:
        if data['sign'] > 0:
            print(f"{u} → {v}  Trust")
        else:
            print(f"{u} → {v}  Distrust")
