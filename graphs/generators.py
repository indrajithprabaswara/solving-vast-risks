import networkx as nx
import numpy as np

def generate_er_graph(n, p, directed=False, seed=None):
    """Generates an Erdős-Rényi graph."""
    return nx.erdos_renyi_graph(n, p, directed=directed, seed=seed)

def generate_ba_graph(n, m, seed=None):
    """Generates a Barabási-Albert graph."""
    return nx.barabasi_albert_graph(n, m, seed=seed)

def generate_ws_graph(n, k, p, seed=None):
    """Generates a Watts-Strogatz graph."""
    return nx.watts_strogatz_graph(n, k, p, seed=seed)

def generate_sbm_graph(n, k_blocks, p_in, p_out, seed=None):
    """Generates a Stochastic Block Model graph.
    
    Args:
        n (int): Total number of nodes.
        k_blocks (int): Number of blocks (communities).
        p_in (float): Probability of intra-block edges.
        p_out (float): Probability of inter-block edges.
        seed (int): Random seed.
    """
    sizes = [n // k_blocks] * k_blocks
    # Distribute remainder
    for i in range(n % k_blocks):
        sizes[i] += 1
    
    p_matrix = np.full((k_blocks, k_blocks), p_out)
    np.fill_diagonal(p_matrix, p_in)
    
    return nx.stochastic_block_model(sizes, p_matrix, seed=seed)

def generate_directed_variant(graph, seed=None):
    """Converts an undirected graph to a directed one with random edge directions."""
    rng = np.random.default_rng(seed)
    directed_graph = nx.DiGraph()
    directed_graph.add_nodes_from(graph.nodes())
    
    for u, v in graph.edges():
        if rng.random() < 0.5:
            directed_graph.add_edge(u, v)
        else:
            directed_graph.add_edge(v, u)
            
    return directed_graph

def generate_assortative_graph(n, degree_sequence, correlation_type='degree', seed=None):
    """Generates a graph with specific assortativity properties.
    Note: Direct generation of assortative graphs is complex. 
    Here we use configuration model as base and could use Xulvi-Brunet & Sokolov algorithm 
    if strictly needed, but for now we stick to configuration model which is neutral 
    unless modified.
    
    For this codebase, we will rely on checking assortativity of generated graphs 
    or selecting from generated population if needed, as standard generators 
    don't easily parameterize 'target assortativity' directly without iterative rewiring.
    
    For now, returning a configuration model graph.
    """
    return nx.configuration_model(degree_sequence, seed=seed)

def generate_config_model_graph(n, avg_degree, seed=None):
    """Generates a random graph with same degree sequence as ER but randomized structure."""
    rng = np.random.default_rng(seed)
    # Generate target ER degrees
    # G_er = nx.erdos_renyi_graph(n, avg_degree / (n - 1), seed=seed)
    # degrees = [d for n, d in G_er.degree()]
    
    # Or just Poisson degrees directly
    degrees = rng.poisson(avg_degree, n)
    if sum(degrees) % 2 == 1: degrees[0] += 1
    
    G = nx.configuration_model(degrees, seed=seed)
    G = nx.Graph(G) # Remove multigraph
    G.remove_edges_from(nx.selfloop_edges(G))
    return G
