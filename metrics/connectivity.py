import networkx as nx
import numpy as np

def get_giant_component_fraction(graph):
    """Computes the fraction of nodes in the largest connected component."""
    if graph.number_of_nodes() == 0:
        return 0.0
    
    # For directed graphs, we usually look at Weakly Connected Component for "reach" potential
    # OR Strongly Connected Component for mutual reach.
    # Instruction context: "Connectivity / communication health... Giant component..."
    # Usually percolation refers to GC in undirected, or Giant Weak/Strong in directed.
    # Given the bidirectional nature of "connected on paper", we'll default to 
    # largest component for undirected, and largest STRONGLY connected for directed 
    # unless specified otherwise.
    # BUT, typically S_func is about "can information flow?", which in directed is Strong.
    
    if graph.is_directed():
        components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)
    else:
        components = sorted(nx.connected_components(graph), key=len, reverse=True)
        
    if not components:
        return 0.0
        
    return len(components[0]) / graph.number_of_nodes()

def compute_s_struct(graph):
    """Computes S_struct: Giant component fraction of the structural graph."""
    return get_giant_component_fraction(graph)

def compute_s_func(functional_graph):
    """Computes S_func: Giant component fraction of the functional graph."""
    return get_giant_component_fraction(functional_graph)

def compute_reachability(graph, num_sources=50, max_steps=None, seed=None):
    """Computes benign diffusion reach: Fraction of nodes reached from random sources.
    
    Args:
        graph: The graph (functional or structural).
        num_sources: Number of random sources to test.
        max_steps: Bounded steps (optional).
        seed: Random seed.
    """
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes())
    if not nodes:
        return 0.0
        
    n = len(nodes)
    actual_sources = min(num_sources, n)
    sources = rng.choice(nodes, size=actual_sources, replace=False)
    
    total_reached = 0
    
    for source in sources:
        # BFS with depth limit
        if max_steps:
            # Use single_source_shortest_path_length (returns dict target->dist)
            # Filter by dist <= max_steps
            lengths = nx.single_source_shortest_path_length(graph, source, cutoff=max_steps)
            reached_count = len(lengths) # includes source itself
        else:
            # Full reachability
            if graph.is_directed():
                # For directed, descendants
                reached = nx.descendants(graph, source)
                reached_count = len(reached) + 1 # +1 for source
            else:
                # For undirected, size of component containing source
                # (Optimization: could map components first)
                reached = nx.node_connected_component(graph, source)
                reached_count = len(reached)
                
        total_reached += reached_count
        
    # Average fraction relative to N
    return (total_reached / actual_sources) / n

def compute_efficiency(graph):
    """Computes global efficiency.
    
    Warning: Expensive O(N^2). Use sampling for large N.
    """
    # NetworkX has robust efficiency calculation but can be slow
    # For N=10k, this is too slow.
    # Approximating via sampling pairs?
    # Instructions say: "approximate (sampling-based) on functional graph"
    
    n = graph.number_of_nodes()
    if n == 0: 
        return 0.0
    
    # Sample k pairs
    k_pairs = 1000 # Configurable?
    nodes = list(graph.nodes())
    
    rng = np.random.default_rng()
    
    # We need pairs (u, v) where u != v
    # Random sampling
    total_eff = 0.0
    
    for _ in range(k_pairs):
        u, v = rng.choice(nodes, size=2, replace=False)
        try:
            dist = nx.shortest_path_length(graph, u, v)
            total_eff += 1.0 / dist
        except nx.NetworkXNoPath:
            pass
            
    return total_eff / k_pairs
