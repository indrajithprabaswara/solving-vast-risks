import numpy as np

def compute_degree_moments(graph):
    """Computes the first and second moments of the degree distribution.
    
    Returns:
        tuple: (mean_degree, mean_squared_degree)
    """
    degrees = [d for n, d in graph.degree()]
    k_mean = np.mean(degrees)
    k2_mean = np.mean(np.array(degrees)**2)
    return k_mean, k2_mean

def get_degree_distribution(graph):
    """Returns the degree distribution of the graph."""
    degrees = [d for n, d in graph.degree()]
    return degrees
