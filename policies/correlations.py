import numpy as np

def generate_topic_covariance(structure, d, params=None):
    """Generates a covariance matrix for topic correlations.
    
    Args:
        structure (str): 'identity', 'block', or 'toeplitz'.
        d (int): Dimension (number of topics).
        params (dict): Structure-specific parameters.
        
    Returns:
        np.ndarray: d x d covariance matrix.
    """
    if params is None:
        params = {}
        
    if structure == 'identity':
        return np.eye(d)
    
    elif structure == 'block':
        # items in the same block have correlation rho
        rho = params.get('rho', 0.5)
        block_size = params.get('block_size', 10)
        cov = np.eye(d)
        for i in range(0, d, block_size):
            end = min(i + block_size, d)
            block = np.full((end - i, end - i), rho)
            np.fill_diagonal(block, 1.0)
            cov[i:end, i:end] = block
        return cov
        
    elif structure == 'toeplitz':
        # correlation decays with distance: rho^|i-j|
        rho = params.get('rho', 0.5)
        indices = np.arange(d)
        distances = np.abs(indices[:, None] - indices[None, :])
        return np.power(rho, distances)
        
    else:
        raise ValueError(f"Unknown correlation structure: {structure}")
