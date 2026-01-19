import numpy as np

def generate_archetypes(k, d, seed=None):
    """Generates K archetype policy vectors of dimension d."""
    rng = np.random.default_rng(seed)
    return rng.random((k, d))

def generate_policies(n, archetypes, sigma, correlation_params=None, seed=None):
    """Generates N policy vectors, optionally correlated."""
    rng = np.random.default_rng(seed)
    k, d = archetypes.shape
    
    # Base assignment
    assignments = rng.integers(0, k, size=n)
    base_policies = archetypes[assignments]
    
    # Noise structure
    corr_type = correlation_params if isinstance(correlation_params, str) else 'identity'
    if correlation_params and isinstance(correlation_params, dict):
         corr_type = correlation_params.get('type', 'identity')
         
    if corr_type == 'identity' or not corr_type:
        noise = rng.normal(0, sigma, size=(n, d))
    elif corr_type == 'block':
        # Block-diagonal covariance for topics
        noise = np.zeros((n, d))
        block_size = 10
        num_blocks = d // block_size
        for i in range(num_blocks):
             shared = rng.normal(0, sigma, size=(n, 1))
             indep = rng.normal(0, sigma/2, size=(n, block_size))
             noise[:, i*block_size : (i+1)*block_size] += 0.5 * shared + indep
        rem = d % block_size
        if rem > 0:
            noise[:, -rem:] = rng.normal(0, sigma, size=(n, rem))
            
    elif corr_type == 'toeplitz':
        noise = np.zeros((n, d))
        alpha = 0.8
        curr = rng.normal(0, sigma, size=n)
        for j in range(d):
            noise[:, j] = curr
            curr = alpha * curr + rng.normal(0, sigma * np.sqrt(1-alpha**2), size=n)
            
    else:
        noise = rng.normal(0, sigma, size=(n, d))
        
    policies = np.clip(base_policies + noise, 0, 1)
    return policies

def generate_heterogeneous_strictness(n, dist_params, seed=None):
    """Generates heterogeneous strictness values (theta_i) for agents."""
    rng = np.random.default_rng(seed)
    dist_type = dist_params.get('type', 'fixed')
    
    if dist_type == 'fixed':
        val = dist_params.get('value', 0.5)
        return np.full(n, val)
    elif dist_type == 'uniform':
        low = dist_params.get('low', 0.0)
        high = dist_params.get('high', 1.0)
        return rng.uniform(low, high, size=n)
    elif dist_type == 'normal':
        loc = dist_params.get('loc', 0.5)
        scale = dist_params.get('scale', 0.1)
        return np.clip(rng.normal(loc, scale, size=n), 0, 1)
    elif dist_type == 'beta':
        a = dist_params.get('a', 2.0)
        b = dist_params.get('b', 2.0)
        return rng.beta(a, b, size=n)
    elif dist_type == 'degree_proportional':
        # Hubs are stricter (or looser).
        # proportional: theta_i ~ k_i / k_max
        # inverse: theta_i ~ 1 - (k_i / k_max)
        degrees = dist_params.get('degrees')
        if degrees is None:
            raise ValueError("degrees must be provided for degree_proportional strictness")
            
        k = np.array(degrees)
        k_min, k_max = k.min(), k.max()
        
        # Scale to [0, 1]
        if k_max == k_min:
            normalized_k = np.zeros(n)
        else:
            normalized_k = (k - k_min) / (k_max - k_min)
            
        mode = dist_params.get('mode', 'proportional')
        base_theta = dist_params.get('base_theta', 0.5) # Center or scale?
        
        # Simple Linear mapping: theta = low + (high-low) * normalized_k
        low = dist_params.get('low', 0.0)
        high = dist_params.get('high', 1.0)
        
        if mode == 'proportional':
            return low + (high - low) * normalized_k
        elif mode == 'inverse':
            return low + (high - low) * (1.0 - normalized_k)
        else:
            raise ValueError(f"Unknown degree mode: {mode}")
            
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")
