import numpy as np

def generate_bundle_sizes(num_messages, dist_params, seed=None):
    """Generates bundle sizes (number of topics per message).
    
    Args:
        num_messages (int): Number of messages to generate.
        dist_params (dict): Parameters for distribution (e.g., {'type': 'fixed', 'value': 5}).
        seed (int): Random seed.
        
    Returns:
        np.ndarray: Array of bundle sizes.
    """
    rng = np.random.default_rng(seed)
    dist_type = dist_params.get('type', 'fixed')
    
    if dist_type == 'fixed':
        val = dist_params.get('value', 5)
        return np.full(num_messages, val)
        
    elif dist_type == 'geometric':
        p = dist_params.get('p', 0.2)
        # Geometric in numpy is number of trials to success, we usually want size >= 1
        # so we sample and add 1, or use geometric directly if mapped correctly.
        # numpy geometric: P(X=k) = (1-p)^(k-1) * p, k >= 1.
        return rng.geometric(p, size=num_messages)
        
    elif dist_type == 'lognormal':
        mean = dist_params.get('mean', 1.0)
        sigma = dist_params.get('sigma', 0.5)
        val = rng.lognormal(mean, sigma, size=num_messages)
        return np.maximum(1, np.round(val)).astype(int)
        
    elif dist_type == 'powerlaw':
        a = dist_params.get('a', 2.5) # exponent
        # using zipf or pareto
        val = rng.zipf(a, size=num_messages)
        return val
    
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

def sample_topics(num_messages, bundle_sizes, num_topics, dist_params, seed=None):
    """Samples topic indices for each message.
    
    Args:
        num_messages (int): Number of messages.
        bundle_sizes (np.ndarray): Array of bundle sizes.
        num_topics (int): Total number of available topics (d).
        dist_params (dict): Parameters for topic frequency (e.g., {'type': 'uniform'}).
        seed (int): Random seed.
        
    Returns:
        list of np.ndarray: List where each element is an array of topic indices for a message.
    """
    rng = np.random.default_rng(seed)
    dist_type = dist_params.get('type', 'uniform')
    
    weights = None
    if dist_type == 'uniform':
        p = np.full(num_topics, 1.0 / num_topics)
    elif dist_type == 'zipf':
        a = dist_params.get('a', 1.5)
        # Zipf weights: 1/k^a
        ranks = np.arange(1, num_topics + 1)
        weights = 1.0 / np.power(ranks, a)
        p = weights / np.sum(weights)
    else:
        raise ValueError(f"Unknown topic distribution type: {dist_type}")
        
    topics = []
    for b in bundle_sizes:
        # We assume replacement=False for topics within a single bundle usually,
        # but replacement=True is also possible. Assuming No Replacement within bundle
        # ensures distinct topics per message.
        # If b > num_topics, we must allow replacement or cap b.
        actual_b = min(b, num_topics)
        t = rng.choice(num_topics, size=actual_b, replace=False, p=p)
        topics.append(t)
        
    return topics

def generate_intensities(num_messages, bundle_sizes, policies=None, seed=None):
    """Generates intensities for topics in messages.
    
    If policies are provided (for agents), we could generate relative intensities, but
    usually generation is independent of receiver.
    We generate intensities in [0, 1].
    
    Args:
        num_messages (int): Number of messages.
        bundle_sizes (np.ndarray): Array of bundle sizes.
        seed (int): Random seed.
        
    Returns:
        list of np.ndarray: List of intensity arrays.
    """
    rng = np.random.default_rng(seed)
    intensities = []
    for b in bundle_sizes:
        # Uniform intensities in [0, 1] usually, or beta
        intensities.append(rng.random(b))
    return intensities

def generate_unsafe_labels(num_messages, bundle_sizes, rate, seed=None):
    """Generates unsafe boolean labels for each topic in the bundles.
    
    Args:
        num_messages (int): Number of messages.
        bundle_sizes (np.ndarray): Array of bundle sizes.
        rate (float): Probability that a topic is unsafe.
        seed (int): Random seed.
        
    Returns:
        list of np.ndarray: List of boolean arrays (True = unsafe).
    """
    rng = np.random.default_rng(seed)
    labels = []
    for b in bundle_sizes:
        labels.append(rng.random(b) < rate)
    return labels

class MessageFactory:
    """Helper to create full message objects."""
    
    def __init__(self, d, bundle_dist, topic_dist, unsafe_rate=0.0):
        self.d = d
        self.bundle_dist = bundle_dist
        self.topic_dist = topic_dist
        self.unsafe_rate = unsafe_rate
        
    def create_messages(self, n, seed=None):
        rng = np.random.default_rng(seed)
        
        # 1. Bundle sizes
        sizes = generate_bundle_sizes(n, self.bundle_dist, seed=rng.integers(1e9))
        
        # Guard: Cap sizes at dimension d
        sizes = np.minimum(sizes, self.d)
        
        # 2. Topics
        topics = sample_topics(n, sizes, self.d, self.topic_dist, seed=rng.integers(1e9))
        
        # 3. Intensities
        intensities = generate_intensities(n, sizes, seed=rng.integers(1e9))
        
        # 4. Unsafe labels
        unsafe_flags = generate_unsafe_labels(n, sizes, self.unsafe_rate, seed=rng.integers(1e9))
        
        messages = []
        for i in range(n):
            messages.append({
                'topics': topics[i],
                'intensities': intensities[i],
                'unsafe': unsafe_flags[i]
            })
        return messages
