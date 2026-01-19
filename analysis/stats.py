import numpy as np
import scipy.stats as stats

def calculate_95ci(data, method='normal'):
    """
    Calculates 95% Confidence Interval for the mean.
    
    Args:
        data (list or np.array): Data points (e.g., from multiple seeds).
        method (str): 'normal' (t-distribution) or 'bootstrap'.
        
    Returns:
        tuple: (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    if n < 2:
        return np.mean(data), np.mean(data), np.mean(data)
        
    mean = np.mean(data)
    sem = stats.sem(data)
    
    if method == 'normal':
        # t-distribution for small samples
        h = sem * stats.t.ppf((1 + 0.95) / 2., n-1)
        return mean, mean - h, mean + h
        
    elif method == 'bootstrap':
        # Bootstrap resampling
        rng = np.random.default_rng(42)
        means = []
        for _ in range(1000):
            sample = rng.choice(data, size=n, replace=True)
            means.append(np.mean(sample))
        return mean, np.percentile(means, 2.5), np.percentile(means, 97.5)

def effect_size_cohens_d(group1, group2):
    """
    Calculates Cohen's d effect size between two groups.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    
    if pooled_std == 0:
        return 0.0
        
    return (mean1 - mean2) / pooled_std
