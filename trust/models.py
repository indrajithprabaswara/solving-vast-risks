import numpy as np

class TrustModel:
    def update(self, current_trust, interaction_result):
        raise NotImplementedError

class LinearTrustModel(TrustModel):
    def __init__(self, increment=0.01, decrement=0.05, min_trust=0.0, max_trust=1.0):
        self.increment = increment
        self.decrement = decrement
        self.min_trust = min_trust
        self.max_trust = max_trust
        
    def update(self, current_trust, interaction_result):
        """
        interaction_result: 'success' (accept/repair), 'failure' (refuse/redact), 'neutral'
        """
        if interaction_result == 'success':
            delta = self.increment
        elif interaction_result == 'failure':
            delta = -self.decrement
        else:
            delta = 0.0
            
        new_trust = current_trust + delta
        return np.clip(new_trust, self.min_trust, self.max_trust)

class NonlinearTrustModel(TrustModel):
    """Updates trust based on a logistic-like or decay curve."""
    def __init__(self, alpha=0.1, beta=0.2, drift=-0.001):
        self.alpha = alpha # growth rate
        self.beta = beta   # penalty factor
        self.drift = drift # constant decay
        
    def update(self, current_trust, interaction_result):
        delta = self.drift
        if interaction_result == 'success':
            # Trust grows slower as it gets higher? Or constant boost?
            # Simple nonlinear: delta += alpha * (1 - current_trust)
            delta += self.alpha * (1.0 - current_trust)
        elif interaction_result == 'failure':
            # Trust drops proportional to current trust?
            delta -= self.beta * current_trust
            
        new_trust = current_trust + delta
        return np.clip(new_trust, 0.0, 1.0)
        
def should_block(trust_score, threshold=0.1):
    return trust_score < threshold
