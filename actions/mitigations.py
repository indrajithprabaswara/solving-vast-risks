import numpy as np
from actions.core import ActionType, ActionDecision

def adaptive_chunking_mitigation(message, policy, risk_evaluator=None, strategy='risk_aware', max_risk=0.5, chunk_size=1):
    """
    Splits message into chunks based on strategy.
    
    Strategies:
    - 'max_size': Split into chunks of size `chunk_size`.
    - 'risk_aware': If risk > max_risk, split into single topics (extreme safety).
    """
    topics = message['topics']
    intensities = message['intensities']
    unsafe = message['unsafe']
    n = len(topics)
    
    chunks = []
    
    if strategy == 'fixed_size' or strategy == 'max_size':
        # Split into fixed-size bundles
        chunk_size = max(1, chunk_size)
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            chunks.append({
                'topics': topics[i:end],
                'intensities': intensities[i:end],
                'unsafe': unsafe[i:end]
            })

    elif strategy == 'risk_aware':
        # If risky, atomize. If safe, send whole (or chunk largley).
        risk = 0.0
        if risk_evaluator:
            # Assumes policy is passed, or handled by evaluator closure
            risk, _ = risk_evaluator(message, policy)
            
        if risk > max_risk:
            # High risk -> Atomize (size 1)
            for i in range(n):
                chunks.append({
                    'topics': np.array([topics[i]]),
                    'intensities': np.array([intensities[i]]),
                    'unsafe': np.array([unsafe[i]])
                })
        else:
            # Low risk -> Send as is
            chunks.append(message)
            
    elif strategy == 'random':
        # Random Chunking:
        # With probability p (param), chunk into size 1. Else keep whole.
        prob_chunk = 0.5 # Default
        if 'mitigation_params' in policy and 'probability' in policy['mitigation_params']:
             prob_chunk = policy['mitigation_params']['probability']
             
        import random
        if random.random() < prob_chunk:
             # Chunk it (size 1)
             for i in range(n):
                chunks.append({
                    'topics': np.array([topics[i]]),
                    'intensities': np.array([intensities[i]]),
                    'unsafe': np.array([unsafe[i]])
                })
        else:
             chunks.append(message)
    else:
        # Fallback: Send original
        chunks.append(message)
             
    return chunks

def broker_moderation_mitigation(message, broker_policy, broker_strictness=0.5):
    """
    Simulates a broker node filtering/cleaning the message before delivery.
    
    Args:
        message (dict): The original message.
        broker_policy (np.ndarray): The broker's safety policy.
        broker_strictness (float): Strictness of the broker.
        
    Returns:
        dict: The filtered message (Redact-first logic).
    """
    # Import locally to avoid circular imports if any, though actions.core is safe
    from safety_eval.evaluator import evaluate_safety
    
    # Broker evaluates
    safety_res = evaluate_safety(message, broker_policy, broker_strictness)
    
    if not safety_res['has_violations']:
        return message
        
    # If violations, broker REDACTS them.
    # Broker is usually a "trusted middlebox" that removes harm.
    violations = safety_res['violations']
    topics = message['topics']
    intensities = message['intensities']
    unsafe_labels = message['unsafe']
    
    keep_mask = ~violations
    
    if not np.any(keep_mask):
        # Empty message (dropped)
        return {
            'topics': np.array([]),
            'intensities': np.array([]),
            'unsafe': np.array([])
        }
    
    return {
        'topics': topics[keep_mask],
        'intensities': intensities[keep_mask],
        'unsafe': unsafe_labels[keep_mask]
    }

def adaptive_strictness_control(current_theta, ptx_est, pc_target, margin=0.05, lr=0.01):
    """
    Adjusts strictness theta to maintain p_tx >= p_c + margin.
    
    Control Rule:
    Error = p_tx - (p_c + margin)
    
    If Error > 0 (Safe connectivity excess): 
       We can afford to be stricter? 
       Wait. p_tx decreases as theta increases (stricter).
       If p_tx > target, we are connected enough. We can raise theta (be stricter) to reduce unsafe exposure?
       Or do we want to keep theta as high as possible while satisfying constraint?
       Usually: Maximize Safety (Theta) s.t. p_tx >= p_c + margin.
       
       So if p_tx > target, Increase Theta.
       If p_tx < target, Decrease Theta (must loosen to restore connectivity).
       
    Args:
        current_theta (float): Current strictness.
        ptx_est (float): Estimated effective transmission.
        pc_target (float): Critical threshold of graph.
        margin (float): Safety margin.
        lr (float): Learning rate.
        
    Returns:
        float: New theta.
    """
    error = ptx_est - (pc_target + margin)
    
    # Update
    # If error > 0 (Too connected), we increase theta (stricter).
    # If error < 0 (Disconnected), we decrease theta (looser).
    delta = lr * error
    
    new_theta = current_theta + delta
    return np.clip(new_theta, 0.0, 1.0)

