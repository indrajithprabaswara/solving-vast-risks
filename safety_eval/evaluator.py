import numpy as np

def evaluate_safety(message, policy, theta):
    """Evaluates a message against a receiver's policy and strictness theta.
    
    Args:
        message (dict): {'topics': [], 'intensities': [], 'unsafe': []}
        policy (np.ndarray): Agent's policy vector of size d.
        theta (float): Strictness threshold.
        
    Returns:
        dict: Evaluation result containing:
            - 'violations': boolean array, True if item violates policy.
            - 'is_safe': boolean, True if NO violations occur.
            - 'unsafe_content_delivered': boolean array, True if unsafe content is NOT caught (i.e. if it passes filter).
              Note: A violation occurs if intensity > policy[topic] * some_factor??
              
              Let's define the mismatch condition clearly from context.
              "mismatch driver: message intensity relative to receiver policy"
              Usually violation: intensity > policy[topic] OR distance(intensity, policy) > threshold.
              
              Given theta is "strictness", higher theta -> more strict -> easier to reject.
              
              Common model: 
              Refuse if ANY (intensity - policy[topic]) > (1 - theta).
              Or simpler: 
              Refuse if ANY mismatch > threshold.
              
              Let's assume:
              Mismatch_k = |intensity_k - policy[topic_k]|  (or just intensity if policy is acceptance threshold).
              
              Re-reading: "A receiver applies a safety filter with a tunable strictness theta".
              "Even if agents are almost aligned per topic..."
              
              Let's standardize on:
              Violation_k = True if (Intensity_k > Policy[topic_k]) ?? 
              
              Wait, if Policy is "Safety Policy", maybe Policy[k] is the SAFE LIMIT.
              So if Intensity > Policy[k], it's unsafe?
              
              "Strictness theta" usually modulates the filter.
              Maybe Effective_Limit = Policy[k] + (1-theta)?
              Or Violation if Mismatch > (1 - theta).
              
              Let's go with the alignment/mismatch model often used in opinion dynamics but adapted:
              Mismatch = abs(Intensity - Policy[Topic])  (Assuming Intensity and Policy are positions in [0,1])
              Allowed if Mismatch <= (1 - theta).
              So Violation if Mismatch > (1 - theta).
              
              This fits "strictness theta":
              theta=1 => Mismatch must be <= 0 (Perfect alignment needed).
              theta=0 => Mismatch must be <= 1 (Anything goes).
              
              This works well.
    """
    topics = message['topics']
    intensities = message['intensities']
    
    if len(topics) == 0:
        return {
            'violations': np.array([], dtype=bool),
            'mismatch_scores': np.array([], dtype=float),
            'has_violations': False
        }
        
    # Get relevant policy values
    receiver_vals = policy[topics]
    
    # Calculate mismatch (absolute difference)
    mismatch_scores = np.abs(intensities - receiver_vals)
    
    # Violation condition
    # strictness theta makes the tolerance (1 - theta) smaller.
    tolerance = 1.0 - theta
    violations = mismatch_scores > tolerance
    
    return {
        'violations': violations,
        'mismatch_scores': mismatch_scores,
        'has_violations': np.any(violations)
    }

def packet_risk_diagnostics(message, policy):
    """Calculates potential risk metrics without applying a specific theta."""
    topics = message['topics']
    intensities = message['intensities']
    
    if len(topics) == 0:
        return 0.0, 0.0
        
    receiver_vals = policy[topics]
    mismatch_scores = np.abs(intensities - receiver_vals)
    
    return np.mean(mismatch_scores), np.max(mismatch_scores)
