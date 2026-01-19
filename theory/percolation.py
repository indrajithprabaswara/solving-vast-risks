import numpy as np
from scipy.optimize import brentq
from graphs.utils import compute_degree_moments
from messages.generator import MessageFactory
from policies.generator import generate_policies
from safety_eval.evaluator import evaluate_safety

def compute_pc(graph):
    """Computes the critical percolation threshold p_c for the graph.
    
    p_c = <k> / (<k^2> - <k>)
    """
    k_mean, k2_mean = compute_degree_moments(graph)
    if k2_mean - k_mean <= 0:
        return 1.0 # Should not happen for connected graphs usually
    return k_mean / (k2_mean - k_mean)

def estimate_ptx_curve(policy_factory_func, msg_factory, num_samples=1000, thetas=None):
    """
    Estimates p_tx(theta) by Monte Carlo simulation.
    
    Args:
        policy_factory_func: Function that returns representative policies (single or set).
        msg_factory: MessageFactory instance.
        num_samples: Number of interactions to simulate per theta.
        thetas: Array of theta values to evaluate.
        
    Returns:
        tuple: (thetas, ptx_values)
    """
    if thetas is None:
        thetas = np.linspace(0, 1, 21)
        
    # Generate representative agents
    # For speed, we might just generate 2 single agents if i.i.d
    # Or a sample of N if heterogeneous.
    # Let's assume we can generate a "receiver" and measure success.
    # But Policy generation depends on N and archetypes.
    # We'll expect the caller to provide a "Receiver Policy Generator" or sample policies.
    
    # Simplified: Generate a dummy "Receiver" from the same distribution as the simulation.
    # We need a callback or wrapper. 
    # Let's assume policy_factory_func returns a policy vector.
    
    ptx_values = []
    
    # Pre-generate messages and policies to speed up
    # We act as a "mean field" estimator.
    
    for theta in thetas:
        successes = 0
        for _ in range(num_samples):
            # 1. Generate Msg
            msgs = msg_factory.create_messages(1)
            msg = msgs[0]
            
            # 2. Generate Receiver Policy
            policy = policy_factory_func()
            
            # 3. Evaluate
            # Assumption: p_tx implies probability of ACCEPT (or safe flow).
            # "Refuse-first: success iff no violated topics"
            res = evaluate_safety(msg, policy, theta)
            if not res['has_violations']:
                successes += 1
                
        ptx_values.append(successes / num_samples)
        
    return thetas, np.array(ptx_values)

def predict_theta_star(thetas, ptx_values, pc):
    """Predicts theta* where p_tx(theta) = pc."""
    # Find intersection using interpolation or root finding
    # p_tx is monotonically decreasing with theta (usually).
    
    # Check bounds
    if np.all(ptx_values > pc):
        return 1.0 # Transition is at max strictness or never?
        # If transmission always > pc, then it's always connected?
        # Or no collapse.
    if np.all(ptx_values < pc):
        return 0.0 # Already collapsed
        
    # Interpolate
    try:
        # We want theta where ptx - pc = 0
        # ptx is y, theta is x.
        # Function f(theta) = interp(theta) - pc
        theta_star = brentq(lambda t: np.interp(t, thetas, ptx_values) - pc, 0, 1)
        return theta_star
    except ValueError:
        return np.nan
