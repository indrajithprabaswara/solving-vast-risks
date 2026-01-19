from enum import Enum, auto
import numpy as np

class ActionType(Enum):
    ACCEPT = auto()
    REFUSE = auto()
    REDACT = auto()
    REPAIR = auto()
    block = auto() # using lowercase to avoid conflict if any (though auto() handles it)
    BLOCK = block

class ActionDecision:
    def __init__(self, action_type, delivered_message=None, metadata=None):
        self.action_type = action_type
        self.delivered_message = delivered_message
        self.metadata = metadata or {}

def decide_action(safety_result, policy_type='refuse', repair_config=None, rng=None):
    """Decides the action based on safety evaluation and policy.
    
    Args:
        safety_result (dict): Output from evaluate_safety.
        policy_type (str): 'refuse', 'redact', 'repair'.
        repair_config (dict): Params for repair (e.g., {'p_repair': 0.5}).
        rng (np.random.Generator): Random number generator.
        
    Returns:
        ActionDecision
    """
    if rng is None:
        rng = np.random.default_rng()
        
    violations = safety_result['violations']
    has_violations = safety_result['has_violations']
    
    # Baseline: No violations -> ACCEPT
    if not has_violations:
        return ActionDecision(ActionType.ACCEPT)
        
    # Handling Violations
    if policy_type == 'refuse':
        return ActionDecision(ActionType.REFUSE)
        
    elif policy_type == 'redact':
        # Drop violating topics
        # We need to return which indices are kept.
        # Ideally, we return a "mask" or the filtered message structure?
        # The simulation engine will need to know WHAT was delivered to compute utility.
        # We'll return the keep_mask in metadata.
        keep_mask = ~violations
        
        # If everything is redacted, it's effectively a REFUSE (or empty delivery)
        if not np.any(keep_mask):
             return ActionDecision(ActionType.REFUSE, metadata={'reason': 'all_redacted'})
             
        return ActionDecision(ActionType.REDACT, metadata={'keep_mask': keep_mask})
        
    elif policy_type == 'repair':
        # Probabilistic repair
        p_repair = repair_config.get('p_repair', 0.5) if repair_config else 0.5
        if rng.random() < p_repair:
            return ActionDecision(ActionType.REPAIR) # Implies repaired and accepted
        else:
             return ActionDecision(ActionType.REFUSE, metadata={'reason': 'repair_failed'})
             
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")
