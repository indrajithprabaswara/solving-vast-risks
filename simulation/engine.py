import numpy as np
import networkx as nx
from collections import defaultdict

from messages.generator import MessageFactory
from safety_eval.evaluator import evaluate_safety
from actions.core import decide_action, ActionType
from actions.mitigations import adaptive_chunking_mitigation
from trust.models import should_block

class SimulationEngine:
    def __init__(self, config):
        self.config = config
        self.rng = np.random.default_rng(config.get('seed', 42))
        
        # State
        self.graph = None
        self.policies = None
        self.strictness = None
        self.agents_trust_scores = defaultdict(lambda: defaultdict(float)) # u -> v -> trust
        self.edge_success_rates = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        self.stats = defaultdict(int)
        self.history = []
        
        # Components
        self.message_factory = None
        
    def setup(self, graph, policies, strictness):
        self.graph = graph
        self.policies = policies
        self.strictness = strictness
        
        # Initialize Trust
        trust_init = self.config.get('trust_init', 0.5)
        for u, v in self.graph.edges():
            self.agents_trust_scores[u][v] = trust_init
            self.agents_trust_scores[v][u] = trust_init
            
        # Message Factory
        msg_conf = self.config['messages']
        self.message_factory = MessageFactory(
            d=self.policies.shape[1],
            bundle_dist=msg_conf['bundle_dist'],
            topic_dist=msg_conf['topic_dist'],
            unsafe_rate=msg_conf.get('unsafe_rate', 0.0)
        )
        
    def run_analytic_snapshot(self, p_tx_func=None):
        """
        Generates a functional graph snapshot using analytic probabilities.
        Iterates only over existing structural edges.
        """
        # Reset edge success rates to be populated deterministically/probabilistic snapshot
        self.edge_success_rates = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        
        # Check if graph is directed or undirected base
        is_directed_base = self.graph.is_directed()
        edges = list(self.graph.edges())
        
        # Precompute or use p_tx function
        # If p_tx_func is None, assume default theoretical model: (1 - theta^2)^B
        # Only valid for uniform policies and independent topics
        
        for u, v in edges:
            # Determine edges to test
            pairs_to_test = [(u, v)]
            if not is_directed_base:
                # If structure is undirected, functional link can exist both ways independently
                pairs_to_test.append((v, u))
                
            for src, dst in pairs_to_test:
                receiver_theta = self.strictness[dst]
                
                # Calculate p_tx
                if p_tx_func:
                    p = p_tx_func(receiver_theta)
                else:
                    # Fallback default analytic
                    # Check bundle size distribution? Assume fixed if fallback used.
                    # We access config to see B?
                    # Safer to require p_tx_func for complex cases, but fallback for safety:
                    # Assume B=5 default or config value
                    b_val = 5
                    if 'messages' in self.config and 'bundle_dist' in self.config['messages']:
                        dist = self.config['messages']['bundle_dist']
                        if dist.get('type') == 'fixed':
                            b_val = dist.get('value', 5)
                            
                    p = (1 - receiver_theta**2)**b_val
                
                # Clip p
                p = np.clip(p, 0.0, 1.0)
                
                # Sample
                is_open = self.rng.random() < p
                
                # Store
                # We interpret "snapshot" as 1 attempt, 1 success (if open) or 0 (if closed)
                # This works with get_functional_graph if threshold <= 0.99
                self.edge_success_rates[(src, dst)]['attempts'] = 1
                self.edge_success_rates[(src, dst)]['successes'] = 1 if is_open else 0
    
    def run(self, steps=1000, burn_in=100):
        edges = list(self.graph.edges())
        num_edges = len(edges)
        
        for step in range(steps):
            # 1. Select Edge (randomly)
            # Or iterate all edges? "Event scheduler (edge sampling)"
            # Usually random edge per step or N interactions per turn.
            # Let's do random edge sampling.
            u, v = edges[self.rng.integers(num_edges)]
            
            # Direction? If graph is undirected, pick direction.
            if not self.graph.is_directed():
                if self.rng.random() < 0.5:
                    u, v = v, u
            
            # 2. Check Block/Edge Deletion
            # If trust is too low, skip (functionally broken)
            # OR if we are using explicit BLOCK action, edge might be removed from self.graph
            
            # 3. Generate Message from U
            # We generate 1 message
            raw_message_list = self.message_factory.create_messages(1, seed=self.rng.integers(1e9))
            start_message = raw_message_list[0]
            
            # 4. Mitigations (Sender/Broker side)
            messages_to_send = [start_message]
            
            # Check for chunking configuration
            mitigation_list = self.config.get('mitigations', [])
            if 'adaptive_chunking' in mitigation_list:
                 # Determine strategy from config (default to risk_aware if not specified)
                 # We can pass these in config['mitigation_params'] ideally
                 strat = self.config.get('mitigation_params', {}).get('chunking_strategy', 'risk_aware')
                 c_size = self.config.get('mitigation_params', {}).get('chunk_size', 1)
                 
                 # Apply chunking
                 # "U" applies it. U evaluates risk using its own policy or a robust estimator.
                 # We use U's policy as proxy for risk evaluation logic.
                 
                 # Simple risk evaluator: counts unsafe matches against U's policy
                 def sender_risk_evaluator(msg, pol):
                     res = evaluate_safety(msg, pol, theta=self.strictness[u])
                     # Risk = fraction of unsafe topics? or just any violation?
                     # Let's say risk is 1.0 if any violation, else 0.0
                     return 1.0 if res['has_violations'] else 0.0, res
                 
                 messages_to_send = adaptive_chunking_mitigation(
                     start_message, 
                     self.policies[u], 
                     risk_evaluator=sender_risk_evaluator,
                     strategy=strat,
                     chunk_size=c_size
                 )
            
            # Track Transmission Overhead
            self.stats['total_packets_sent'] += len(messages_to_send)
            
            # 5. Delivery & Evaluation at V
            receiver_policy = self.policies[v]
            theta = self.strictness[v]
            
            interaction_benign_delivered = 0
            interaction_unsafe_delivered = 0
            
            for msg in messages_to_send:
                # Process packet
                delivered_topics, delivered_unsafe = self._process_single_message(u, v, msg, receiver_policy, theta)
                
                interaction_benign_delivered += np.sum(~delivered_unsafe)
                interaction_unsafe_delivered += np.sum(delivered_unsafe)

            # Interaction-level Stats
            original_topics = start_message['topics']
            original_unsafe = start_message['unsafe']
            
            # DEBUG
            # print(f"DEBUG: unsafe type={type(original_unsafe)}, dtype={getattr(original_unsafe, 'dtype', 'None')}")
            # print(f"DEBUG: unsafe val={original_unsafe}")
            
            # Ensure bool numpy array
            # Handle lists, object arrays, etc.
            original_unsafe = np.array(original_unsafe, dtype=bool)
            
            has_benign_content = np.any(~original_unsafe)
            
            self.stats['total_benign_sent'] += np.sum(~original_unsafe) # Count from original (don't double count chunks)
            self.stats['total_unsafe_sent'] += np.sum(original_unsafe)
            
            self.stats['total_benign_delivered'] += interaction_benign_delivered
            self.stats['total_unsafe_delivered'] += interaction_unsafe_delivered
            
            # False Refusal: Had benign, but delivered ZERO benign across all chunks
            if has_benign_content and interaction_benign_delivered == 0:
                self.stats['count_false_refusals'] += 1
                
            # Benign Packet Count (Interaction level)
            if has_benign_content:
                self.stats['count_benign_packets_sent'] += 1 # Actually "Interactions"
                
    def _process_single_message(self, u, v, message, policy, theta):
        """Returns (delivered_topics, delivered_unsafe_labels)"""
        # 1. Safety Eval
        safety_res = evaluate_safety(message, policy, theta)
        
        # 2. Action
        policy_type = self.config['policies'].get('action_type', 'refuse')
        decision = decide_action(safety_res, policy_type=policy_type, rng=self.rng)
        
        # 3. Trust Update 
        # (TODO: if blocked, remove edge? For E5 we disabled it)
        
        # 4. Record Success/Fail for Connectivity
        edge_key = (u, v)
        self.edge_success_rates[edge_key]['attempts'] += 1
        
        delivered_topics = np.array([])
        delivered_unsafe = np.array([], dtype=bool)
        
        if decision.action_type in [ActionType.ACCEPT, ActionType.REPAIR]:
            self.edge_success_rates[edge_key]['successes'] += 1
            delivered_topics = np.array(message['topics'])
            delivered_unsafe = np.array(message['unsafe'], dtype=bool)
            
        elif decision.action_type == ActionType.REDACT:
            # Partial success? 
            # Usually counts as success if anything gets through? 
            # Or weighted? Base connectivity metric is binary packet success.
            # If REDACT passes > 0 topics, is it success?
            # Let's say yes for S_func.
            keep_mask = decision.metadata.get('keep_mask')
            if keep_mask is not None and np.any(keep_mask):
                self.edge_success_rates[edge_key]['successes'] += 1
                delivered_topics = np.array(message['topics'])[keep_mask]
                delivered_unsafe = np.array(message['unsafe'], dtype=bool)[keep_mask]
                
        return delivered_topics, delivered_unsafe

    def get_functional_graph(self, threshold=0.5):
        """Builds functional graph based on empirical success rates."""
        g_func = nx.DiGraph() # Always directed func graph?
        if not self.graph.is_directed():
             # If base is undirected, func is still directed (u->v success != v->u success)
             pass
        g_func.add_nodes_from(self.graph.nodes())
        
        for (u, v), data in self.edge_success_rates.items():
            if data['attempts'] > 0:
                rate = data['successes'] / data['attempts']
                if rate >= threshold:
                    g_func.add_edge(u, v)
                    
        return g_func
