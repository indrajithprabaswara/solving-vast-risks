import numpy as np

def compute_exposure_rate(simulation_stats):
    """
    Unsafe exposure rate: Unsafe topics delivered / total delivered topics.
    
    stats needed:
    - total_unsafe_delivered
    - total_topics_delivered
    """
    unsafe = simulation_stats.get('total_unsafe_delivered', 0)
    total = simulation_stats.get('total_topics_delivered', 0)
    
    if total == 0:
        return 0.0
        
    return unsafe / total

def compute_false_refusal_rate(simulation_stats):
    """
    False refusal rate: Packets with benign content where receiver delivers no benign content.
    
    stats needed:
    - count_benign_packet_refused: Packet had at least one benign topic but was fully refused/blocked.
    - count_benign_packet_sent: Total packets with at least one benign topic.
    """
    refused = simulation_stats.get('count_false_refusals', 0)
    total = simulation_stats.get('count_benign_packets_sent', 0)
    
    if total == 0:
        return 0.0
        
    return refused / total

def compute_benign_utility_retained(simulation_stats):
    """
    Benign utility retained: Benign topics delivered / benign topics sent.
    """
    delivered = simulation_stats.get('total_benign_delivered', 0)
    sent = simulation_stats.get('total_benign_sent', 0)
    
    if sent == 0:
        return 0.0 # Or 1.0? usually 0 if nothing flow
        
    return delivered / sent
