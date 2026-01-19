"""
run_smoke_test.py

This script performs a minimal execution of the simulation engine to verify
that the environment is correctly set up and all dependencies are installed.

It runs a tiny instance of the Phase Transition experiment (E1) with:
- N=100 agents
- 1 Trial
- 3 Theta points

Expected Runtime: < 10 seconds.
"""

import os
import sys
import pandas as pd

# Ensure we can import local modules
# In standard Python usage, current dir is in path, but we enforce it for safety
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from experiments.runner import run_sweep
except ImportError as e:
    print(f"FAIL: Could not import experiment modules. Check dependencies.\nError: {e}")
    sys.exit(1)

def run_smoke_test():
    print("--- Starting Smoke Test ---")
    
    # Define a tiny configuration matching runner.py requirements
    smoke_config = {
        'graph': {
            'type': 'ER',
            'n': 100,
            'avg_degree': 5,
            'directed': True
        },
        'policies': {
            'd': 10,
            'k': 5,          # Number of archetypes
            'sigma': 0.1,    # Noise
            'strictness_dist': {'type': 'fixed', 'value': 0.5},
            'correlation': None
        },
        'messages': {
            'bundle_dist': {'type': 'fixed', 'value': 1},
            'topic_dist': {'type': 'uniform'},
            'unsafe_rate': 0.0
        },
        'simulation': {
            'steps': 2 
        },
        'mitigations': [],
        'experiment_id': 'SMOKE_TEST',
        'seed': 42,
        'compute_theory': False,
        'simulation_mode': 'temporal'
    }
    
    # Wrap in list as run_sweep expects a list of configs
    configs = [smoke_config]
    
    print("Running minimal simulation...")
    try:
        results = run_sweep(configs, n_jobs=1) # Run serial
    except Exception as e:
        print(f"FAIL: Simulation crashed.\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    if not results:
        print("FAIL: No results returned.")
        sys.exit(1)
        
    print(f"Success. Generated {len(results)} data points.")
    
    print("--- SMOKE TEST PASSED ---")

if __name__ == "__main__":
    run_smoke_test()
