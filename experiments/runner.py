import os
import argparse
import pandas as pd
import numpy as np
import joblib
from tqdm import tqdm
from datetime import datetime

from graphs.generators import generate_er_graph, generate_ba_graph, generate_ws_graph, generate_sbm_graph
from policies.generator import generate_archetypes, generate_policies, generate_heterogeneous_strictness
from simulation.engine import SimulationEngine
from metrics.connectivity import compute_s_struct, compute_s_func, compute_reachability
from metrics.safety import compute_exposure_rate, compute_false_refusal_rate
from theory.percolation import compute_pc, estimate_ptx_curve, predict_theta_star

def run_single_config(config):
    """Runs a single simulation configuration."""
    seed = config.get('seed', 42)
    rng = np.random.default_rng(seed)
    
    # 1. Generate Graph
    graph_type = config['graph']['type']
    n = config['graph']['n']
    
    if graph_type == 'ER':
        # Derive p from avg_degree if provided
        avg_k = config['graph'].get('avg_degree', 10)
        p = avg_k / (n - 1)
        graph = generate_er_graph(n, p, seed=int(rng.integers(1e9)))
    elif graph_type == 'BA':
        m = config['graph'].get('m', 5)
        graph = generate_ba_graph(n, m, seed=int(rng.integers(1e9)))
    elif graph_type == 'WS':
        k = config['graph'].get('k', 10)
        p_rewire = config['graph'].get('p', 0.1)
        graph = generate_ws_graph(n, k, p_rewire, seed=int(rng.integers(1e9)))
    elif graph_type == 'SBM':
        k_blocks = config['graph'].get('k_blocks', 5)
        p_in = config['graph'].get('p_in', 0.1)
        p_out = config['graph'].get('p_out', 0.01)
        graph = generate_sbm_graph(n, k_blocks, p_in, p_out, seed=int(rng.integers(1e9)))
    elif graph_type == 'ConfigModel':
        avg_k = config['graph'].get('avg_degree', 10)
        from graphs.generators import generate_config_model_graph
        graph = generate_config_model_graph(n, avg_k, seed=int(rng.integers(1e9)))
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Directed Override
    if config['graph'].get('directed', False) and not graph.is_directed():
        from graphs.generators import generate_directed_variant
        graph = generate_directed_variant(graph, seed=int(rng.integers(1e9)))
        
    # 2. Generate Policies
    d = config['policies']['d']
    k_arch = config['policies']['k']
    sigma = config['policies']['sigma']
    archetypes = generate_archetypes(k_arch, d, seed=int(rng.integers(1e9)))
    policies = generate_policies(n, archetypes, sigma, correlation_params=config['policies'].get('correlation'), seed=int(rng.integers(1e9)))
    
    # 3. Generate Strictness
    # Extract degrees properly depending on graph type
    degrees = [d for n, d in graph.degree()]
    
    # Inject degrees into strictness_dist param if needed
    strict_conf = config['policies'].get('strictness_dist', {'type': 'fixed', 'value': 0.5})
    if strict_conf['type'] == 'degree_proportional':
        strict_conf['degrees'] = degrees
        
    strictness = generate_heterogeneous_strictness(n, strict_conf, seed=int(rng.integers(1e9)))
    
    # 4. Theory Prediction (Optional, but required for Analytic Mode)
    theta_star_pred = np.nan
    simulation_mode = config.get('simulation_mode', 'temporal') # Default to temporal if not set
    
    p_tx_func = None

    # Pre-calculate Theory for Analytic Mode OR if requested
    if config.get('compute_theory', False) or simulation_mode == 'analytic':
        pc = compute_pc(graph)
        
        # Policy factory for theory estimator
        def policy_factory():
             idx = rng.choice(k_arch)
             base = archetypes[idx]
             noise = rng.normal(0, sigma, size=d)
             return np.clip(base + noise, 0, 1)
             
        # Message factory
        from messages.generator import MessageFactory
        msg_conf = config['messages']
        msg_factory = MessageFactory(d, msg_conf['bundle_dist'], msg_conf['topic_dist'])
        
        thetas_est, ptx_vals = estimate_ptx_curve(policy_factory, msg_factory, num_samples=1000) # Ensure sufficient samples
        theta_star_pred = predict_theta_star(thetas_est, ptx_vals, pc)
        
        # Create interpolation function for Analytic Snapshot
        # p_tx_vals is for specific thetas.
        def interpolated_p_tx(theta):
            return np.interp(theta, thetas_est, ptx_vals)
            
        p_tx_func = interpolated_p_tx

    # 5. Run Simulation
    engine = SimulationEngine(config)
    engine.setup(graph, policies, strictness)
    
    if simulation_mode == 'analytic':
        if p_tx_func is None:
             # Fallback if compute_theory was somehow false but analytic requested (shouldn't happen with logic above)
             pass 
        engine.run_analytic_snapshot(p_tx_func=p_tx_func)
        functional_threshold = 0.5 # Snapshot edges are 0 or 1. Threshold 0.5 works.
    else:
        # Temporal
        engine.run(steps=config['simulation']['steps'])
        functional_threshold = 0.5 # FIXED: Raised from 0.05 to 0.5 for majority success
    
    # 6. Metrics
    g_func = engine.get_functional_graph(threshold=functional_threshold) 
    
    s_struct = compute_s_struct(graph)
    s_func = compute_s_func(g_func)
    reach = compute_reachability(g_func)
    
    # Compute rates (Note: Analytic snapshot might have empty stats for "packets sent" unless engine populates them pseudo-ly)
    exposure = 0.0
    false_refusal = 0.0
    if simulation_mode == 'temporal':
        exposure = compute_exposure_rate(engine.stats)
        false_refusal = compute_false_refusal_rate(engine.stats)
    
    results = {
        's_struct': s_struct,
        's_func': s_func,
        'reach': reach,
        'theta_star_pred': theta_star_pred,
        'exposure_rate': exposure,
        'false_refusal_rate': false_refusal,
        'config': config 
    }
    
    # Add raw stats for detailed analysis (overhead, etc.)
    for k, v in engine.stats.items():
        results[f"stats_{k}"] = v

    
    return results

def run_sweep(sweep_configs, n_jobs=4):
    """Runs a parameter sweep in parallel."""
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_single_config)(cfg) for cfg in tqdm(sweep_configs)
    )
    return results

def save_results(results, filename):
    """Saves results to Parquet."""
    # Flatten config
    # We need to flatten the dict to store in dataframe nicely
    rows = []
    for res in results:
        row = res.copy()
        conf = row.pop('config')
        # Flatten config keys
        # Simple flattening for depth 2
        for k, v in conf.items():
            if isinstance(v, dict):
                for sk, sv in v.items():
                    row[f"{k}_{sk}"] = str(sv) # store as string to be safe if complex
            else:
                row[k] = v
        rows.append(row)
        
    df = pd.DataFrame(rows)
    df.to_parquet(filename)
    print(f"Saved results to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add args...
    pass
