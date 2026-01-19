import os
import pandas as pd
from experiments.runner import run_sweep, save_results
from experiments.configs import get_experiment_configs
from analysis.plotting import plot_all, load_data

def run_all_experiments():
    print("Starting Full Experiment Suite (E1-E7)...")
    
    all_results = []
    experiment_ids = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7']
    
    os.makedirs('experiments/results', exist_ok=True)
    
    for exp_id in experiment_ids:
        print(f"--- Running {exp_id} ---")
        configs = get_experiment_configs(exp_id)
        if not configs:
            print(f"No configs for {exp_id}, skipping.")
            continue
            
        print(f"  > {len(configs)} configurations.")
        results = run_sweep(configs, n_jobs=4)
        
        # Tag results with experiment ID
        for r in results:
            r['experiment_id'] = exp_id
            
        all_results.extend(results)
        
        # Save intermediate
        filename = f'experiments/results/{exp_id}_results.parquet'
        save_results(results, filename)
        
    # Save combined
    print("--- Saving Combined Results ---")
    filename_all = 'experiments/results/all_results.parquet'
    save_results(all_results, filename_all)
    
    # Plot
    print("--- Generating Plots ---")
    try:
        df = load_data(filename_all)
        plot_all(df, 'experiments/results')
        print("Plots generated in experiments/results/")
    except Exception as e:
        print(f"Plotting failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_experiments()
