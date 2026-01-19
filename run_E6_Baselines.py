import os
import shutil
import argparse
from experiments.runner import run_sweep, save_results
from experiments.configs import get_experiment_configs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='E6', help='Experiment ID to run')
    args = parser.parse_args()

    print(f"Running Experiment {args.experiment}...")
    configs = get_experiment_configs(args.experiment)
    print(f"Total configs: {len(configs)}")
    
    # Run
    results = run_sweep(configs, n_jobs=4)
    
    # Save
    os.makedirs('experiments/results', exist_ok=True)
    filename = f'experiments/results/{args.experiment}_results.parquet'
    save_results(results, filename)
    print("Done.")
