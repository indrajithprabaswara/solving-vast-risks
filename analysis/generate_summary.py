import pandas as pd
import numpy as np
import os
import json
import ast

RESULTS_DIR = r'experiments/results'
OUTPUT_FILE = r'experiments/results/summary.json'

def load_data(filepath):
    if not os.path.exists(filepath): return None
    df = pd.read_parquet(filepath)
    # Norm theta
    col_theta = 'policies_strictness_dist'
    if col_theta not in df.columns: col_theta = 'policies.strictness_dist'
    if col_theta in df.columns:
        def get_theta(x):
            try:
                if isinstance(x, dict): return float(x.get('value', 0))
                val_str = str(x)
                if '{' in val_str:
                    d = ast.literal_eval(val_str)
                    return float(d.get('value', 0))
                return float(x)
            except: return np.nan
        df['theta'] = df[col_theta].apply(get_theta)
    return df

def get_theta_star(df):
    # Find theta where s_func < 0.5 (first time)
    # Assumes df sorted by theta
    stats = df.groupby('theta')['s_func'].mean().sort_index()
    for t, v in stats.items():
        if v < 0.5: return t
    return 1.0

def generate_json():
    summary = {}
    
    # E2: Bundle Amp
    df_e2 = load_data(os.path.join(RESULTS_DIR, 'E2_paper_results.parquet'))
    if df_e2 is not None and 'messages_bundle_dist' in df_e2.columns:
        # Extract value from stringified dict
        def get_bundle_val(x):
            try:
                if isinstance(x, dict): return int(x.get('value', 1))
                val_str = str(x)
                if '{' in val_str:
                    d = ast.literal_eval(val_str)
                    return int(d.get('value', 1))
                return int(x)
            except: return 1
        df_e2['bundle_val'] = df_e2['messages_bundle_dist'].apply(get_bundle_val)
        
        summary['E2'] = {}
        for b in sorted(df_e2['bundle_val'].unique()):
            sub = df_e2[df_e2['bundle_val'] == b]
            th_star = get_theta_star(sub)
            summary['E2'][f'bundle_{b}'] = th_star
            
    # E8: Chunking Overhead
    df_e8 = load_data(os.path.join(RESULTS_DIR, 'E8_paper_results.parquet'))
    if df_e8 is not None:
        summary['E8'] = {}
        if 'stats_total_packets_sent' in df_e8.columns:
             for mode in df_e8['experiment_mode'].unique():
                 sub = df_e8[df_e8['experiment_mode'] == mode]
                 overhead = sub['stats_total_packets_sent'].sum() / (sub['stats_count_benign_packets_sent'].sum() + 1e-9)
                 summary['E8'][f'{mode}_overhead'] = round(overhead, 2)
                 
    # E7: Robustness
    df_e7 = load_data(os.path.join(RESULTS_DIR, 'E7_paper_results.parquet'))
    if df_e7 is not None and 'robustness_axis' in df_e7.columns:
        summary['E7'] = []
        for ax in df_e7['robustness_axis'].unique():
            sub = df_e7[df_e7['robustness_axis'] == ax]
            th_star = get_theta_star(sub)
            summary['E7'].append({'axis': ax, 'theta_star': th_star})

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(summary, f, indent=2)
    print("Summary generated.")

if __name__ == "__main__":
    generate_json()
