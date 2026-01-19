import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import ast

def load_data(filename):
    df = pd.read_parquet(filename)
    # Post-process columns
    return normalize_df(df)

def normalize_df(df):
    """Normalize flattened config columns."""
    
    # 1. Theta
    # Try to find strictness value
    # Config keys like 'policies_strictness_dist' might contain "{'type': 'fixed', 'value': X}"
    
    strictness_cols = [c for c in df.columns if 'strictness_dist' in c]
    if strictness_cols:
        col = strictness_cols[0]
        def extract_val(x):
            try:
                if isinstance(x, dict): return float(x['value'])
                if isinstance(x, float): return x
                d = ast.literal_eval(str(x))
                return float(d.get('value', 0))
            except:
                return np.nan
        df['theta'] = df[col].apply(extract_val)
        
    # 2. Bundle Size
    bundle_cols = [c for c in df.columns if 'bundle_dist' in c]
    if bundle_cols:
        col = bundle_cols[0]
        def extract_b(x):
            try:
                if isinstance(x, dict): return int(x['value'])
                if isinstance(x, (int, float)): return int(x)
                d = ast.literal_eval(str(x))
                if d.get('type') == 'fixed': return int(d.get('value', 0))
                return 0 # non-fixed
            except:
                return 0
        df['bundle_size'] = df[col].apply(extract_b)
        
    # 3. Experiment Mode (E6)
    if 'experiment_mode' in df.columns:
        pass # already there
        
    # 4. Graph N (E4)
    # Check graph_n or similar
    n_cols = [c for c in df.columns if 'graph_n' in c or 'graph' in c and 'n' in c] # imprecise
    # Assuming standard flattener: 'graph_n'
    if 'graph_n' in df.columns:
        df['N'] = df['graph_n'].astype(float)
        
    # Ensure numerics
    for metric in ['s_func', 's_struct', 'reach', 'exposure_rate', 'false_refusal_rate', 'theta_star_pred']:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors='coerce')
            
    return df

def plot_phase_transition(df, output_path, suffix=''):
    """Plots S_func vs Theta."""
    if 'theta' not in df.columns:
        print("Skipping phase transition plot: theta undefined")
        return
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='theta', y='s_func', label='S_func', marker='o')
    if 's_struct' in df.columns:
        sns.lineplot(data=df, x='theta', y='s_struct', label='S_struct', linestyle='--')
        
    plt.xlabel('Strictness (Theta)')
    plt.ylabel('Giant Component')
    plt.title(f'Phase Transition {suffix}')
    plt.savefig(os.path.join(output_path, f'phase_transition{suffix}.png'))
    plt.close()

def plot_bundle_amplification(df, output_path):
    """Plots S_func vs Theta for different bundle sizes."""
    if 'bundle_size' not in df.columns or 'theta' not in df.columns:
         return
         
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='theta', y='s_func', hue='bundle_size', marker='o', palette='viridis')
    plt.title('Bundle Amplification')
    plt.savefig(os.path.join(output_path, 'bundle_amplification.png'))
    plt.close()

def plot_fss(df, output_path):
    """Plots S_func vs Theta for different N."""
    if 'N' not in df.columns or 'theta' not in df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='theta', y='s_func', hue='N', marker='o', palette='plasma')
    plt.title('Finite-Size Scaling')
    plt.savefig(os.path.join(output_path, 'finite_size_scaling.png'))
    plt.close()

def plot_frontier(df, output_path):
    """Plots Pareto Frontier: Safety (1-Exposure) vs Connectivity (S_func)."""
    # Or Exposure vs False Refusal?
    # Paper says: Unsafe Exposure vs False Refusal
    if 'exposure_rate' not in df.columns or 'false_refusal_rate' not in df.columns:
        return
        
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='exposure_rate', y='false_refusal_rate', hue='experiment_mode', style='theta', s=100)
    plt.xlabel('Unsafe Exposure Rate')
    plt.ylabel('False Refusal Rate')
    plt.title('Safety Frontier')
    plt.savefig(os.path.join(output_path, 'frontier.png'))
    plt.close()

def plot_all(df, output_path):
    os.makedirs(output_path, exist_ok=True)
    plot_phase_transition(df, output_path, '_all')
    plot_bundle_amplification(df, output_path)
    plot_fss(df, output_path)
    plot_frontier(df, output_path)
