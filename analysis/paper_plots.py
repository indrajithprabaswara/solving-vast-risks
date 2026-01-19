import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    df = pd.read_parquet(filepath)
    
    # helper to normalize columns
    def norm_val(x):
        try:
            if isinstance(x, dict): return float(x.get('value', 0))
            if isinstance(x, str) and '{' in x:
                d = ast.literal_eval(x)
                return float(d.get('value', 0))
            return float(x)
        except: return np.nan
        
    if 'policies_strictness_dist' in df.columns:
        df['theta'] = df['policies_strictness_dist'].apply(norm_val)
    elif 'policies.strictness_dist' in df.columns:
        df['theta'] = df['policies.strictness_dist'].apply(norm_val)
        
    return df

def plot_theory_overlay(df, output_dir):
    """
    Plots Observed Connectivity vs Theory Prediction.
    """
    if df is None: return
    
    # Filter for E3 or meaningful runs
    # We want to show S_func (Observed) and p_tx_est (Theory) on same plot?
    # No, we promised theta* vs theta* scatter.
    # But theta*_obs is hard to extract from 5 points.
    
    # Let's plot Overlay:
    # X: Theta
    # Y: Connectivity
    # Series: S_func (Obs), p_tx_est (Predicted Transmission Probability)
    
    # If p_tx_est is not saved in results, we can't plot curve.
    # runner.py saves 'theta_star_pred'. It doesn't save the full p_tx curve points!
    # Aaargh.
    # But 'theta_star_pred' is the critical point.
    # So we can look for where S_func crosses 0 -- is it near theta_star_pred?
    
    plt.figure(figsize=(8,8))
    
    # We plot S_func vs Theta for various configs.
    # Group by config (excluding theta)
    df_e3 = df.copy()
    
    # helper for bundle val
    def get_bundle_val(x):
        try:
            if isinstance(x, dict): return int(x.get('value', 1))
            val_str = str(x)
            if '{' in val_str:
                d = ast.literal_eval(val_str)
                return int(d.get('value', 1))
            return int(x)
        except: return 1
        
    col_bund = 'messages_bundle_dist'
    if col_bund in df_e3.columns:
        df_e3['bundle_val'] = df_e3[col_bund].apply(get_bundle_val)
    else:
        # try find it
        c = [x for x in df_e3.columns if 'bundle_dist' in x]
        if c:
            df_e3['bundle_val'] = df_e3[c[0]].apply(get_bundle_val)
        else:
            return
    
    # Iterate over unique bundle sizes
    b_sizes = sorted(df_e3['bundle_val'].unique())
    
    colors = sns.color_palette("viridis", len(b_sizes))
    
    for i, b in enumerate(b_sizes):
        subset = df_e3[df_e3['bundle_val'] == b]
        if subset.empty: continue
        
        # Plot S_func curve
        sns.lineplot(data=subset, x='theta', y='s_func', label=f'Observed (B={b})', color=colors[i], marker='o')
        
        # Plot Theory Predicted Theta* (Mean)
        mean_pred = subset['theta_star_pred'].mean()
        if not np.isnan(mean_pred):
            plt.axvline(mean_pred, color=colors[i], linestyle='--', alpha=0.8, label=f'Predicted (B={b})')
            
    plt.title("Theory Validation: Observed Transition vs Predicted Critical Point")
    plt.xlabel("Strictness (Theta)")
    plt.ylabel("Functional S")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'theory_validation_overlay.png'))
    plt.close()

def plot_chunking_results(df, output_dir):
    """Chunking efficacy and overhead."""
    if df is None: return
    if 'experiment_mode' not in df.columns: return
    
    # 1. S_func vs Theta per Mode
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='theta', y='s_func', hue='experiment_mode', marker='o')
    plt.title("Chunking Efficacy: Connectivity vs Strictness")
    plt.ylabel("Functional S (Largest Comp)")
    plt.savefig(os.path.join(output_dir, 'chunking_efficacy.png'))
    plt.close()
    
    # 2. Overhead vs Theta
    # stats_total_packets_sent / stats_count_benign_packets_sent
    # Handle NaNs
    if 'stats_total_packets_sent' in df.columns and 'stats_count_benign_packets_sent' in df.columns:
        df['overhead_ratio'] = df['stats_total_packets_sent'] / df['stats_count_benign_packets_sent'].replace(0, np.nan)
        
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df, x='theta', y='overhead_ratio', hue='experiment_mode', marker='o')
        plt.title("Chunking Overhead: Packets per Benign Interaction")
        plt.ylabel("Packets Sent / Benign Interaction")
        plt.savefig(os.path.join(output_dir, 'chunking_overhead.png'))
        plt.close()
    
def run_analysis():
    res_dir = 'experiments/results'
    os.makedirs(res_dir, exist_ok=True)
    
    # E3: Theory
    df_e3 = load_data(os.path.join(res_dir, 'E3_paper_results.parquet'))
    plot_theory_overlay(df_e3, res_dir)
    
    # E8: Chunking
    df_e8 = load_data(os.path.join(res_dir, 'E8_paper_results.parquet'))
    plot_chunking_results(df_e8, res_dir)
    
    # E7: Robustness
    df_e7 = load_data(os.path.join(res_dir, 'E7_paper_results.parquet'))
    if df_e7 is not None and 'robustness_axis' in df_e7.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_e7, x='theta', y='s_func', hue='robustness_axis', style='robustness_axis', markers=True)
        plt.title("Robustness: Phase Transition across variations")
        plt.savefig(os.path.join(res_dir, 'robustness_summary.png'))
        plt.close()

if __name__ == "__main__":
    run_analysis()
