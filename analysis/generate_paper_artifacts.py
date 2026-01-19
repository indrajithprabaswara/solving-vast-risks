import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ast
import json

# Configuration
RESULTS_DIR = r'experiments/results'
OUTPUT_DIR_SUMMARIES = r'experiments/results/summaries'
OUTPUT_DIR_FIGURES = r'experiments/results/figures' # Using existing flat dir structure technically, but can make subdirs
os.makedirs(OUTPUT_DIR_SUMMARIES, exist_ok=True)
os.makedirs(OUTPUT_DIR_FIGURES, exist_ok=True)

# --- Helpers ---
def load_and_normalize(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    df = pd.read_parquet(filepath)
    
    # helper for stringified dicts
    def parse_col(x, key='value', cast_func=float):
        try:
            if isinstance(x, dict): return cast_func(x.get(key, 0))
            val_str = str(x)
            if '{' in val_str:
                d = ast.literal_eval(val_str)
                return cast_func(d.get(key, 0))
            return cast_func(x)
        except: return np.nan

    # Normalize Theta
    # Priority: policies_target_theta > policies_strictness_dist
    col_target = [c for c in df.columns if 'target_theta' in c]
    col_strict = [c for c in df.columns if 'strictness' in c][0]
    
    if col_target:
        # Use target_theta if available, fillna with parse_col of strictness
        target_vals = df[col_target[0]]
        parsed_vals = df[col_strict].apply(lambda x: parse_col(x, 'value', float))
        df['theta'] = target_vals.fillna(parsed_vals)
        # If target_theta was all NaNs/None for some reason (e.g. not in config), use parsed
        if df['theta'].isna().all():
             df['theta'] = parsed_vals
    else:
        df['theta'] = df[col_strict].apply(lambda x: parse_col(x, 'value', float))
    
    # Normalize Bundle Size
    col_bund = [c for c in df.columns if 'bundle_dist' in c]
    if col_bund:
        df['bundle_val'] = df[col_bund[0]].apply(lambda x: parse_col(x, 'value', int))
        
    return df

def get_theta_star_obs(df_group):
    # Method 1: Peak Susceptibility (if multiple seeds)
    # Susceptibility = N * Var(S_func)
    # But we need var across seeds per theta
    seeds = df_group['iteration'].nunique() if 'iteration' in df_group.columns else 1
    
    stats = df_group.groupby('theta')['s_func'].agg(['mean', 'var'])
    
    if len(stats) < 2:
        return 0.0, 'insufficient_points'

    theta_star = 0.0
    method = 'slope_peak'
    
    try:
        if seeds > 1:
            # N = 1000 usually
            # Chi = 1000 * Var
            # We just need argmax Var
            if 'var' in stats.columns and not stats['var'].isna().all() and stats['var'].sum() > 1e-9:
                 theta_star = stats['var'].idxmax()
                 method = 'chi_peak'
            else:
                 # Fallback
                 grads = np.abs(np.gradient(stats['mean']))
                 theta_star = stats.index[np.argmax(grads)]
                 method = 'slope_peak'
        else:
            # Slope method
            grads = np.abs(np.gradient(stats['mean']))
            theta_star = stats.index[np.argmax(grads)]
            method = 'slope_peak'
    except Exception as e:
        print(f"Error in get_theta_star_obs: {e}")
        return 0.0, 'error'
        
    return theta_star, method

# --- Artifact 1: Theory Validation ---
def generate_theory_artifact():
    print("Generating Artifact 1: Theory Validation...")
    df = load_and_normalize(os.path.join(RESULTS_DIR, 'E3_paper_results.parquet'))
    if df is None: return

    results = []
    
    # Group by condition: Bundle Size
    # In E3, we varied bundle size. Graph is likely fixed (ER). Policy fixed.
    # Group by 'bundle_val'
    
    if 'bundle_val' not in df.columns:
        print("bundle_val missing in E3")
        return

    for b_val, group in df.groupby('bundle_val'):
        print(f"Processing bundle {b_val}...")
        try:
            # Predicted Theta
            theta_pred_mean = group['theta_star_pred'].mean() if 'theta_star_pred' in group.columns else np.nan
            theta_pred_std = group['theta_star_pred'].std() if 'theta_star_pred' in group.columns else 0.0
            
            # Observed Theta
            theta_obs, method = get_theta_star_obs(group)
            
            results.append({
                'experiment': 'E3',
                'bundle_val': b_val,
                'theta_star_pred_mean': theta_pred_mean,
                'theta_star_pred_std': theta_pred_std,
                'theta_star_obs': theta_obs,
                'obs_method': method,
                'error': theta_pred_mean - theta_obs
            })
        except Exception as e:
            print(f"Error processing {b_val}: {e}")
            import traceback
            traceback.print_exc()

        
    res_df = pd.DataFrame(results)
    res_df['abs_error'] = res_df['error'].abs()
    
    # Save Table
    res_df.to_csv(os.path.join(OUTPUT_DIR_SUMMARIES, 'theory_theta_star_table.csv'), index=False)
    
    # Plot
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=res_df, x='theta_star_obs', y='theta_star_pred_mean', hue='bundle_val', s=100, palette='viridis')
    # x=y line
    lims = [0, 1]
    plt.plot(lims, lims, '--k', alpha=0.5, label='Ideal')
    plt.title(f"Predicted vs Observed Theta* (MAE={res_df['abs_error'].mean():.3f})")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR_FIGURES, 'fig_theta_pred_vs_obs.png'))
    plt.close()
    print("Artifact 1 Done.")

# --- Artifact 2: Robustness ---
def generate_robustness_artifact():
    print("Generating Artifact 2: Robustness Completion...")
    # Prioritize completed sweep if it exists
    completed_path = os.path.join(RESULTS_DIR, 'robustness_e7_completed.parquet')
    if os.path.exists(completed_path):
        filepath = completed_path
    else:
        filepath = os.path.join(RESULTS_DIR, 'E7_paper_results.parquet')
        
    df = load_and_normalize(filepath)
    if df is None: return

    if 'robustness_axis' not in df.columns:
        print("Robustness results missing axis column")
        return

    # Aggregate results for all axes
    robustness_results = []
    
    plt.figure(figsize=(10, 6))
    for ax, group in df.groupby('robustness_axis'):
        theta_star, method = get_theta_star_obs(group)
        robustness_results.append({
            'axis': ax,
            'theta_star': theta_star,
            'method': method,
            'min_theta': group['theta'].min(),
            'max_theta': group['theta'].max()
        })
        # Plot S-curves
        sns.lineplot(data=group, x='theta', y='s_func', label=ax, marker='o')

    plt.title('Robustness: Phase Transitions Across Axes')
    plt.xlabel('Theta (Strictness)')
    plt.ylabel('S_func (Connectivity)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_FIGURES, 'fig_e7_robustness_scurves.png'))
    plt.close()

    # Chi peaks plot
    plt.figure(figsize=(10, 6))
    for ax, group in df.groupby('robustness_axis'):
        var_df = group.groupby('theta')['s_func'].var().reset_index()
        var_df.columns = ['theta', 'chi']
        sns.lineplot(data=var_df, x='theta', y='chi', label=ax, marker='s')
    
    plt.title('Robustness: Chi Peaks (Variance)')
    plt.xlabel('Theta')
    plt.ylabel('Susceptibility Proxy (Chi)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR_FIGURES, 'fig_e7_chi_peaks.png'))
    plt.close()

    res_df = pd.DataFrame(robustness_results)
    res_df.to_csv(os.path.join(OUTPUT_DIR_SUMMARIES, 'robustness_e7_summary.csv'), index=False)
    
    print("Artifact 2 Done.")

# --- Artifact 3: Chunking ---
def generate_chunking_artifact():
    print("Generating Artifact 3: Chunking with Unsafe Exposure...")
    # Prioritize unsafe study if it exists
    unsafe_path = os.path.join(RESULTS_DIR, 'chunking_with_unsafe.parquet')
    if os.path.exists(unsafe_path):
        filepath = unsafe_path
    else:
        filepath = os.path.join(RESULTS_DIR, 'E8_paper_results.parquet')
        
    df = load_and_normalize(filepath)
    if df is None: return
    
    if 'experiment_mode' not in df.columns: return
    
    df['overhead'] = df['stats_total_packets_sent'] / (df['stats_count_benign_packets_sent'] + 1e-9)
    # Safety: delivered_unsafe / total_unsafe
    df['exposure_rate'] = df['stats_total_unsafe_delivered'] / (df['stats_total_unsafe_sent'] + 1e-9)
    # False Refusal: count_false_refusals / count_benign_packets_sent
    df['false_refusal_rate'] = df['stats_count_false_refusals'] / (df['stats_count_benign_packets_sent'] + 1e-9)
    
    stats_cols = ['s_func', 'reach', 'exposure_rate', 'false_refusal_rate', 'overhead']
    summary = df.groupby(['experiment_mode', 'theta'])[stats_cols].mean().reset_index()
    summary_std = df.groupby(['experiment_mode', 'theta'])[stats_cols].std().reset_index()
    
    summary.to_csv(os.path.join(OUTPUT_DIR_SUMMARIES, 'chunking_detailed_table.csv'), index=False)
    
    # Frontiers
    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=summary, x='false_refusal_rate', y='exposure_rate', hue='experiment_mode', style='experiment_mode', s=100)
    plt.title("Safety Frontier: Exposure vs False Refusal")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR_FIGURES, 'fig_chunking_frontier_with_unsafe.png'))
    plt.close()
    
    # Overhead vs Gain
    pivoted = summary.pivot(index='theta', columns='experiment_mode', values=['s_func', 'overhead', 'exposure_rate'])
    if 'baseline' in pivoted['s_func'].columns:
        plt.figure(figsize=(7, 6))
        for m in [c for c in pivoted['s_func'].columns if c != 'baseline']:
            gain = pivoted['s_func'][m] - pivoted['s_func']['baseline']
            cost = pivoted['overhead'][m]
            plt.plot(cost, gain, 'o-', label=m)
            
        plt.xlabel('Overhead (Packets/Msg)')
        plt.ylabel('Connectivity Gain (Δ S_func)')
        plt.title('Cost-Benefit: Overhead vs Connectivity Gain')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR_FIGURES, 'fig_chunking_overhead_tradeoff.png'))
        plt.close()
        
    print("Artifact 3 Done.")


if __name__ == "__main__":
    generate_theory_artifact()
    generate_robustness_artifact()
    generate_chunking_artifact()
    print("All artifacts generated.")
