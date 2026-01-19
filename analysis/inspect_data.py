import pandas as pd
import glob
import os

files = glob.glob('experiments/results/*_paper_results.parquet')

for f in files:
    print(f"\n--- {f} ---")
    try:
        df = pd.read_parquet(f)
        print("Columns:", list(df.columns))
        print("Sample Row:", df.iloc[0].to_dict())
        if 'E7' in f:
             if 'robustness_axis' in df.columns:
                 print("Robustness Axes:", df['robustness_axis'].unique())
                 # check hetero_beta range
                 sub = df[df['robustness_axis'] == 'hetero_beta']
                 if not sub.empty:
                     if 'policies_strictness_dist_value' in sub.columns:
                         print("Beta Theta (Value):", sorted(sub['policies_strictness_dist_value'].unique()))
                     else:
                          # try identifying theta col
                          cols = [c for c in sub.columns if 'strictness' in c]
                          print("Beta Strictness Cols:", cols)
                          # print unique values of likely theta
    except Exception as e:
        print(f"Error reading {f}: {e}")
