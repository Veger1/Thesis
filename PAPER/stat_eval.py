import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import wilcoxon

# Define paths to your method data
file_paths = {
    'pseudo_omega': 'final_data/pseudo.xlsx',
    # 'minmax_omega': 'final_data/minmax.xlsx',
    # 'graph_1' : 'final_data/graph_1.xlsx',
    # 'graph_10': 'final_data/graph_10.xlsx',
    # 'graph_100': 'final_data/graph_100.xlsx',
    # 'graph_1K': 'final_data/graph_1K.xlsx',
    # 'graph_10K': 'final_data/graph_10K.xlsx',
    # 'graph_100K': 'final_data/graph_100K.xlsx',
    # 'graph_1M': 'final_data/graph_1M.xlsx',
    # 'graph_10M': 'final_data/graph_10M.xlsx',
    # 'graph_100M': 'final_data/graph_100M.xlsx',
    "fast_graph_1": "fast.xlsx",
    "slow_graph_1": "slow.xlsx",
}

# Load data
data = {method: pd.read_excel(path) for method, path in file_paths.items()}

# Metrics to test
metrics = ['Zero Crossings', 'Omega² Avg', 'Energy', 'Stiction Time']

# Use pseudo_omega as reference
baseline = data['pseudo_omega'].set_index(['Iteration'])

# Store results
results = []

for method, df in data.items():
    if method == 'pseudo_omega':
        continue  # skip baseline

    df = df.set_index(['Iteration'])
    common_idx = baseline.index.intersection(df.index)

    for metric in metrics:
        if metric in df.columns and metric in baseline.columns:
            base_vals = baseline.loc[common_idx, metric]
            comp_vals = df.loc[common_idx, metric]

            # Paired t-test
            # t_stat, p_val = ttest_rel(comp_vals, base_vals, nan_policy='omit')
            try:
                stat, p_val = wilcoxon(comp_vals, base_vals)
            except ValueError:
                stat, p_val = None, None
            mean_diff = (comp_vals - base_vals).mean()
            # significant = p_val < 0.05
            significant = p_val is not None and p_val < 0.05

            results.append({
                'Compared To': 'pseudo_omega',
                'Method': method,
                'Metric': metric,
                'Mean Diff': mean_diff,
                'p-value': p_val,
                'Significant (p<0.05)': significant
            })

# Create DataFrame and export or print
result_df = pd.DataFrame(results)
print(result_df)

# Optionally save:
# result_df.to_excel("wilcoxon_test_vs_pseudo.xlsx", index=False)
