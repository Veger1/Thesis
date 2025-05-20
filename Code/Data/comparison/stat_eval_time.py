import pandas as pd
from scipy.stats import ttest_rel
import os

# File mappings
file_paths = {
    'pseudo_omega': 'pseudo_omega.xlsx',
    'minmax_omega': 'minmax_omega.xlsx',
    'graph_PID_k_0': 'graph_PID_k_0.xlsx',
    'graph_PID_k_10000': 'graph_PID_k_10000.xlsx',
    'graph_OPT_k_0': 'graph_OPT_k_0.xlsx',
    'graph_OPT_k_10000': 'graph_OPT_k_10000.xlsx',
    'OPT': 'OPT.xlsx'
}

# Load baseline pseudo
df_pseudo = pd.read_excel(file_paths['pseudo_omega'])
solve_col_pseudo = next((col for col in df_pseudo.columns if col.lower().replace('_', '').startswith('solve')), None)
setup_col_pseudo = next((col for col in df_pseudo.columns if col.lower().replace('_', '').startswith('setup')), None)
pseudo_total = df_pseudo[solve_col_pseudo] + (df_pseudo[setup_col_pseudo] if setup_col_pseudo else 0)
pseudo_mean = pseudo_total.mean()

# Store results
results = [{
    'Method': 'pseudo_omega',
    'Mean Total Time': pseudo_mean,
    'Mean Diff (method - pseudo)': 0.0,
    'p-value': '',
    'Significant (p<0.05)': ''
}]

# T-tests for all other methods
for method, file in file_paths.items():
    if method == 'pseudo_omega':
        continue

    if os.path.exists(file):
        df = pd.read_excel(file)
        solve_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('solve')), None)
        setup_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('setup')), None)

        if solve_col:
            total_time = df[solve_col] + (df[setup_col] if setup_col else 0)
            min_len = min(len(pseudo_total), len(total_time))
            total_time = total_time[:min_len]
            pseudo_subset = pseudo_total[:min_len]

            stat, pval = ttest_rel(total_time, pseudo_subset)
            results.append({
                'Method': method,
                'Mean Total Time': total_time.mean(),
                'Mean Diff (method - pseudo)': total_time.mean() - pseudo_mean,
                'p-value': f"{pval:.1e}",
                'Significant (p<0.05)': pval < 0.05
            })

# Final table
ttest_df = pd.DataFrame(results)
print(ttest_df.to_string(index=False))
