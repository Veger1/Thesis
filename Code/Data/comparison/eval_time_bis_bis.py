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

# Load pseudo setup time
df_pseudo = pd.read_excel(file_paths['pseudo_omega'])
setup_col_pseudo = next((col for col in df_pseudo.columns if col.lower().replace('_', '').startswith('setup')), None)

if setup_col_pseudo is None:
    raise ValueError("Setup time column not found in pseudo_omega.xlsx")

pseudo_setup = df_pseudo[setup_col_pseudo]
pseudo_mean = pseudo_setup.mean()

# Collect results
results = [{
    'Method': 'pseudo_omega',
    'Mean Setup Time': pseudo_mean,
    'Mean Diff (method - pseudo)': 0.0,
    'p-value': '',
    'Significant (p<0.05)': ''
}]

# Compare other methods (only if they contain setup time)
for method, file in file_paths.items():
    if method == 'pseudo_omega':
        continue

    if os.path.exists(file):
        df = pd.read_excel(file)
        setup_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('setup')), None)

        if setup_col:
            setup_time = df[setup_col]
            min_len = min(len(pseudo_setup), len(setup_time))
            stat, pval = ttest_rel(setup_time[:min_len], pseudo_setup[:min_len])
            results.append({
                'Method': method,
                'Mean Setup Time': setup_time.mean(),
                'Mean Diff (method - pseudo)': setup_time.mean() - pseudo_mean,
                'p-value': f"{pval:.1e}",
                'Significant (p<0.05)': pval < 0.05
            })

# Output table
setup_df = pd.DataFrame(results)
print(setup_df.to_string(index=False))
