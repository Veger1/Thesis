import pandas as pd
import os

# Only methods with setup time
file_paths = {
    'graph_OPT_k_0': 'graph_OPT_k_0.xlsx',
    'graph_OPT_k_10000': 'graph_OPT_k_10000.xlsx',
    'OPT': 'OPT.xlsx'
}

results = []

for method, file in file_paths.items():
    if os.path.exists(file):
        df = pd.read_excel(file)
        setup_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('setup')), None)
        if setup_col:
            setup_time = df[setup_col]
            results.append({
                'Method': method,
                'Mean Setup Time': setup_time.mean()
            })

# Output
setup_df = pd.DataFrame(results)
print(setup_df.to_string(index=False))
