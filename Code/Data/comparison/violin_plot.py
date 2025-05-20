import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define paths to the 7 Excel files
file_paths = {
    'pseudo_omega': 'pseudo_omega.xlsx',
    'minmax_omega': 'minmax_omega.xlsx',
    'graph_PID_k_0': 'graph_PID_k_0.xlsx',
    'graph_PID_k_10000': 'graph_PID_k_10000.xlsx',
    'graph_OPT_k_0': 'graph_OPT_k_0.xlsx',
    'graph_OPT_k_10000': 'graph_OPT_k_10000.xlsx',
    'OPT': 'OPT.xlsx'
}
matlab_colors = [
    '#0072BD',  # blue
    '#D95319',  # orange
    '#EDB120',  # yellow
    '#7E2F8E',  # purple
    '#77AC30',  # green
    '#A2142F',  # red
    '#4DBEEE'   # light blue
]

label_map = {
    'pseudo_omega': r'Pseudo $\omega$',
    'minmax_omega': r'Minmax $\omega$',
    'graph_PID_k_0': r'Graph & PID' + '\n' + r'($k = 0$)',
    'graph_PID_k_10000': r'Graph & PID' + '\n' + r'($k = 10{,}000$)',
    'graph_OPT_k_0': r'Graph & NLP' + '\n' + r'($k = 0$)',
    'graph_OPT_k_10000': r'Graph & NLP' + '\n' + r'($k = 10{,}000$)',
    'OPT': r'NLP'
}

method_order = list(file_paths.keys())
# palette = {method: color for method, color in zip(method_order, matlab_colors)}
palette = {label_map[method]: color for method, color in zip(method_order, matlab_colors)}
# Load all data and label with method name
dfs = []
for method, file in file_paths.items():
    if os.path.exists(file):
        df = pd.read_excel(file)
        df['Method'] = method
        dfs.append(df)
    else:
        print(f"File not found: {file}")

# Combine into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

combined_df['Method Label'] = combined_df['Method'].map(label_map)


# Metrics to plot
metrics = ['Zero Crossings', 'Omega² Avg', 'Energy (J)', 'Stiction Time (s)']

# Plot each metric
# for metric in metrics:
#     plt.figure(figsize=(10, 5))
#     sns.violinplot(x='Method', y=metric, data=combined_df, inner='quartile', cut=0, palette=palette,
#         order=method_order, legend=False)
#     plt.title(f'Violin Plot of {metric} Across Methods')
#     plt.xticks(rotation=45)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.violinplot(
        x='Method Label',
        y=metric,
        data=combined_df,
        inner='quartile',
        cut=0,
        palette=palette,
        order=[label_map[key] for key in method_order],
        hue='Method Label',
        legend=False
    )
    # plt.title(f'Violin Plot of {metric} Across Methods')
    plt.xticks(rotation=0, ha='center', wrap=True)  # no tilt, wrap text
    plt.xlabel("")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
