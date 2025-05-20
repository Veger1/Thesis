import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define file locations and labels
file_paths = {
    'pseudo_omega': 'pseudo_omega.xlsx',
    'minmax_omega': 'minmax_omega.xlsx',
    'graph_PID_k_0': 'graph_PID_k_0.xlsx',
    'graph_PID_k_10000': 'graph_PID_k_10000.xlsx',
    'graph_OPT_k_0': 'graph_OPT_k_0.xlsx',
    'graph_OPT_k_10000': 'graph_OPT_k_10000.xlsx',
    'OPT': 'OPT.xlsx'
}

label_map = {
    'pseudo_omega': r'Pseudo $\omega$',
    'minmax_omega': r'Minmax $\omega$',
    'graph_PID_k_0': r'Graph \& PID' + '\n' + r'($k = 0$)',
    'graph_PID_k_10000': r'Graph \& PID' + '\n' + r'($k = 10{,}000$)',
    'graph_OPT_k_0': r'Graph \& NLP' + '\n' + r'($k = 0$)',
    'graph_OPT_k_10000': r'Graph \& NLP' + '\n' + r'($k = 10{,}000$)',
    'OPT': r'NLP'
}


# Collect and normalize time data
time_data = []
for method, file in file_paths.items():
    if os.path.exists(file):
        df = pd.read_excel(file)
        label = label_map[method]
        solve_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('solve')), None)
        setup_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('setup')), None)

        if solve_col:
            time_data.append(df[[solve_col]].assign(Method=label, Metric='Solve Time').rename(columns={solve_col: 'Time'}))
        if setup_col:
            time_data.append(df[[setup_col]].assign(Method=label, Metric='Setup Time').rename(columns={setup_col: 'Time'}))

        # Optionally: total time if both present
        if solve_col and setup_col:
            total_time = df[solve_col] + df[setup_col]
            time_data.append(pd.DataFrame({
                'Time': total_time,
                'Method': label,
                'Metric': 'Total Time'
            }))

# Combine and plot
if time_data:
    time_df = pd.concat(time_data, ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.stripplot(x='Method', y='Time', data=time_df, hue='Metric', dodge=True, jitter=True, alpha=0.7)
    plt.yscale('log')
    plt.title("Computation Times Across Methods (Log Scale)")
    plt.xlabel("")
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Metric')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("No solve/setup time data found.")
