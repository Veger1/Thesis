import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# File locations and LaTeX-style labels
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
    'graph_PID_k_0': r'Graph & PID' + '\n' + r'($k = 0$)',
    'graph_PID_k_10000': r'Graph & PID' + '\n' + r'($k = 10{,}000$)',
    'graph_OPT_k_0': r'Graph & NLP' + '\n' + r'($k = 0$)',
    'graph_OPT_k_10000': r'Graph & NLP' + '\n' + r'($k = 10{,}000$)',
    'OPT': r'NLP'
}

# Collect total times
total_time_data = []

for method, file in file_paths.items():
    if os.path.exists(file):
        df = pd.read_excel(file)
        label = label_map[method]

        # Try to find solve/setup time using loose matching
        solve_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('solve')), None)
        setup_col = next((col for col in df.columns if col.lower().replace('_', '').startswith('setup')), None)

        if solve_col:
            solve = df[solve_col]
            setup = df[setup_col] if setup_col else 0
            total_time = solve + setup
            total_time_data.append(pd.DataFrame({
                'Time': total_time,
                'Method': label
            }))
        else:
            print(f"⚠️ No solve time found for {file}")

# Combine and plot total time
if total_time_data:
    total_df = pd.concat(total_time_data, ignore_index=True)

    plt.figure(figsize=(10, 5))
    sns.stripplot(x='Method', y='Time', data=total_df, jitter=True, alpha=0.7)
    plt.yscale('log')
    # plt.title("Total Time (Solve + Setup) per Method")
    plt.xlabel("")
    plt.ylabel("Total Time (s)")
    # plt.xticks(rotation=45, ha='right')
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("❌ No valid total time data found.")
