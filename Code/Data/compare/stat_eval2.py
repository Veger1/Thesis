import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import numpy as np

# Load Excel file
xls = pd.ExcelFile("summary_metrics.xlsx")
dfs = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values(by='k')

def analyze_vs_baseline(df, metric, title_label):
    # Pivot by (k, Seed)
    grouped = df.groupby(['k', 'Seed'])[metric].mean().unstack(level=0)
    grouped = grouped.sort_index(axis=1)

    # Use first k as baseline
    k0 = grouped.columns[0]
    delta_means, p_values, significant, labels = [], [], [], []

    for k in grouped.columns[1:]:
        x = grouped[k0]
        y = grouped[k]
        paired = pd.concat([x, y], axis=1, keys=['base', 'comp']).dropna()
        t_stat, p_val = ttest_rel(paired['base'], paired['comp'])
        mean_diff = (paired['comp'] - paired['base']).mean()
        delta_means.append(mean_diff)
        p_values.append(p_val)
        significant.append(p_val < 0.05)
        labels.append(f"{k0}→{k}")

    # Plot
    colors = ['red' if sig else 'gray' for sig in significant]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, delta_means, color=colors)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel(f'Mean Δ {title_label} vs. k=0')
    plt.title(f'Change in {title_label} Compared to Baseline (k=0)')
    for i, (val, sig) in enumerate(zip(delta_means, significant)):
        label = "*" if sig else ""
        plt.text(i, val, label, ha='center',
                 va='bottom' if val >= 0 else 'top', fontsize=12)
    plt.tight_layout()
    plt.show()

# Run for Energy and Omega² Avg
analyze_vs_baseline(df, metric='Zero Crossings', title_label='Zero Crossings')
analyze_vs_baseline(df, metric='Stiction Time (limit=100)', title_label='Stiction Time')
analyze_vs_baseline(df, metric='Energy', title_label='Energy')
analyze_vs_baseline(df, metric='Omega² Avg', title_label='Vibration (Omega² Avg)')
