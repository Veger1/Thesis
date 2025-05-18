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
    grouped = df.groupby(['k', 'Seed'])[metric].mean().unstack(level=0)
    grouped = grouped.sort_index(axis=1)

    k_values = grouped.columns.tolist()
    k0 = k_values[0]  # baseline
    baseline_mean = grouped[k0].mean()

    delta_means = []
    p_values = []
    significant = []
    labels = []
    target_means = []

    for k in k_values[1:]:
        x = grouped[k0]
        y = grouped[k]
        paired = pd.concat([x, y], axis=1, keys=['base', 'comp']).dropna()
        t_stat, p_val = ttest_rel(paired['base'], paired['comp'])
        mean_diff = (paired['comp'] - paired['base']).mean()
        delta_means.append(mean_diff)
        p_values.append(p_val)
        significant.append(p_val < 0.05)
        target_means.append(paired['comp'].mean())
        labels.append(f"{k0}→{k}")

    # Plot
    colors = ['gray' if sig else 'gray' for sig in significant]
    plt.figure(figsize=(10, 4))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    bars = plt.bar(labels, delta_means, color=colors)
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel(f'Mean Δ {title_label} vs. k={k0}')
    # plt.title(f'Change in {title_label} Compared to Baseline (k={k0})')

    # Annotate bars with actual means
    for i, (mean_target, sig) in enumerate(zip(target_means, significant)):
        label = "*" if sig else ""
        base_str = f"μ₀={baseline_mean:.2f}"
        target_str = f"μₖ={mean_target:.2f}"
        full_label = f"{label}\n{base_str}\n{target_str}"
        # full_label = f"Δμₖ={mean_target-baseline_mean:.2f}"
        va_pos = 'bottom' if delta_means[i] >= 0 else 'top'
        plt.text(i, delta_means[i], full_label, ha='center', va=va_pos, fontsize=9)

    plt.tight_layout()
    plt.show()

# Apply to Energy and Vibration
analyze_vs_baseline(df, metric='Energy', title_label='Energy (J)')
analyze_vs_baseline(df, metric='Omega² Avg', title_label='Vibration')
analyze_vs_baseline(df, metric='Zero Crossings', title_label='Zero Crossings')
analyze_vs_baseline(df, metric='Stiction Time (limit=100)', title_label='Stiction Time (s)')
