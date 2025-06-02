import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Confirm the file is in the correct location or update this path accordingly
excel_file = "summary_metrics.xlsx"

# Load all sheets from the Excel file
xls = pd.ExcelFile(excel_file)
all_dfs = [pd.read_excel(xls, sheet_name=sheet) for sheet in xls.sheet_names]
combined_df = pd.concat(all_dfs, ignore_index=True)

# List of metrics to plot
metrics = [
    'Zero Crossings',
    'Omega² Avg',
    'Energy',
    'Stiction Time (limit=100)'
]

# Create a boxplot for each metric grouped by parameter k
for metric in metrics:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=combined_df['k'], y=combined_df[metric])
    plt.title(f'{metric} vs. Parameter k (All Slews Combined)')
    plt.xlabel('Parameter k')
    plt.ylabel(metric)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
