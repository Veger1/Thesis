import pandas as pd

df = pd.read_excel("pseudo_omega.xlsx")

metrics = ['Zero Crossings', 'Omega² Avg', 'Energy (J)', 'Stiction Time (s)']
baseline_means = {metric: df[metric].mean() for metric in metrics}
print(baseline_means)
