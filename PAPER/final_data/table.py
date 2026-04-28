import pandas as pd
import numpy as np
import re

from matplotlib.pyplot import legend


# -----------------------------
# Helper to convert "graph_10K" → 10000
# -----------------------------
def parse_K(method_name):

    if not method_name.startswith("graph_"):
        return None

    val = method_name.replace("graph_", "")

    if val.endswith("K"):
        return float(val[:-1]) * 1e3
    elif val.endswith("M"):
        return float(val[:-1]) * 1e6
    else:
        return float(val)


# -----------------------------
# Load comparison table
# -----------------------------
file_comparison = "wilcoxon_test_vs_pseudo.xlsx"

df = pd.read_excel(file_comparison)

# Convert comma decimal numbers
df["Mean Diff"] = df["Mean Diff"].astype(str).str.replace(",", ".").astype(float)

# Keep only graph_K rows
df_graph = df[df["Method"].str.startswith("graph_")].copy()

# Compute K values
df_graph["K"] = df_graph["Method"].apply(parse_K)

# -----------------------------
# Build arrays per metric
# -----------------------------
metrics = ["Energy", "Omega² Avg", "Stiction Time", "Zero Crossings"]

data = {}

for metric in metrics:

    sub = df_graph[df_graph["Metric"] == metric].sort_values("K")

    K = sub["K"].to_numpy()
    vals = sub["Mean Diff"].to_numpy()

    data[metric] = vals

# Common K array
K = df_graph[df_graph["Metric"] == "Energy"].sort_values("K")["K"].to_numpy()


energy_diff = data["Energy"]
omega2_diff = data["Omega² Avg"]
stiction_diff = data["Stiction Time"]
zero_cross_diff = data["Zero Crossings"]


# -----------------------------
# Load baseline pseudo_omega runs
# -----------------------------
file_baseline = "pseudo.xlsx"

df_base = pd.read_excel(file_baseline)

# Convert comma decimals
cols = ["Stiction Time", "Energy", "Omega² Avg", "Timing"]

for c in cols:
    df_base[c] = df_base[c].astype(str).str.replace(",", ".").astype(float)

baseline = {
    "energy": df_base["Energy"].mean(),
    "omega2": df_base["Omega² Avg"].mean(),
    "stiction": df_base["Stiction Time"].mean(),
    "zero_cross": df_base["Zero Crossings"].mean()
}


# -----------------------------
# Convert diff → absolute values
# -----------------------------
# energy = (baseline["energy"] + energy_diff) / baseline["energy"]
# omega2 = (baseline["omega2"] + omega2_diff) / baseline["omega2"]
# stiction = (baseline["stiction"] + stiction_diff)/ baseline["stiction"]
# zero_cross = (baseline["zero_cross"] + zero_cross_diff)/baseline["zero_cross"]
energy = energy_diff
omega2 = omega2_diff
stiction = stiction_diff
zero_cross = cd zero_cross_diff

from scipy.stats import wilcoxon

# -----------------------------
# Build sign table
# -----------------------------
# -----------------------------
# Build table with relative differences (%)
# -----------------------------
methods = [f"graph_{int(k) if k < 1e6 else int(k/1e6)}{'K' if k<1e6 else 'M'}" for k in K]

baseline_row = pd.DataFrame([{
    "Method": "Pseudoinverse",
    "Energy": baseline["energy"],
    "Omega² Avg": baseline["omega2"],
    "Stiction Time": baseline["stiction"],
    "Zero Crossings": baseline["zero_cross"]
}])

df_table = pd.DataFrame({
    "Method": methods,
    "Energy": energy,
    "Omega² Avg": omega2,
    "Stiction Time": stiction,
    "Zero Crossings": zero_cross
})

df_table = pd.concat([baseline_row, df_table], ignore_index=True)
df_table = df_table.round(2)

print("\nWilcoxon comparison vs pseudoinverse:\n")
print(df_table.to_string(index=False))
