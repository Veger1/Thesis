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
energy = (baseline["energy"] + energy_diff) / baseline["energy"]
omega2 = (baseline["omega2"] + omega2_diff) / baseline["omega2"]
stiction = (baseline["stiction"] + stiction_diff)/ baseline["stiction"]
zero_cross = (baseline["zero_cross"] + zero_cross_diff)/baseline["zero_cross"]


# -----------------------------
# Final NumPy arrays
# -----------------------------
K = np.array(K)
energy = (np.array(energy)-1.0)*100
omega2 = (np.array(omega2)-1.0)*100
stiction = (np.array(stiction)-1.0)*100
zero_cross = (np.array(zero_cross)-1.0)*100


print("K:", K)
print("Energy:", energy)
print("Omega^2:", omega2)
print("Stiction:", stiction)
print("Zero crossings:", zero_cross)

import matplotlib.pyplot as plt

plt.rcParams.update({
        "font.size": 6,
        "axes.labelsize": 6,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "lines.linewidth": 0.8,
    })
plt.figure(figsize=(3.25, 2.6))
# plt.plot(K, zero_cross, marker='o', label='Zero Crossings', markersize=2)
# plt.plot(K, stiction, marker='v', label='Stiction Time', markersize=2)
# plt.plot(K, omega2, marker='s', label='Omega² Average', markersize=2)
# plt.plot(K, energy, marker='D', label='Energy', markersize=2)
plt.plot(K, zero_cross, linestyle='-', label='Zero Crossings')
plt.plot(K, stiction, linestyle='--', label='Stiction Time')
plt.plot(K, omega2, linestyle='-.', label='Omega² Average')
plt.plot(K, energy, linestyle=':', label='Energy')
plt.axhline(0, color='black', linewidth=0.5)
plt.text(1000000, -2, "pseudoinverse",
         ha='center', va='top')
plt.xscale("log")
plt.xlabel("K")
plt.ylabel("Change relative to pseudoinverse (%)")
plt.legend(loc='upper left')
plt.xticks([1, 100, 10000, 1e6, 1e8])
plt.xlim(1, 1e8)
plt.grid(True, linewidth=0.4, alpha=0.5)
plt.tight_layout()
plt.show()