import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
file_comparison = "graph_1.xlsx"

# Load file
df = pd.read_excel(file_comparison)

# If needed: convert comma decimals (only if column is not integer)
df["Number of Layers"] = (
    df["Number of Layers"]
    .astype(str)
    .str.replace(",", ".")
    .astype(float)
)

# Plot histogram
plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 7,
        "lines.linewidth": 0.8,
    })

plt.figure(figsize=(3.25, 1.6))
plt.grid(True, linewidth=0.4, alpha=0.5)
plt.hist(df["Number of Layers"],  bins=range(int(df["Number of Layers"].min()),
                                            int(df["Number of Layers"].max()) + 2), edgecolor="black", linewidth=0.5)

plt.xlabel("Number of Layers")
plt.ylabel("Frequency")
# plt.title("Histogram of Number of Layers")
# plt.show()
print(np.mean(df["Number of Layers"]))

import numpy as np
import matplotlib.pyplot as plt

# Create a smooth Stribeck-like curve
x = np.logspace(-3, 2, 500)  # lubrication parameter (log scale)

# Synthetic curve: high friction -> dip -> slight rise
mu = 0.15 * np.exp(-2 * x) + 0.02 + 0.04 * np.log10(1 + x)

plt.figure(figsize=(3.25, 2.0))
plt.plot(x, mu, linewidth=2)

# Vertical region boundaries (chosen for illustration)
x1, x2 = 0.01, 1

plt.axvline(x1, linestyle="--")
plt.axvline(x2, linestyle="--")

# Labels for regions
plt.text(0.002, 0.12, "1", fontsize=8)  # Boundary lubrication
plt.text(0.05, 0.12, "2", fontsize=8)   # Mixed lubrication
plt.text(10, 0.12, "3", fontsize=8)     # Hydrodynamic lubrication

# Axis formatting
plt.xscale("log")
plt.xlabel("Lubrication parameter (η·v / P)")
plt.ylabel("Friction coefficient μ")
# plt.title("Stribeck Curve")
plt.xticks([])
plt.yticks([])
plt.gca().tick_params(axis='x', which='both', bottom=False, top=False)

# Optional region names (comment out if you only want numbers)
plt.text(0.0006, 0.105, "Boundary")
plt.text(0.03, 0.105, "Mixed")
plt.text(2, 0.105, "Hydrodynamic")

# plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.show()
