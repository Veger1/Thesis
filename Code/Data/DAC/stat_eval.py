import pandas as pd
from scipy.stats import ttest_rel

# Load both Excel files
df1 = pd.read_excel("slew1/slew1_results.xlsx")
df2 = pd.read_excel("slew2/slew2_results.xlsx")

# Combine into one DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# Pivot so we can access reference values easily
pivoted = df.pivot(index="file", columns="class", values=["solve_time", "iteration_count"])

# Drop rows where ref data is missing
pivoted = pivoted.dropna(subset=[("solve_time", "ref"), ("iteration_count", "ref")])

# Store the statistical results
results = []

for class_name in ["osl", "tsl", "osl_tsl", "og", "cg", "og_cg"]:
    if class_name not in pivoted["solve_time"].columns:
        continue

    # Calculate % change compared to 'ref'
    solve_time_change = (
        (pivoted["solve_time"][class_name] - pivoted["solve_time"]["ref"]) /
        pivoted["solve_time"]["ref"] * 100
    )

    iteration_change = (
        (pivoted["iteration_count"][class_name] - pivoted["iteration_count"]["ref"]) /
        pivoted["iteration_count"]["ref"] * 100
    )

    # Drop NaN pairs for t-test
    solve_ref = pivoted["solve_time"]["ref"]
    solve_other = pivoted["solve_time"][class_name]
    iter_ref = pivoted["iteration_count"]["ref"]
    iter_other = pivoted["iteration_count"][class_name]

    # Only keep matched data
    solve_data = pd.concat([solve_ref, solve_other], axis=1).dropna()
    iter_data = pd.concat([iter_ref, iter_other], axis=1).dropna()

    # Paired t-tests
    solve_t_stat, solve_p = ttest_rel(solve_data.iloc[:, 0], solve_data.iloc[:, 1])
    iter_t_stat, iter_p = ttest_rel(iter_data.iloc[:, 0], iter_data.iloc[:, 1])

    results.append({
        "class": class_name,
        "mean_solve_time_change (%)": solve_time_change.mean(),
        "mean_iteration_change (%)": iteration_change.mean(),
        "solve_time_p_value": solve_p,
        "iteration_p_value": iter_p
    })

# Save statistical summary
results_df = pd.DataFrame(results)
results_df.to_excel("combined_solver_comparison_vs_ref.xlsx", index=False)

print("Saved combined comparison results to 'combined_solver_comparison_vs_ref.xlsx'")
