import pickle
from tabulate import tabulate
import numpy as np

def analyze_results(file_path):
    # Load the results from the .pkl file
    with open(file_path, "rb") as f:
        results = pickle.load(f)

    # Initialize accumulators for metrics and timing
    methods = ["safe_run", "run_minmax", "run_pseudo"]
    metrics_keys = ["stiction_time", "energy", "zero_crossing", "omega_squared", "number_of_layers"]
    timing_keys = ["total_time"]

    # Create dictionaries to store values for standard deviation calculation
    metrics_values = {method: {key: [] for key in metrics_keys} for method in methods}
    timing_values = {method: {key: [] for key in timing_keys} for method in methods}

    # Iterate through the results and collect values
    for result in results:
        for method in methods:
            if result[method]["timing"] is not None:
                for key in metrics_keys:
                    metrics_values[method][key].append(result[method]["metrics"][key])
                for key in timing_keys:
                    timing_values[method][key].append(result[method]["timing"][key])

    # Calculate averages and standard deviations
    rows = []
    for method in methods:
        avg_metrics = {key: np.mean(metrics_values[method][key]) for key in metrics_keys}
        std_metrics = {key: np.std(metrics_values[method][key]) for key in metrics_keys}
        avg_timing = {key: np.mean(timing_values[method][key]) for key in timing_keys}
        std_timing = {key: np.std(timing_values[method][key]) for key in timing_keys}

        rows.append(
            [method]
            + [avg_metrics[key] for key in metrics_keys]
            + [std_metrics[key] for key in metrics_keys]
            + [avg_timing[key] for key in timing_keys]
            + [std_timing[key] for key in timing_keys]
        )

    # Tabulate results
    headers = (
        ["Method"]
        + [f"avg_{key}" for key in metrics_keys]
        + [f"std_{key}" for key in metrics_keys]
        + [f"avg_{key}" for key in timing_keys]
        + [f"std_{key}" for key in timing_keys]
    )
    print(tabulate(rows, headers=headers, floatfmt=".4f"))

# Call the function with the path to the .pkl file
analyze_results("normalized_test/50_10K.pkl")
analyze_results("normalized_test/50_100K.pkl")
analyze_results("normalized_test/50_1M.pkl")
analyze_results("normalized_test/50_10M.pkl")
analyze_results("simulation_results.pkl")
