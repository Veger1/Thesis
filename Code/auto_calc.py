from casadi import arctan
from sympy import symbols, exp, tanh, pi, Mul
from itertools import product
from Code.repeat_solver import fast_solve_ocp, calc_cost, solve_ocp
from scipy.io import savemat
import signal
import sys

# Define symbolic variable
w, t = symbols('w t')

# Define base cost function components, keep these basic
base_costs = {
    "gaussian": lambda a: exp(-a * w ** 2),
    "tanh": lambda k: (tanh(k * (w + pi * 10))) * 0.5 + (tanh(-k * (w - pi * 10))) * 0.5,
    "speed": lambda b: b * w ** 2,
    "linear": lambda b: b * w,
    "time_dependent": lambda c: (8 / (1 + exp(-c * (t - 60)))),
    "time_dep": lambda c: c*t
}

# Define parameter ranges for variation
param_ranges = {
    "a": [0.1, 0.05, 0.01, 0.005, 0.002, 0.001],  # Vary Gaussian decay rate
    "k": [1.0, 2.0],  # Vary tanh steepness
    "b": [1 / 700000, 1 / 350000, 1 / 200000, 1 / 100000],  # Vary speed scaling
    "c": [0.5, 1, 2]  # Time-dependent scaling
}
params_for_cost = {
    "gaussian": "a",
    "tanh": "k",
    "speed": "b",
    "linear": "b",
    "time_dependent": "c",
    "time_dep": "c"
}

# Define cost function combinations
cost_combinations = [
    # ["gaussian", "tanh"],
    # ["gaussian", "speed"],
    # ["tanh", "linear"],
    # ["gaussian", "tanh", "speed"],
    # ["gaussian*time_dependent"],
    ["gaussian", "speed*time_dep"]
]

# Define optimization parameters
num_intervals, N = 16, 500
scaling, time = 1.0, float(N / 10)

# Memory Estimation
num_points = num_intervals * N
values_per_point = 10  # 10 numbers stored per point
bytes_per_value = 4  # Assuming 32-bit (4 bytes per number)
memory_per_solution = num_points * values_per_point * bytes_per_value  # in bytes
memory_per_solution_MB = memory_per_solution / (1024 ** 2)  # Convert to MB

# Precompute all cost expressions with parameters
cost_expressions = []
for cost_names in cost_combinations:
    # Split cost names if multiplication is involved
    cost_names_split = [name.split('*') for name in cost_names]
    cost_names_flat = [item for sublist in cost_names_split for item in sublist]

    param_lists = [param_ranges[params_for_cost[name]] for name in cost_names_flat]  # Get valid parameter lists

    for param_values in product(*param_lists):  # Generate all combinations
        param_dict = {params_for_cost[name]: value for name, value in zip(cost_names_flat, param_values)}

        # Build cost expression
        cost_expr = 0  # Initialize to 0 for addition
        for cost_name in cost_names:
            if '*' in cost_name:
                # Handle multiplication
                components = cost_name.split('*')
                component_exprs = [base_costs[comp](param_dict[params_for_cost[comp]]) for comp in components]
                cost_expr += Mul(*component_exprs)  # Add the product to the cost expression
            else:
                # Handle addition
                cost_expr += base_costs[cost_name](param_dict[params_for_cost[cost_name]])

        # Create a unique identifier
        param_str = "_".join([f"{k}{v}" for k, v in param_dict.items()])
        cost_name = "_".join(cost_names) + f"_{param_str}"

        cost_expressions.append((cost_name, cost_expr))

# Display total combinations and estimated memory usage
total_combinations = len(cost_expressions)
total_memory_MB = total_combinations * memory_per_solution_MB
print(f"Total cost function combinations: {total_combinations}")
print(f"Estimated memory usage after solving: {total_memory_MB:.2f} MB")

# Storage for results
results = {}
completed_solutions = 0


# Function to handle KeyboardInterrupt
def save_and_exit():
    global results, completed_solutions
    print("\nSaving results before exiting...")
    savemat('Data/Auto/all_results_partial.mat', results)  # Save all completed results
    print(f"✅ {completed_solutions} solutions saved successfully!")
    sys.exit(0)


# Catch Ctrl+C (KeyboardInterrupt)
signal.signal(signal.SIGINT, lambda signum, frame: save_and_exit())

# Solve optimization for each cost expression
try:
    for cost_name, cost_expr in cost_expressions:
        print(f"Running optimization for {cost_name}...")
        if True:  # replace with try, add exception handling
            # Solve optimization
            t_sol, w_sol, alpha_sol, T_sol = solve_ocp(cost_expr, num_intervals, N, time, scaling)
            cost, total_cost, cost_graph, omega_axis = calc_cost(w_sol, cost_expr, t_sol)

            # Store results
            results[cost_name] = {
                'all_t': t_sol,
                'all_w_sol': w_sol,
                'all_alpha_sol': alpha_sol,
                'all_T_sol': T_sol,
                'cost': cost,
                'total_cost': total_cost,
                'cost_graph': cost_graph,
                'omega_axis': omega_axis,
                'cost_expr': str(cost_expr)
            }

            # Save results to a separate file
            clean_cost_name = cost_name.replace('*', 'X')
            savemat(f'Data/Auto/{clean_cost_name}.mat', results[cost_name])
            completed_solutions += 1

except KeyboardInterrupt:
    save_and_exit()  # Save results and exit safely

# Final save if loop completes normally
savemat('Data/Auto/all_results.mat', results)
print(f"✅ All {completed_solutions} solutions saved successfully!")