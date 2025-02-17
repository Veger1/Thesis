from casadi import arctan
from sympy import symbols, exp, tanh, pi
from itertools import product
from Code.repeat_solver import fast_solve_ocp, calc_cost
from scipy.io import savemat

# Define symbolic variable
w = symbols('w')

# Define base cost function components
base_costs = {
    "gaussian": lambda a: exp(-a * w ** 2),
    "tanh": lambda k: (tanh(k * (w + pi * 10))) * 0.5 + (tanh(-k * (w - pi * 10))) * 0.5,
    "speed": lambda b: b * w ** 2,
    "linear": lambda b: b * w
}

# Define parameter ranges for variation
param_ranges = {
    "a": [0.001, 0.002],  # Vary Gaussian decay rate
    "k": [1.0, 2.0],  # Vary tanh steepness
    "b": [1 / 700000, 1 / 350000]  # Vary speed scaling
}

# Define cost function combinations (sum of selected terms)
cost_combinations = [
    ["gaussian", "tanh"],
    ["gaussian", "speed"],
    ["tanh", "linear"],
    ["gaussian", "tanh", "speed"]
]

# Define optimization parameters
num_intervals, N = 4, 100
scaling, time = 1.0, float(N / 10)

# Store results
results = {}

# Iterate through cost function combinations
for cost_names in cost_combinations:
    # Iterate through all parameter value combinations
    for params in product(*param_ranges.values()):
        # Assign parameter values
        param_dict = dict(zip(param_ranges.keys(), params))

        # Construct cost expression by summing selected terms
        cost_expr = sum(base_costs[name](param_dict[name[0]]) for name in cost_names)

        # Generate a unique name based on cost functions and parameter values
        param_str = "_".join([f"{k}{v}" for k, v in param_dict.items()])
        cost_name = "_".join(cost_names) + f"_{param_str}"

        print(f"Running optimization for {cost_name}...")

        try:
            # Solve optimization
            t_sol, w_sol, alpha_sol, T_sol = fast_solve_ocp(cost_expr, num_intervals, N, time, scaling)
            cost, total_cost, cost_graph, omega_axis = calc_cost(w_sol, cost_expr)

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
            savemat(f'Data/output_{cost_name}.mat', results[cost_name])

        except Exception as e:
            print(f"Error while processing {cost_name}: {e}")

# Optionally, save all results in one file
savemat('Data/all_results.mat', results)
