from rockit import Ocp
import numpy as np
from sympy import symbols, exp, Abs
from Code.repeat_solver import solve_ocp, calc_cost
from scipy.io import savemat

num_intervals, N = 1, 50
scaling, time = 1.0, float(N/10)

ocp = Ocp()
w = ocp.state(4)
c1, c2 = 10, 0.001  # Define the cost function terms
objective = c1 * np.exp(-c2 * (w ** 2))

w = symbols('w') # Use symbolic variable
cost_expr = c1 * exp(-c2 * (w ** 2))

t_sol, w_sol, alpha_sol, T_sol = 0, 0, 0, 0
cost, total_cost, cost_graph, omega_axis = 0, 0, 0, 0

try:  # Skip this part if optimization has already been done
    t_sol, w_sol, alpha_sol, T_sol = solve_ocp(objective, num_intervals, N, time, scaling)
except Exception as e1:
    print(f"Error: {e1}")

try: # Import data from the previous optimization if exists
    cost, total_cost, cost_graph, omega_axis = calc_cost(w_sol, cost_expr)
except Exception as e2:
    print(f"Error: {e2}")

data_to_save = {
    'all_t': t_sol,
    'all_w_sol': w_sol,
    'all_alpha_sol': alpha_sol,
    'all_T_sol': T_sol,
    'cost': cost,
    'total_cost': total_cost,
    'cost_graph': cost_graph,
    'omega_axis': omega_axis,
    'objective': objective,
    'cost_expr': cost_expr
}
savemat('output.mat', data_to_save)

