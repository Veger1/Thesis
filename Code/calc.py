from sympy import symbols, exp, Abs, tanh, atan, pi
from Code.repeat_solver import *
from scipy.io import savemat

num_intervals, N = 8, 500
scaling, time = 1.0, float(N/10)

w, t = symbols('w t')
k = 1.0
a = 0.001
b = 1/700000
X, Y = -pi*10, pi*10
offset = 60
tanh_expr = (tanh(k * (w - X))) * 0.5 + (tanh(-k * (w - Y))) * 0.5# Hyperbolic tangent function
gaus_expr = exp(-a*w**2)  # Gaussian function
speed_expr = b*w**2
lin_expr = b*w
time_dependent_expr = (1+tanh(k*(t-offset)))/2
cost_expr = gaus_expr + speed_expr*time_dependent_expr

t_sol, w_sol, alpha_sol, T_sol = 0, 0, 0, 0
cost, total_cost, cost_graph, omega_axis = 0, 0, 0, 0

try:  # Skip this part if optimization has already been done
    t_sol, w_sol, alpha_sol, T_sol = solve_ocp(cost_expr, num_intervals, N, time, scaling)
except Exception as e1:
    print(f"Error: {e1}")

try:  # Import data from the previous optimization if exists
    cost, total_cost, cost_graph, omega_axis = calc_cost(w_sol, cost_expr, t_sol)
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
    'cost_expr': str(cost_expr)
}
savemat('Data/output.mat', data_to_save)


