import timeit
from casadi import arctan
from sympy import symbols, exp, Abs, tanh, atan, pi
from Code.repeat_solver import solve_ocp

num_intervals, N = 8, 1000
scaling, time = 1.0, float(N/10)

w = symbols('w') # Use symbolic variable
k = 1.0
a = 0.001
b = 1/7000000
X, Y = -pi*10, pi*10
tanh_expr = (tanh(k * (w - X))) * 0.5 + (tanh(-k * (w - Y))) * 0.5# Hyperbolic tangent function
gaus_expr = exp(-a*w**2) # Gaussian function
speed_expr = b*w**2
lin_expr = b*w
cost_expr = gaus_expr

t_sol, w_sol, alpha_sol, T_sol = 0, 0, 0, 0
def run_solver():
    solve_ocp(cost_expr, num_intervals, N, time, scaling)
    pass

execution_time = timeit.timeit(run_solver, number=5)
print(f"Solver took {execution_time:.4f} seconds.")
