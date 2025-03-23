from scipy.io import savemat
import MPC
import MPC_fast
from sympy import symbols, exp, Abs, tanh, atan, pi
from plot_MPC import plot_mpc_evolution


w, t = symbols('w t')
a = 0.01
b = 1 / 700000
offset = 60
k = 1.0
# time_dependent = (1 + tanh(k * (t - offset))) / 2
objective = exp(-a * w ** 2) #+ time_dependent*b * w ** 2

# _, w_sol1, _, _ = MPC.solve_interval(objective, total_points = 1000)
# _, w_sol2, _, _ = MPC_fast.solve_interval(objective, total_points = 1000)

all_w, w_sol, all_alpha, all_torque = MPC.solve(objective, total_points = 8000, horizon=30)
# _, w_sol4, _, _ = MPC_fast.solve(objective, total_points = 1000)

data_to_save = {
    'all_w_sol': w_sol,
    'all_alpha_sol': all_alpha,
    'all_T_sol': all_torque,
    'all_w': all_w
}
savemat('Data/output.mat', data_to_save)

plot_mpc_evolution(w_sol, all_w, total_points=8000, horizon=30)
