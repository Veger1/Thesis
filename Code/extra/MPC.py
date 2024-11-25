import matplotlib.pyplot as plt
import numpy as np
from casadi import arctan
from sympy import symbols, exp, Abs, tanh, atan, pi
from Code.init_helper import load_data, initialize_constants
from Code.repeat_solver import solve_ocp, calc_cost, MPC
from scipy.io import savemat
import tkinter as tk
from tkinter import messagebox

test_data_T_full = load_data()  # This is the full test data (8004 samples)
helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()

N = 5
scaling, time = 1.0, float(N/10)
Irw, ts = 0.00002392195, 0.1

w = symbols('w') # Use symbolic variable
k = 1.0
a = 0.001
b = 1/7000000
X, Y = -pi*10, pi*10
tanh_expr = (tanh(k * (w - X))) * 0.5 + (tanh(-k * (w - Y))) * 0.5# Hyperbolic tangent function
gaus_expr = exp(-a*w**2) # Gaussian function
speed_expr = b*w**2
lin_expr = b*w
cost_expr = speed_expr

t_sol, w_sol, alpha_sol, T_sol = 0, 0, 0, 0
cost, total_cost, cost_graph, omega_axis = 0, 0, 0, 0

omega = Omega_start.flatten()
all_omega = []

# for i in range(len(test_data_T_full[0])-N):
for i in range(100):
    data = test_data_T_full[:,i:i+N]
    alpha = MPC(data, N, time, cost_expr, R_rwb_pseudo, Null_Rbrw, I_inv, Omega_max, T_max, scaling, Omega_start)
    T_rw = R_rwb_pseudo @ data[:,1] + Null_Rbrw.flatten() * alpha

    all_omega.append(omega.reshape(-1,1))
    omega_dot = T_rw / Irw
    omega = omega + ts * omega_dot
all_omega = np.concatenate(all_omega, axis=1).transpose()

plt.figure()
plt.plot(all_omega)
plt.show()

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
# savemat('Data/output.mat', data_to_save)



