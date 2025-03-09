import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import sum1
from rockit import Ocp, MultipleShooting
from scipy.io import savemat

from init_helper import load_data, initialize_constants
import sys
import time


full_data = load_data()
helper, I_inv, R_pseudo, Null_R, Omega_max, w_initial, T_max = initialize_constants()

total_points = 8000
w_current = w_initial
# w_current = np.random.uniform(-100, 100, (4, 1))  # 4x1 column vector
# alpha = np.linspace(-0.001, 0.001, 500).reshape(1, 500)

w_sol = np.zeros((4, total_points+1))
torque_sol = np.zeros((4, total_points+1))
w_sol[:, 0] = w_current.flatten()
alpha_sol = np.zeros((1, total_points))


def minmax_torque(torque_sc, omega):
    T_rw = R_pseudo @ torque_sc  # + Null_R @ alpha
    alpha_null = -T_rw / Null_R
    alpha_best = (max(alpha_null) + min(alpha_null)) / 2
    return alpha_best


def minmax_omega(torque_sc, omega):
    nominator = - omega - 0.1*I_inv @ R_pseudo @ torque_sc
    denominator = 0.1*I_inv @ Null_R
    alpha_null = nominator / denominator
    alpha_best = (max(alpha_null) + min(alpha_null)) / 2
    return alpha_best


def pseudo(torque_sc, omega):
    return np.zeros((1, 1))


def solve(calc_alpha_func=pseudo):
    global w_current
    for i in range(total_points):
        T_sc = full_data[:, i].reshape(3, 1)
        alpha = calc_alpha_func(T_sc, w_current).reshape(1, 1)
        alpha_sol[:, i] = alpha.flatten()
        T_rw = R_pseudo @ T_sc + Null_R @ alpha

        der_state = I_inv @ T_rw
        w_current = w_current + der_state * 0.1
        w_sol[:, i+1] = w_current.flatten()
        torque_sol[:, i] = T_rw.flatten()
    torque_sol[:, i] = T_rw.flatten()


def save(name='output'):
    data_to_save = {
        'all_w_sol': w_sol.T,
        'all_alpha_sol': alpha_sol,
        'all_T_sol': torque_sol.T,
        'all_t': np.linspace(0, total_points/10, total_points+1)
    }
    savemat(f'Data/Realtime/{name}.mat', data_to_save)


def plot():
    plt.plot(w_sol.T)
    plt.show()
    plt.plot(alpha_sol.T)
    plt.show()


solve()
save('pseudo')
# plot()


