import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import sum1
from rockit import Ocp, MultipleShooting
from init_helper import load_data, initialize_constants
import sys
import time


full_data = load_data()
helper, I_inv, R_pseudo, Null_R, Omega_max, w_initial, T_max = initialize_constants()

total_points = 8000
w_current = w_initial
w_current = np.random.uniform(-100, 100, (4, 1))  # 4x1 column vector
alpha = np.linspace(-0.001, 0.001, 500).reshape(1, 500)

w_sol = np.zeros((4, total_points+1))
w_sol[:, 0] = w_current.flatten()

for i in range(total_points):
    T_sc = full_data[:, i].reshape(3, 1)
    # alpha = calc_alpha(T_sc, w_current)
    T_rw = R_pseudo @ T_sc + Null_R @ alpha
    T_rw = abs(T_rw)
    if i == 2100:
        plt.plot(alpha.flatten(), T_rw.T, )
        T_rw = R_pseudo @ T_sc # + Null_R @ alpha
        plt.plot([0], abs(max(T_rw)), 'ro')
        plt.plot([0], abs(min(T_rw)), 'ro')
        plt.plot([0], (abs(max(T_rw))+abs(min(T_rw)))/2, 'ro')
        plt.show()

    T_rw = R_pseudo @ T_sc
    der_state = I_inv @ T_rw
    w_current = w_current + der_state * 0.1
    w_sol[:, i+1] = w_current.flatten()


def calc_alpha(T_sc, w_current):
    pass


# plt.plot(w_sol.T)
# plt.show()

