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
# w_current = np.random.uniform(-100, 100, (4, 1))
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

def squared_omega(torque_sc, omega):
    omega_new = omega + 0.1*I_inv @ R_pseudo @ torque_sc
    alpha =  constrained_alpha(omega_new)
    return -alpha/(0.1*41802.61224523921)


def pseudo(torque_sc, omega):
    return np.zeros((1, 1))

def optimal_alpha(omega):
    opt_alpha  = (- omega[0] + omega[1] - omega[2] + omega[3])/4
    return opt_alpha

def line_constraint(omega):
    segments = np.zeros((4, 2))
    for i in range(4):
        first = (omega_min - omega[i])*(-1) ** i
        second = (-omega_min - omega[i])*(-1) ** i
        segments[i] = np.sort([first[0], second[0]]).flatten() # Potentially sort manually
    return segments

def overlap_constraint(segments):
    segments = segments[np.argsort(segments[:, 0])]
    merged = [segments[0]]

    for current in segments[1:]:
        last_merged = merged[-1]

        if current[0] <= last_merged[1]:
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)
    return np.array(merged)

def constrained_alpha(omega):
    segments = line_constraint(omega)
    segments = overlap_constraint(segments)
    opt_alpha = optimal_alpha(omega)
    for segment in segments:
        start, end = segment[0], segment[1]

        if start <= opt_alpha <= end:
            dist_start = abs(opt_alpha - start)
            dist_end = abs(opt_alpha - end)

            if dist_start == dist_end:
                return start if abs(start) < abs(end) else end
            else:
                return start if dist_start < dist_end else end

    return opt_alpha

def omega_squared(omega):
    return omega + np.array([[1], [-1], [1], [-1]]) * constrained_alpha(omega)

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


def sum_squared(omega):
    return np.sum(omega**2)

def repeat_omega_squared(omega):
    for i in range(omega.shape[1]):
        omega[:, i] = omega_squared(omega[:, i].reshape(4, 1)).flatten()
    return omega


def convert_to_index(matrix):
    signs = (matrix >= 0).astype(int)  # Convert positive numbers to 1, negative to 0
    indices = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3] * 1  # Binary to decimal conversion
    return indices

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

omega_min = 10

solve(squared_omega)
w2 = w_sol.copy()
w1 = repeat_omega_squared(w_sol)


sumsq1 = np.sum(w1**2, axis=0)
sumsq2 = np.sum(w2**2, axis=0)
diff = sumsq1 - sumsq2
# Create subplots
fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

# Plot the first plot in the first subplot
ax[0].plot(w1.T)
ax[0].set_title('w1')

# Plot the second plot in the second subplot
ax[1].plot(w2.T)
ax[1].set_title('w2')
y_min = min(w1.min(), w2.min())
y_max = max(w1.max(), w2.max())
for a in ax:
    a.set_ylim(y_min, y_max)
# Show the plot
plt.tight_layout()  # Adjust spacing between subplots
plt.show()

index = convert_to_index(w1)
plt.plot(index)
plt.show()







