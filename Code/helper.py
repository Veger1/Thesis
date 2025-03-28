"""
This file contains helper functions that are used in the main code.
"""
import numpy as np
from sympy import symbols, lambdify
from config import OMEGA_START, R_PSEUDO, IRW, R


def rpm_to_rad(rpm):
    return rpm * 2 * np.pi / 60

def rad_to_rpm(rad):
    return rad * 60 / (2 * np.pi)

def convert_expr(ocp, objective_expr, w):
    ocp_t = ocp.t
    w_sym, t_sym = symbols('w t')
    objective_expr_casadi = lambdify((w_sym, t_sym), objective_expr, 'numpy')
    objective_expr_casadi = objective_expr_casadi(w, ocp_t)
    return objective_expr_casadi

def exponential_moving_average(signal, alpha=0.1):
    """Apply exponential moving average to smooth the signal."""
    smoothed = np.zeros_like(signal)
    smoothed[:, 0] = signal[:, 0]  # Initialize with first value
    for i in range(1, signal.shape[1]):
        smoothed[:, i] = alpha * signal[:, i] + (1 - alpha) * smoothed[:, i-1]
    return smoothed

def hysteresis_filter(signal, low_threshold, high_threshold):
    """Apply hysteresis to prevent rapid switching."""
    flag = np.zeros(signal.shape[1], dtype=bool)
    active = False  # Start with no low torque detection

    for i in range(signal.shape[1]):
        if np.all(abs(signal[:, i]) < low_threshold):
            active = True
        elif np.all(abs(signal[:, i]) > high_threshold):
            active = False
        flag[i] = active
    return flag

def detect_transitions(signal):
    signal = np.array(signal, dtype=bool)  # Ensure it's a boolean NumPy array
    diff_signal = np.diff(signal.astype(int))  # Convert to int and compute difference

    rising_edges = np.where(diff_signal == 1)[0] + 1  # +1 to shift index correctly
    falling_edges = np.where(diff_signal == -1)[0] + 1

    return rising_edges, falling_edges

def pseudo_sol(data):
    length = data.shape[1]
    omega_sol = np.zeros((4, length + 1))
    omega_sol[:, 0] = OMEGA_START.flatten()
    for i in range(length):
        omega_sol[:, i + 1] = (omega_sol[:, i].reshape((4,1)) + 0.1 * R_PSEUDO @ data[:, i].reshape(3, 1) / IRW).flatten()
    return omega_sol

def total_momentum(data):
    omega_sol = pseudo_sol(data)
    momentum = R @ omega_sol
    return momentum

def nullspace_alpha(data):
    length = data.shape[1]
    alpha = np.zeros(length)
    for i in range(length):
        alpha[i] = (- data[0, i] + data[1, i] - data[2, i] + data[3, i]) / 4
    return alpha




