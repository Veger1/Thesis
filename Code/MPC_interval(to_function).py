import numpy as np
import sys
import time
from casadi import sum1
from matplotlib import pyplot as plt
from rockit import Ocp, MultipleShooting
from init_helper import load_data, initialize_constants

start_time = time.time()

# Load data and initialize constants
full_data = load_data()
helper, I_inv, R_pseudo, Null_R, Omega_max, w_initial, T_max = initialize_constants()

# Parameters
total_points = 1000  # Total points to simulate
horizon_length = 5  # Horizon length for MPC
update_interval = 5  # Solve MPC every 3 timesteps

# Initialize storage for results
actual_w = np.zeros((4, total_points))  # Store actual past trajectory
all_alpha = np.zeros(total_points)  # Store all control actions
all_T_rw = np.zeros((4, total_points))  # Store all reaction wheel torques

# Initial state
w_current = w_initial
alpha_buffer = np.zeros(update_interval)  # Buffer to hold 3 control actions
alpha_buffer_next = np.zeros(update_interval)


ocp = Ocp(t0=0, T=horizon_length / 10)
w = ocp.state(4)
alpha = ocp.control()
T_sc = ocp.parameter(3, grid='control')

w0 = ocp.parameter(4)  # Initial state/guess parameter
ocp.set_value(w0, w_initial)

data = full_data[:, :horizon_length]  # Input torque profile CHANGE
ocp.set_value(T_sc, data)

T_rw = R_pseudo @ T_sc + Null_R @ alpha  # Model/Dynamics
der_state = I_inv @ T_rw
ocp.set_der(w, der_state)

ocp.subject_to(-T_max <= (T_rw <= T_max))  # Constraints
ocp.subject_to(-Omega_max <= (w <= Omega_max))

ocp.subject_to(ocp.at_t0(w) == w0)
ocp.set_initial(w, w0)

a = 0.01
objective_expr_casadi = np.exp(-a * w ** 2)
objective = ocp.integral(sum1(objective_expr_casadi))
ocp.add_objective(objective)

solver_opts = {
    "print_time": False,
    "ipopt": {"print_level": 0, "sb": "yes"}
}
ocp.solver('ipopt', solver_opts)
ocp.method(MultipleShooting(N=horizon_length, M=1, intg='rk'))

constraint = ocp.sample(T_sc, grid='control-')[1]
states = ocp.sample(w, grid='control')[1]
controls = ocp.sample(alpha, grid='control-')[1]
torque = ocp.sample(T_rw, grid='control')[1]

prob_solve = ocp.to_function('prob_solve', [w0, constraint],
                             [states, controls, torque, constraint])

# MPC loop
for i in range(total_points):
    sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
    sys.stdout.flush()

    if i % update_interval == 0:  # Solve MPC every 3 steps
        alpha_buffer = alpha_buffer_next
        data = np.hstack([full_data[:, i:i + 1]] * horizon_length)
        w_sol, alpha_sol, torque_sol, _ = prob_solve(w_current, data)
        alpha_buffer_next[:] = alpha_sol[:update_interval]
        ocp.set_value(w0, w_current)
        # Store first 3 control actions

    j = i % update_interval
    data_live = full_data[:, i]
    alpha = alpha_buffer[j].reshape(1)
    torque = R_pseudo @ data_live + Null_R @ alpha
    w_current = w_current + I_inv @ torque.reshape(-1, 1) * 0.1

    actual_w[:, i] = w_current.reshape(-1)  # Store actual trajectory
    all_alpha[i] = alpha_buffer[j]  # Apply buffered control action
    all_T_rw[:, i] = torque  # Store first torque action

end_time = time.time()
execution_time = end_time - start_time
print(f"\nScript took {execution_time:.4f} seconds.")
plt.plot(actual_w.T)
plt.show()

