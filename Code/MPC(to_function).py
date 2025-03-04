import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import sum1
from rockit import Ocp, MultipleShooting
from init_helper import load_data, initialize_constants
import sys
import time

start_time = time.time()

# Load data and initialize constants
full_data = load_data()
helper, I_inv, R_pseudo, Null_R, Omega_max, w_initial, T_max = initialize_constants()

# Parameters
total_points = 8000  # Total points to simulate
horizon_length = 20  # Horizon length for MPC

# Initialize storage for results
all_w = np.zeros((4, horizon_length + 1, total_points))  # Store predictions over time
actual_w = np.zeros((4, total_points))  # Store actual past trajectory
all_alpha = np.zeros(total_points)  # Store all control actions
all_T_rw = np.zeros((4, total_points))  # Store all reaction wheel torques

# Initial state
w_current = w_initial

ocp = Ocp(t0=0, T=horizon_length/10)
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
ocp.set_initial(w, w0)  # Only work for FIRST interval, does not update for subsequent intervals

a = 0.01
b = 1 / 700000
offset = 60
k = 1.0
time_dependent = (1 + np.tanh(k * (ocp.t - offset))) / 2
objective_expr_casadi = np.exp(-a * w ** 2) #+ time_dependent*b * w ** 2
objective = ocp.integral(sum1(objective_expr_casadi))
ocp.add_objective(objective)

solver_opts = {
        "print_time": False,  # Suppress overall solver timing
        "ipopt": {
            "print_level": 0,  # Disable IPOPT output
            "sb": "yes"  # Suppress banner output
        }
    }
ocp.solver('ipopt', solver_opts)
ocp.method(MultipleShooting(N=horizon_length, M=1, intg='rk'))

constraint = ocp.sample(T_sc, grid='control-')[1]  # input/output

states = ocp.sample(w, grid='control')[1]  # output
controls = ocp.sample(alpha, grid='control-')[1]
torque = ocp.sample(T_rw, grid='control')[1]

prob_solve = ocp.to_function('prob_solve', [w0, constraint],
                             [states, controls, torque, constraint])

# MPC loop (now runs before any plotting)
for i in range(total_points):
    sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
    sys.stdout.flush()

    #data = full_data[:, i:i + horizon_length]  # Horizon torque data
    data = np.hstack([full_data[:, i:i + 1]] * horizon_length)
    w_sol, alpha_sol, torque_sol, _ = prob_solve(w_current, data)
    w_current = w_sol[:, 1]
    ocp.set_value(w0, w_current)
    # ocp.set_initial(w, w_current)  # Does nothing

    w_sol = np.array(w_sol)
    torque_sol = np.array(torque_sol)

    actual_w[:, i] = w_sol[:, 0]  # Store actual trajectory
    all_w[:, :, i] = w_sol  # Store full prediction
    all_alpha[i] = alpha_sol[0]
    all_T_rw[:, i] = torque_sol[:, 0]

end_time = time.time()
execution_time = end_time - start_time
print(f"Script took {execution_time:.4f} seconds.")

# --- PLOTTING (after calculations) ---
fig, ax = plt.subplots(figsize=(10, 6))
actual_lines = [ax.plot([], [], label=f"State {j + 1}", linestyle="-")[0] for j in range(4)]
pred_lines = [ax.plot([], [], linestyle="dotted")[0] for j in range(4)]

ax.set_xlim(0, total_points + horizon_length)
ax.set_ylim(-Omega_max, Omega_max)
ax.set_title("MPC Prediction vs Actual Evolution")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Angular Velocity w")
ax.legend()
ax.grid()


def update_plot(frame):
    """Update live plot with new predictions and actual trajectory."""
    if frame >= total_points:
        return actual_lines + pred_lines  # Stop updating when simulation ends

    for j in range(4):
        # Actual values up to time `frame`
        actual_lines[j].set_data(range(frame), actual_w[j, :frame])

        # MPC prediction for `frame`
        pred_lines[j].set_data(range(frame, frame + horizon_length + 1), all_w[j, :, frame])

    return actual_lines + pred_lines


ani = animation.FuncAnimation(fig, update_plot, frames=total_points, interval=10, blit=True)
write = False
if write:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('Data/MPC_output.mp4', writer=writer)
plt.show()
