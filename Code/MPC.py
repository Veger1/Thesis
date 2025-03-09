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
total_points = 500  # Total points to simulate
horizon_length = 25  # Horizon length for MPC

# Initialize storage for results
all_w = np.zeros((4, horizon_length + 1, total_points))  # Store predictions over time
actual_w = np.zeros((4, total_points))  # Store actual past trajectory
all_alpha = np.zeros(total_points)  # Store all control actions
all_T_rw = np.zeros((4, total_points))  # Store all reaction wheel torques

# Initial state
w_current = w_initial

# MPC loop (now runs before any plotting)
for i in range(total_points):
    sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
    sys.stdout.flush()
    # Create OCP
    ocp = Ocp(t0=i/10, T=horizon_length / 10)
    w = ocp.state(4)  # Angular velocity (4 states)
    alpha = ocp.control()  # Control
    T_sc = ocp.parameter(3, grid='control')  # External torque profile

    # Set input torques
    data = full_data[:, i:i + horizon_length]  # Horizon torque data
    ocp.set_value(T_sc, data)

    # Dynamics
    T_rw = R_pseudo @ T_sc + Null_R @ alpha
    der_state = I_inv @ T_rw
    ocp.set_der(w, der_state)

    ocp.subject_to(-T_max <= (T_rw <= T_max))  # Reaction wheel torque limits
    ocp.subject_to(-Omega_max <= (w <= Omega_max))  # Angular velocity limits

    # Initial condition for the current horizon
    ocp.subject_to(ocp.at_t0(w) == w_current)
    if i != 0:
        ocp.set_initial(w, w_sol.T)

    # Objective function
    a = 0.1
    b = 1 / 700000
    offset = 30
    k = 1.0
    time_dependent = (1 + np.tanh(k * (ocp.t - offset))) / 2
    objective_expr_casadi = np.exp(-a * w ** 2) + time_dependent*b * w ** 2
    objective = ocp.integral(sum1(objective_expr_casadi))
    ocp.add_objective(objective)

    # Solve
    solver_opts = {
        "print_time": False,  # Suppress overall solver timing
        "ipopt": {
            "print_level": 0,  # Disable IPOPT output
            "sb": "yes"  # Suppress banner output
        }
    }
    ocp.solver('ipopt', solver_opts)
    ocp.method(MultipleShooting(N=horizon_length, M=1, intg='rk'))
    sol = ocp.solve()

    # Extract solutions
    ts, w_sol = sol.sample(w, grid='control')
    _, alpha_sol = sol.sample(alpha, grid='control')
    _, T_rw_sol = sol.sample(T_rw, grid='control')

    # Store results
    w_current = w_sol[1, :]
    actual_w[:, i] = w_sol[0, :]  # Store actual trajectory
    all_w[:, :, i] = w_sol.T  # Store full prediction
    all_alpha[i] = alpha_sol[0]
    all_T_rw[:, i] = T_rw_sol[0, :]

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
write = True
if write:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save('Data/MPC_output.mp4', writer=writer)
plt.show()
