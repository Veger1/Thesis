from casadi import sumsqr
from rockit import Ocp, MultipleShooting
import numpy as np
import matplotlib.pyplot as plt
from init_helper import load_data, initialize_constants
from scipy.io import savemat

# Load data and constants
test_data_T_full = load_data()  # This is the full test data (8004 samples)
helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()

# Number of intervals to process (each interval has 101 samples)
num_intervals = 3
scaling = 1.0

# Time vector for each interval (101 points, corresponding to 10 seconds per interval)
N = 101
time = 10.0

w_initial = Omega_start  # Start with the initial condition
alpha_initial = 0

all_t = []
all_w_sol = []
all_alpha_sol = []
all_T_sol = []

for i in range(num_intervals):
    ocp = Ocp(t0=time*i, T=time)  # Create the OCP for the current interval

    w = ocp.state(4)  # 4 states for angular velocities
    alpha = ocp.control()  # control input (acceleration)
    T_sc = ocp.parameter(3, grid='control')  # Set torque as a parameter (3 torques, using 'control' grid)

    test_data_T = test_data_T_full[:, i*N:(i+1)*N ]  # Extract N points from data
    ocp.set_value(T_sc, test_data_T)  # Assign the torque constraint

    T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw * scaling @ alpha
    der_state = I_inv @ T_rw
    ocp.set_der(w, der_state)

    ocp.subject_to(-T_max <= (T_rw <= T_max))  # Add torque constraints
    ocp.subject_to(-Omega_max <= (w <= Omega_max))  # Add saturation constraints

    # Set initial conditions
    ocp.subject_to(ocp.at_t0(w) == w_initial)
    ocp.set_initial(w, w_initial)  # Set initial guess

    c1, c2 = 1, 0.01   # Define the cost function terms
    exp_terms = c1 * np.exp(-c2 * (w ** 2))  # Compute the exponential cost terms
    objective_expr = ocp.integral(sumsqr(exp_terms))  # Define the objective function
    ocp.add_objective(objective_expr)  # Add the objective to the OCP

    ocp.solver('ipopt')  # Use IPOPT solver
    ocp.method(MultipleShooting(N=N, M=1, intg='rk'))

    sol = ocp.solve()  # Solve the problem

    # Post-processing: Sample solutions for this interval
    ts, w_sol = sol.sample(w, grid='control')
    _, alpha_sol = sol.sample(alpha, grid='control')
    _, T_rw_sol = sol.sample(T_rw, grid='control')

    # Update the initial condition for the next interval
    w_initial = w_sol[-1]  # Last value of the current interval

    all_t.append(ts[:-1])
    all_w_sol.append(w_sol[:-1])  # Last value is unique
    all_alpha_sol.append(alpha_sol[:-1])
    all_T_sol.append(T_rw_sol[:-1])


# Concatenate the data along the first axis (if they're all 1D arrays)
all_t = np.concatenate(all_t)
all_w_sol = np.concatenate(all_w_sol)
all_alpha_sol = np.concatenate(all_alpha_sol)
all_T_sol = np.concatenate(all_T_sol)


# Plot RPM vs Time
plt.figure()
plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
plt.axhline(y=0, color='r', linestyle='--', label=f'rpm=0')
plt.plot(all_t, helper.rad_to_rpm(all_w_sol), '-')
plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.title('RPM vs Time')

# Plot Torque vs Time
plt.figure()
plt.axhline(y=T_max, color='r', linestyle='--', label=f'T_max')
plt.axhline(y=-T_max, color='r', linestyle='--', label=f'-T_max')
plt.plot(all_t, all_T_sol, '-')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Torque vs Time')

# Plot Alpha vs Time
plt.figure()
plt.axhline(y=T_max, color='r', linestyle='--', label=f'T_max')
plt.axhline(y=-T_max, color='r', linestyle='--', label=f'-T_max')
plt.plot(all_t, all_alpha_sol, 'o-')
# plt.stairs(all_alpha_sol)
plt.xlabel('Time (s)')
plt.ylabel('Alpha')
plt.title('Alpha vs Time')

plt.figure()
plt.plot(all_t)

plt.show()
