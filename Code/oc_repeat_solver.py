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
num_intervals = 8

# Time vector for each interval (101 points, corresponding to 10 seconds per interval)
N = 1000
time = 100.0
t = np.linspace(0, time, N)
ts = np.linspace(0, time*num_intervals, N*num_intervals+num_intervals)

# Initialize variables for storing solutions
all_w_solutions = []
all_alpha_solutions = []
all_T_rw_solutions = []
all_time_solutions = []
w_initial = Omega_start  # Start with the initial condition
alpha_initial = 0

# Loop through each interval
for i in range(num_intervals):
    print(f"Solving interval {i + 1}/{num_intervals}")

    # Calculate the start index for the current interval
    start_index = i * (N - 1)  # Start index based on overlap
    test_data_T = test_data_T_full[:, start_index:start_index + N]  # Extract torque data for current interval
    print("testdata", len(test_data_T.shape))

    # Create the OCP for the current interval
    ocp = Ocp(T=time)

    # Define states and controls
    w = ocp.state(4)  # 4 states for angular velocities
    alpha = ocp.control()  # control input (acceleration)

    # Set torque as a parameter (3 torques, using 'control' grid)
    T_sc = ocp.parameter(3, grid='control')
    ocp.set_value(T_sc, test_data_T)  # Assign the torque data for the current interval

    # Define the dynamics
    T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw @ alpha
    der_state = I_inv @ T_rw
    ocp.set_der(w, der_state)

    # Add torque constraints
    ocp.subject_to(-T_max <= (T_rw <= T_max))

    # Add saturation constraints
    ocp.subject_to(-Omega_max <= (w <= Omega_max))

    # Set initial conditions
    ocp.subject_to(ocp.at_t0(w) == w_initial)
    ocp.set_initial(w, w_initial)
    ocp.set_initial(alpha, 0)
    if i != 0:
        ocp.subject_to(ocp.at_t0(alpha) == alpha_initial)

    # Define the cost function terms
    c1 = 1  # cost/penalty weight
    c2 = 0.01  # s²/rad²
    d1 = 10
    d2 = 0.1

    # Compute the exponential cost terms
    exp_terms = c1 * np.exp(-c2 * (w ** 2))
    exp_terms_0 = d1 * np.exp(-d2 * (w ** 2))

    # Define the objective function (minimize the sum of squares of exp terms)
    objective_expr = ocp.integral(sumsqr(exp_terms))
    objective_expr_0 = ocp.integral(sumsqr(exp_terms_0))

    # Add the objective to the OCP
    ocp.add_objective(objective_expr)
    ocp.add_objective(objective_expr_0)

    # Pick the solution method
    ocp.solver('ipopt')

    # Use Multiple Shooting method for solving
    ocp.method(MultipleShooting(N=N, M=1, intg='rk'))

    # Solve the problem
    sol = ocp.solve()

    # Post-processing: Sample solutions for this interval
    _, w_sol = sol.sample(w, grid='control')
    _, alpha_sol = sol.sample(alpha, grid='control')
    _, T_rw_sol = sol.sample(T_rw, grid='control')

    # Store solutions
    all_w_solutions.append(w_sol)
    all_alpha_solutions.append(alpha_sol)
    all_T_rw_solutions.append(T_rw_sol)

    # Update the initial condition for the next interval
    w_initial = w_sol[-1]  # Last value of the current interval

# Convert lists to arrays for plotting
all_w_solutions = np.concatenate(all_w_solutions, axis=0)
all_alpha_solutions = np.concatenate(all_alpha_solutions, axis=0)
all_T_rw_solutions = np.concatenate(all_T_rw_solutions, axis=0)

# Convert angular velocity to RPM
rpm_sol = helper.rad_to_rpm(all_w_solutions)

# Plot RPM vs Time
plt.figure()
plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
plt.axhline(y=0, color='r', linestyle='--', label=f'rpm=0')
plt.plot(ts, rpm_sol, '-')
plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.title('RPM vs Time')

# Plot Torque vs Time
plt.figure()
plt.axhline(y=T_max, color='r', linestyle='--', label=f'T_max')
plt.axhline(y=-T_max, color='r', linestyle='--', label=f'-T_max')
plt.plot(ts, all_T_rw_solutions, '-')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Torque vs Time')

# Plot Alpha vs Time
plt.figure()
plt.axhline(y=T_max, color='r', linestyle='--', label=f'T_max')
plt.axhline(y=-T_max, color='r', linestyle='--', label=f'-T_max')
plt.plot(ts, all_alpha_solutions, '-')
plt.xlabel('Time (s)')
plt.ylabel('Alpha')
plt.title('Alpha vs Time')

plt.show()

# Organize arrays into a dictionary
data_dict = {
    'alpha': all_alpha_solutions,
    'time': ts,
    'torque': all_T_rw_solutions,
    'omega': all_w_solutions
}

# Save the data to a .mat file
savemat('my_data.mat', data_dict)

