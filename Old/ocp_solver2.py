from casadi import sumsqr, sum1
from rockit import Ocp, MultipleShooting
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from init_helper import load_data, initialize_constants


# Load data and constants
test_data_T = load_data()
helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()
scaling = 1.0
N = 100
# test_data_T = test_data_T[:, :N]
test_data_T = test_data_T[:, 100:200]
# print(test_data_T.transpose().shape)

ocp = Ocp(T=10.0)  # Create the OCP (Optimal Control Problem)

# Define states and controls
w = ocp.state(4)  # 4 states for angular velocities
alpha = ocp.control()  # control input (acceleration)

# Set torque as a parameter (3 torques, using 'control' grid)
T_sc = ocp.parameter(3, grid='control')
ocp.set_value(T_sc, test_data_T)  # Assign the loaded test data

# Define the dynamics
T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw*scaling @ alpha
der_state = I_inv @ T_rw
ocp.set_der(w, der_state)

# Add torque constraints
ocp.subject_to(-T_max <= (T_rw <= T_max))

# Add saturation constraints
ocp.subject_to(-Omega_max <= (w <= Omega_max))

# Set initial conditions
# ocp.subject_to(ocp.at_t0(w) == Omega_start)
# ocp.set_initial(w, Omega_start)
# ocp.set_initial(alpha, 0)
w_zero = np.array([-324.86531751,  327.53246321, -273.18060751, 154.85590502])
ocp.subject_to(ocp.at_t0(w) == w_zero)
ocp.subject_to(ocp.at_t0(alpha) == -2.1019368227193738e-06)

# Define the cost function terms
c1 = 10
c2 = 0.001
w_res = 30

# Compute the exponential cost terms
exp_terms = c1 * np.exp(-c2 * (w**2))
exp_terms_res = c1 * np.exp(-c2 * ((w - w_res)**2))

# Define the objective function (minimize the sum of squares of exp terms)
objective_expr = ocp.integral(sum1(exp_terms))
objective_expr_res = ocp.integral(sum1(exp_terms_res))
# Add the objective to the OCP
ocp.add_objective(objective_expr)
# ocp.add_objective(objective_expr_res)

# Pick the solution method
ocp.solver('ipopt')

# Use Multiple Shooting method for solving
ocp.method(MultipleShooting(N=N, M=1, intg='rk'))

# Solve the problem
sol = ocp.solve()

# Post-processing and plotting
ts, w_sol = sol.sample(w, grid='control')
_, alpha_sol = sol.sample(alpha, grid='control')
_, T_rw_sol = sol.sample(T_rw, grid='control')
rpm_sol = helper.rad_to_rpm(w_sol)
_, T_sc_sol = sol.sample(T_sc, grid='control')

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
plt.plot(ts, T_rw_sol, '-')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Torque vs Time')

# Plot Alpha vs Time
plt.figure()
plt.plot(ts, alpha_sol, 'o-')
# plt.stairs(all_alpha_sol)
plt.xlabel('Time (s)')
plt.ylabel('Alpha')
plt.title('Alpha vs Time')

plt.show()

# Organize arrays into a dictionary
data_dict = {
    'alpha': alpha_sol,
    'time': ts,
    'torque': T_rw_sol,
    'omega': w_sol,
    'input': test_data_T
}

# Save the data to a .mat file
savemat('my_data.mat', data_dict)

