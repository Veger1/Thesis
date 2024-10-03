from casadi import vertcat, MX, Function, sumsqr
from rockit import *
import numpy as np
from pylab import *
from scipy.linalg import null_space
import scipy.io
import matplotlib.pyplot as plt

from helper import Helper

# create an instance of the helper class
helper = Helper()

data = scipy.io.loadmat('Slew1.mat')

test_data = data['Test']

test_data_T = test_data.T
test_data_T = test_data_T[:, 0:101]

Irw = 0.00002392195
I = np.diag([Irw]*4)
I_inv = np.linalg.inv(I)
beta = np.radians(60)

R_brw = np.array([[np.sin(beta), 0, -np.sin(beta), 0],
                 [0, np.sin(beta), 0, -np.sin(beta)],
                 [np.cos(beta), np.cos(beta), np.cos(beta), np.cos(beta)]])

R_rwb_pseudo = np.linalg.pinv(R_brw)  # Pseudo-inverse
Null_Rbrw = null_space(R_brw) * 2  # Null space
Null_Rbrw_T = Null_Rbrw.T  # Transpose of the null space

Omega_max = helper.rpm_to_rad(6000)
Omega_min = helper.rpm_to_rad(300)
Omega_tgt = helper.rpm_to_rad(1000)

Omega_ref = Omega_tgt * Null_Rbrw  # Similar to MATLAB matrix operation
Omega_start = Omega_ref / 2

T_max = 2.5 * 10**-3

# Time vector (8004 points from 0 to 800 seconds)
# t = np.linspace(0, 800, 8004)
t = np.linspace(0, 10, 101)
N = len(t)

ocp = Ocp(T=10.0)

w = ocp.state(4)  # 4 states
alpha = ocp.control()

T_sc = ocp.parameter(3, grid='control')
ocp.set_value(T_sc, test_data_T)  # Set the required torque as input

# Dynamics
T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw @ alpha
der_state = I_inv @ T_rw
ocp.set_der(w, der_state)

# Torque constraint
ocp.subject_to(-T_max <= (T_rw <= T_max))

# Saturation constraint
ocp.subject_to(-Omega_max <= (w <= Omega_max))

# Initial conditions
ocp.subject_to(ocp.at_t0(w) == Omega_start)
ocp.set_initial(w, Omega_start)
ocp.set_initial(alpha, 0)  # Unnecessary?

c1 = 1
c2 = 0.01

# Compute the exponential for each component in a vectorized manner
exp_terms = c1 * exp(-c2 * (w**2))

# Sum the exponential terms
objective_expr = ocp.integral(sumsqr(exp_terms))  # Zhang includes time

# Add the objective to the OCP
ocp.add_objective(objective_expr)

# Pick a solution method
ocp.solver('ipopt')

# Make it concrete for this ocp
ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
# ocp.method(DirectCollocation(N=N))

# solve
sol = ocp.solve()

"""
# Post-processing
"""

ts, w_sol = sol.sample(w, grid='control')
_, alpha_sol = sol.sample(alpha, grid='control')
_, T_rw_sol = sol.sample(T_rw, grid='control')
rpm_sol = helper.rad_to_rpm(w_sol)

plt.figure()
plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
plt.axhline(y=0, color='r', linestyle='--', label=f'rpm=0')
plt.plot(ts, rpm_sol, '-')
plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.title('RPM vs Time')

plt.figure()
plt.axhline(y=T_max, color='r', linestyle='--', label=f'T_max')
plt.axhline(y=-T_max, color='r', linestyle='--', label=f'-T_max')
plt.plot(ts, T_rw_sol, '-')
plt.xlabel('Time (s)')
plt.ylabel('Torque (Nm)')
plt.title('Torque vs Time')
plt.show()



