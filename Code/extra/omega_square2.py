# alpha = 1/(4*t*J) * [W0,1 W0,2 W0,3 W0,4] * [1, -1, 1, -1] + [T1 T2 T3 T4]  * [1, -1, 1, 1]/4
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
from init_helper import load_data, initialize_constants


# Load data and constants
test_data_T = load_data()
helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()
scaling = 1.0
Irw = 0.00002392195
# Irw = 1.0
print(test_data_T.shape)
t = np.linspace(0, 800, 8004)
ts = t[1] - t[2]

omega = Omega_start
# omega = np.array([[100], [100] ,[100], [100]])  # Custom Initial velocity
print(omega)
all_alpha = np.zeros((1, len(t)))
dot_omega_cmd = 0
all_omega = np.zeros((4, len(t)))
all_max_torque = np.zeros(len(t))

for i in range(len(t)):
    data = test_data_T[:,i]
    T_pseudo = (R_rwb_pseudo @ data)
    omega_k = (omega*Irw/ts).flatten()
    T_virtual = T_pseudo + omega_k
    alpha = -1/4*np.dot(T_virtual, Null_Rbrw.flatten())


    all_alpha[0, i] = alpha
    all_omega[:, i] = omega.flatten()

    T_pseudo = (R_rwb_pseudo @ data)
    T_rw = R_rwb_pseudo @ data + Null_Rbrw.flatten() * alpha
    all_max_torque[i] = max(abs(T_rw))
    if max(abs(T_rw)) > T_max:
        pos = max(T_pseudo[0],T_pseudo[2])
        neg = max(T_pseudo[1],T_pseudo[3])
        # x + pos
        # -x + neg
    omega_dot = T_rw / Irw
    omega = omega.reshape(-1) + ts * omega_dot

    """
    Can give very large torques if large nullsapce omega is present. Add code to prevent this.
    """

# plot results
plt.figure()
for i in range(all_omega.shape[0]):
    plt.plot(t[1:], all_omega[i, 1:], label=f'Omega {i+1}')
plt.xlabel('Time (s)')
plt.ylabel('Omega')
plt.legend()

plt.figure()
plt.plot(t, all_alpha.flatten())

plt.figure()
plt.plot(t, all_max_torque)
plt.show()
