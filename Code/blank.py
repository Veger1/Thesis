from matplotlib import pyplot as plt
from scipy.io import loadmat, savemat
from scipy.integrate import cumulative_trapezoid as cumtrapz
import numpy as np
from init_helper import load_data, initialize_constants

test_data_T_full = load_data()
helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()

T_rw_sol = R_rwb_pseudo @ test_data_T_full
ts = np.linspace(0, 800, 8004)
der_state = I_inv @ T_rw_sol
w0 = cumtrapz(der_state[0], ts, initial=0)
w1 = cumtrapz(der_state[1], ts, initial=0)
w2 = cumtrapz(der_state[2], ts, initial=0)
w3 = cumtrapz(der_state[3], ts, initial=0)
w = np.array([w0, w1, w2, w3])
all_w_sol = w.transpose()+Omega_start.transpose()

limit = helper.rpm_to_rad(300)
plt.figure()
plt.axhline(y=limit, color='r', linestyle='--', label=f'rpm=300')
plt.axhline(y=-limit, color='r', linestyle='--', label=f'rpm=-300')
plt.axhline(y=0, color='r', linestyle='--', label=f'rpm=0')
plt.plot(ts,all_w_sol, 'o-')

plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.title('RPM vs Time')

plt.show()

data_to_save = {
    'all_w_sol': all_w_sol,
    # 'all_t': None,
    # 'all_alpha_sol': None,
    # 'all_T_sol': None
}
# Save to a .mat file
savemat('MPI.mat', data_to_save)

