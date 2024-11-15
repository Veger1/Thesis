from matplotlib import pyplot as plt
from scipy.io import loadmat
from init_helper import load_data, initialize_constants

# Load the .mat file
loaded_data = loadmat('Data/50s/stic2.mat')

# Extract variables
all_t = loaded_data['all_t'].transpose()
all_w_sol = loaded_data['all_w_sol']
all_alpha_sol = loaded_data['all_alpha_sol'].transpose()
all_T_sol = loaded_data['all_T_sol']

test_data_T_full = load_data()  # This is the full test data (8004 samples)
helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()

# Plot RPM vs Time
fig, ax1 = plt.subplots()
ax1.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
ax1.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
ax1.axhline(y=0, color='r', linestyle='--', label=f'rpm=0')
ax1.plot(all_t, helper.rad_to_rpm(all_w_sol[:,1]), 'g-')
ax1.plot(all_t, helper.rad_to_rpm(all_w_sol[:,3]), 'g--')
ax1.set_xlabel('X-axis')
ax1.set_ylabel('RPM', color='g')
ax1.tick_params(axis='y', labelcolor='g')

ax2 = ax1.twinx()
ax2.plot(all_t, helper.rad_to_rpm(all_w_sol[:,0]), 'b-')
ax2.plot(all_t, helper.rad_to_rpm(all_w_sol[:,2]), 'b--')
ax2.set_ylabel('RPM', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.invert_yaxis()
ax1.set_ylim(-6000, 6000)
ax2.set_ylim(6000, -6000)

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
plt.plot(all_t, all_alpha_sol, '-')
# plt.stairs(all_alpha_sol)
plt.xlabel('Time (s)')
plt.ylabel('Alpha')
plt.title('Alpha vs Time')

plt.figure()
plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
plt.axhline(y=0, color='r', linestyle='--', label=f'rpm=0')
plt.plot(all_t, helper.rad_to_rpm(all_w_sol), 'o-')

plt.xlabel('Time (s)')
plt.ylabel('RPM')
plt.title('RPM vs Time')

plt.show()


