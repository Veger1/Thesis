from config import *
from helper import *
from scipy.io import loadmat
import matplotlib.pyplot as plt

def line_constraint(speed):
    parts = np.zeros((4, 2))
    for i in range(4):
        first = (OMEGA_MIN - speed[i]) * (-1) ** i
        second = (-OMEGA_MIN - speed[i]) * (-1) ** i
        parts[i] = np.sort([first[0], second[0]]).flatten() # Potentially sort manually
    return parts

data = load_data('Data/Slew2.mat')
low_torque_flag = hysteresis_filter(data, 0.000005, 0.000015)
low_torque_flag[0:2] = False
rising, falling = detect_transitions(low_torque_flag)
falling = np.insert(falling, 0, 0)
if len(rising) == len(falling):
    section0 = data[:, rising[0]:falling[0]]
    section1 = data[:, rising[1]:falling[1]]
    section2 = data[:, rising[2]:falling[2]]
    section3 = data[:, rising[3]:falling[3]]
else:
    print("Error: rising and falling edges do not match")
    print(rising, falling)
    exit()

momentum4 = pseudo_sol(data)
momentum3 = R @ momentum4
momentum0 = R_PSEUDO @ momentum3


time = np.linspace(0, 800, 8005)
alpha = nullspace_alpha(momentum4)

begin = np.zeros((8, 4))
begin_order = np.array([1, 1, 2, 2, 3, 3, 4, 4])
for j in range(len(falling)):
    for i in range(4):
        begin_omega = momentum4[:, falling[j]]
        begin[2 * i, j] = (OMEGA_MIN - begin_omega[i]) * (-1) ** i
        begin[1 + 2 * i, j] = (-OMEGA_MIN - begin_omega[i]) * (-1) ** i

end = np.zeros((8, 4))
end_order = np.array([1, 1, 2, 2, 3, 3, 4, 4])
for j in range(len(rising)):
    for i in range(4):
        end_omega = momentum4[:, rising[j]]
        end[2 * i, j] = (OMEGA_MIN - end_omega[i]) * (-1) ** i
        end[1 + 2 * i, j] = (-OMEGA_MIN - end_omega[i]) * (-1) ** i
print(end)
length = momentum4.shape[1]
segments = np.zeros((8, length ))
for j in range(length):
    omega = momentum4[:, j]
    for i in range(4):
        segments[2*i, j] = (OMEGA_MIN - omega[i]) * (-1) ** i
        segments[1+2*i, j] = (-OMEGA_MIN - omega[i]) * (-1) ** i


fig, ax = plt.subplots(1, 1, figsize=(9, 6))
ax.plot(time, segments.T, color='gray')

ax.plot(time, alpha, color='black', linestyle='--', label='Alpha')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i in range(0, 8, 2):
    color = colors[i // 2 % len(colors)]  # Cycle through colors
    plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
plt.xlabel("Time (s)")
plt.ylabel("Nullspace component")
plt.title("Gray Bands from 8xN Array")
plt.legend()
plt.show()

