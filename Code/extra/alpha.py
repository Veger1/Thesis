import matplotlib.pyplot as plt
import numpy as np

vector = np.random.rand(4)
norm_rico = vector / np.linalg.norm(vector)
b = np.random.rand(4)
c = np.random.rand(4)
alpha = np.linspace(-5, 5, 1000)

def omega(alpha, rico, b):
    return abs(rico * alpha + b)

def sq_omega(alpha, rico, b):
    return (rico * alpha + b) ** 2

def torque(alpha, rico, c):
    return abs(rico * alpha + c)

def sq_torque(alpha, rico, c):
    return (rico * alpha + c) ** 2

w_results = []
w_results_sq = []
T_results = []
T_results_sq = []

for i in range(len(norm_rico)):
    w_result = omega(alpha, norm_rico[i], b[i])
    w_results.append(w_result)
    w_result_sq = sq_omega(alpha, norm_rico[i], b[i])
    w_results_sq.append(w_result_sq)
    T_result = torque(alpha, norm_rico[i], c[i])
    T_results.append(T_result)
    T_result_sq = sq_torque(alpha, norm_rico[i], c[i])
    T_results_sq.append(T_result_sq)

w_results = np.array(w_results).transpose()
w_results_sq = np.array(w_results_sq).transpose()
W_sq_sum = np.sum(w_results_sq, axis=1)

T_results = np.array(T_results).transpose()
T_results_sq = np.array(T_results_sq).transpose()
T_sq_sum = np.sum(T_results_sq, axis=1)

fig, (ax1, ax3) = plt.subplots(ncols=2, figsize=(12, 6))

# Plot results on the primary y-axis
ax1.plot(alpha, w_results, label='Omega')
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Omega')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(alpha, W_sq_sum, label='Squared Omega')
ax2.set_ylabel('Squared Omega')
ax2.legend(loc='upper right')

ax2.scatter(alpha[np.argmin(W_sq_sum)], np.min(W_sq_sum), color='red', zorder=5)

ax3.plot(alpha, T_results, label='Omega')
ax3.set_xlabel('Alpha')
ax3.set_ylabel('Torque')
ax3.legend(loc='upper left')

ax4 = ax3.twinx()
ax4.plot(alpha, T_sq_sum, label='Squared Torque')
ax4.set_ylabel('Squared Torque')
ax4.legend(loc='upper right')

ax4.scatter(alpha[np.argmin(T_sq_sum)], np.min(T_sq_sum), color='red', zorder=5)

print(alpha[np.argmin(W_sq_sum)], alpha[np.argmin(T_sq_sum)])
plt.show()