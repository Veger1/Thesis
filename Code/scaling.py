import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from init_helper import load_data

torque_profile = load_data()  # This is the full test data (8004 samples)

time = np.linspace(0, 800, 8004)  # Example time array from 0 to 100
def low_pass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y
cutoff_frequency = 1  # Cutoff frequency in Hz
sampling_frequency = 10
filtered_torque_profile = low_pass_filter(torque_profile, cutoff_frequency, sampling_frequency)

# Calculate the first derivative (velocity) using finite difference method
delta_t = np.diff(time)  # Time intervals (should be constant)
torque_derivative = np.diff(filtered_torque_profile) / delta_t  # First derivative
filtered_torque_derivative = low_pass_filter(torque_derivative, cutoff_frequency, sampling_frequency)

# Since np.diff reduces the size of the array by 1, we need to adjust the time array as well
time_derivative = time[1:]  # Remove the first time value to match the size of the derivative

# Calculate the second derivative (acceleration) using finite difference method
delta_t_derivative = np.diff(time_derivative)  # Time intervals for the derivative
torque_second_derivative = np.diff(filtered_torque_derivative) / delta_t_derivative  # Second derivative

# Plot the results
plt.figure(figsize=(10, 6))

# Original torque profile
plt.subplot(3, 1, 1)
plt.plot(time, torque_profile.transpose(), label='Torque Profile', color='b')
plt.title('Torque Profile')
plt.xlabel('Time (s)')
plt.ylabel('Torque')
plt.grid(True)

# First derivative (torque rate of change)
plt.subplot(3, 1, 2)
plt.plot(time_derivative, torque_derivative.transpose(), label='First Derivative', color='g')
plt.title('First Derivative of Torque')
plt.xlabel('Time (s)')
plt.ylabel('dTorque/dt')
plt.yscale('log')
plt.grid(True)

# Second derivative (torque acceleration)
plt.subplot(3, 1, 3)
plt.plot(time_derivative[1:], torque_second_derivative.transpose(), label='Second Derivative', color='r')
plt.title('Second Derivative of Torque')
plt.xlabel('Time (s)')
plt.ylabel('d^2Torque/dt^2')
plt.yscale('log')
plt.grid(True)

plt.tight_layout()
plt.show()

# Output some values for verification
print(f"First derivative (velocity) at t=0: {torque_derivative[0]:.4e}")
print(f"Second derivative (acceleration) at t=0: {torque_second_derivative[0]:.4e}")
