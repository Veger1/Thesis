from scipy.interpolate import interp1d
from scipy.io import loadmat
import numpy as np
import os

# from scipy.special import result

from helper import Helper
helper = Helper()

# Define Omega thresholds
Omega_max = helper.rpm_to_rad(300)
Omega_min = -helper.rpm_to_rad(300)
Omega_tgt = helper.rpm_to_rad(1000)

def time_stiction(dataset):
    data = loadmat(dataset)
    omega, time = data['all_w_sol'], data['all_t'].flatten()
    N, t = len(time), time[2]-time[1]
    stiction_time = np.zeros(4)

    # Count occurrences in stiction zone
    for i in range(N):
        for j in range(4):
            if abs(omega[i, j]) < Omega_max:
                stiction_time[j] += t
    return stiction_time

def time_stiction_accurate(dataset):
    data = loadmat(dataset)
    omega, time = data['all_w_sol'], data['all_t'].flatten()
    N, t = len(time), time[2]-time[1]
    stiction_time = np.zeros(4)
    # What if both points are outside stiction zone but the line crosses the zone?
    for j in range(4):
        for i in range(N - 1):
            if abs(omega[i, j]) < Omega_max or abs(omega[i + 1, j]) < Omega_max:
                if abs(omega[i, j]) < Omega_max and abs(omega[i + 1, j]) < Omega_max:
                    stiction_time[j] += t
                else:
                    if abs(omega[i, j]) < Omega_max:
                        if omega[i+1, j] > Omega_max:
                            crossing_value = Omega_max
                        else:
                            crossing_value = Omega_min
                    else:
                        if omega[i, j] > Omega_max:
                            crossing_value = Omega_max
                        else:
                            crossing_value = Omega_min
                    interpolator = interp1d([omega[i, j], omega[i+1, j]], [time[i], time[i + 1]])
                    crossing_time = interpolator(crossing_value)

                    if abs(omega[i, j]) < Omega_max:
                        # omega[i] is inside, add time from time[i] to crossing
                        stiction_time[j] += crossing_time - time[i]
                    else:
                        # omega[i+1] is inside, add time from crossing to time[i+1]
                        stiction_time[j] += time[i+1] - crossing_time
    return stiction_time

def highest_omega(dataset):
    data = loadmat(dataset)
    omega, time = data['all_w_sol'], data['all_t'].flatten()
    N = len(time)
    highest_value = np.zeros(4)

    for i in range(N):
        for j in range(4):
            if abs(omega[i, j]) > highest_value[j]:
                highest_value[j] = omega[i, j]
    return highest_value

def time_stiction_percentage(dataset):
    stiction_time = time_stiction(dataset)
    total_time = loadmat(dataset)['all_t'].flatten()[-1]
    return stiction_time / total_time

def time_stiction_accurate_percentage(dataset):
    stiction_time = time_stiction_accurate(dataset)
    total_time = loadmat(dataset)['all_t'].flatten()[-1]
    return stiction_time / total_time

def omega_squared_sum(dataset):  # Score of 1 signifies average vibration level is that of Omega_max
    data = loadmat(dataset)
    omega, time = data['all_w_sol'], data['all_t'].flatten()
    N = len(time)
    omega_sqrd_sum = np.zeros(4)
    for i in range(N):
        for j in range(4):
            omega_sqrd_sum[j] += (omega[i, j]/Omega_max) ** 2
    return omega_sqrd_sum

def omega_squared_avg(dataset):
    sqrd_sum = omega_squared_sum(dataset)
    data = loadmat(dataset)
    time = data['all_t'].flatten()
    N = len(time)
    sqrd_avg = sqrd_sum / N
    return sqrd_avg

def repeat_function(func, directory):
    filenames = []
    results = []
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            filepath = os.path.join(directory, filename)
            result = func(filepath)
            filenames.append(filename)
            results.append(result)
    filenames_array = np.array(filenames)
    results_array = np.array(results)
    return filenames_array, results_array


filenames, results = repeat_function(time_stiction, 'Data/100s')
print(filenames),print(results)


