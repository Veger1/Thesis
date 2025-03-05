import csv
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat
import numpy as np
import os
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


def power(dataset):
    data = loadmat(dataset)
    omega, torque = data['all_w_sol'], data['all_T_sol']
    time = data['all_t'].flatten()
    t = time[2] - time[1]
    power = np.zeros(4)
    for i in range(len(time)):
        for j in range(4):
            pwr = omega[i, j] * torque[i, j]
            if pwr > 0:
                power[j] += pwr
    return power


def sum_elements(function):
    def wrapper(*args, **kwargs):
        result = function(*args, **kwargs)
        return np.sum(result)
    return wrapper


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


def save_results_to_csv(functions, directory, output_csv):
    filenames = None
    all_results = []
    headers = ["Filename"]

    for func in functions:
        func_name = func.__name__
        headers.extend([f"{func_name}_rw{i + 1}" for i in range(4)])
        filenames, results = repeat_function(func, directory)
        all_results.append(results)

    all_results = np.hstack(all_results)  # Combine all results horizontally

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)  # Write headers
        for i, filename in enumerate(filenames):
            writer.writerow([filename] + all_results[i].tolist())

    print(f"Results saved to {output_csv}")


def save_results_to_excel(functions, directory, output_xlsx):
    filenames = None
    all_results = []
    headers = ["Filename"]

    # Process each function
    for func in functions:
        func_name = func.__name__
        headers.extend([f"{func_name}_rw{i + 1}" for i in range(4)])
        headers.append(f"{func_name}_sum")  # Sum column
        filenames, results = repeat_function(func, directory)

        # Compute sum of 4 columns per function
        sum_results = np.sum(results, axis=1, keepdims=True)  # Sum along row

        # Stack results with sum column
        results_with_sum = np.hstack([results, sum_results])
        all_results.append(results_with_sum)

    all_results = np.hstack(all_results)  # Combine all results horizontally

    # Create a DataFrame for Excel output
    df = pd.DataFrame(np.column_stack([filenames, all_results]), columns=headers)

    # Save to Excel
    df.to_excel(output_xlsx, index=False, engine='openpyxl')

    print(f"Results saved to {output_xlsx} (Excel format)")


# filenames, results = repeat_function(omega_squared_avg, 'Data/Auto/gauss_speedXtime')
# print(filenames), print(results)
evaluation_functions = [power, omega_squared_avg, time_stiction, time_stiction_accurate]
save_results_to_excel(evaluation_functions, 'Data/Auto/gauss_speedXtime_2', 'Data/Auto/gauss_speedXtime_2/results.xlsx')

# print(sum(time_stiction('Data/Auto/gauss_speedXtime/gaussian_speedXtime_dep_a0.1_b1e-05_c1.mat')))
# print(sum(omega_squared_avg('Data/Auto/gauss_speedXtime/gaussian_speedXtime_dep_a0.1_b1e-05_c1.mat')))
