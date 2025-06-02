import csv
import inspect
import re
import pandas as pd
from scipy.interpolate import interp1d
from scipy.io import loadmat
import numpy as np
import os
from config import *
from helper import *


def time_stiction(dataset, limit=None):
    data = loadmat(dataset)
    omega, time = data['all_w_sol'], data['all_t'].flatten()
    N, t = len(time), time[2]-time[1]
    stiction_time = np.zeros(4)

    if limit is not None:
        omega_max = rpm_to_rad(limit)
    else:
        return None

    # Count occurrences in stiction zone
    for i in range(N):
        for j in range(4):
            if abs(omega[i, j]) < omega_max:
                stiction_time[j] += t
    return stiction_time


def time_stiction_accurate(dataset, limit=None):
    data = loadmat(dataset)
    omega, time = data['all_w_sol'], data['all_t'].flatten()
    N, t = len(time), time[2]-time[1]
    stiction_time = np.zeros(4)

    if limit is not None:
        omega_max = rpm_to_rad(limit)
        omega_min = -omega_max
    else:
        return None

    for j in range(4):
        for i in range(N - 1):
            if abs(omega[i, j]) < omega_max or abs(omega[i + 1, j]) < omega_max:
                if abs(omega[i, j]) < omega_max and abs(omega[i + 1, j]) < omega_max:
                    stiction_time[j] += t
                else:
                    if abs(omega[i, j]) < omega_max:
                        if omega[i+1, j] > omega_max:
                            crossing_value = omega_max
                        else:
                            crossing_value = omega_min
                    else:
                        if omega[i, j] > omega_max:
                            crossing_value = omega_max
                        else:
                            crossing_value = omega_min

                    # ✅ Insert safe interpolation here ↓
                    w0, w1 = omega[i, j], omega[i + 1, j]
                    t0, t1 = time[i], time[i + 1]
                    eps = 1e-12

                    if min(w0, w1) - eps <= crossing_value <= max(w0, w1) + eps:
                        interpolator = interp1d([w0, w1], [t0, t1])
                        crossing_time = interpolator(crossing_value)
                    else:
                        crossing_time = (t0 + t1) / 2  # or skip or log warning

                    # interpolator = interp1d([omega[i, j], omega[i+1, j]], [time[i], time[i + 1]])
                    # crossing_time = interpolator(crossing_value)

                    if abs(omega[i, j]) < omega_max:
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
            omega_sqrd_sum[j] += (omega[i, j] / OMEGA_MIN) ** 2
    return omega_sqrd_sum


def omega_squared_avg(dataset):
    sqrd_sum = omega_squared_sum(dataset)
    data = loadmat(dataset)
    time = data['all_t'].flatten()
    N = len(time)
    sqrd_avg = sqrd_sum / N
    return sqrd_avg


def energy(dataset):
    data = loadmat(dataset)
    omega, torque = data['all_w_sol'], data['all_T_sol']
    time = data['all_t'].flatten()
    t = time[2] - time[1]
    energy_result = np.zeros(4)
    for i in range(len(time)-1):
        for j in range(4):
            pwr = omega[i, j] * torque[i, j] *t
            if pwr > 0:
                energy_result[j] += pwr
    return energy_result


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

def extract_filename_info(filename):
    """
    Extracts the fixed part of the filename and all parameter values (A, B, C, ...).

    Example filename: gaussian_speedXtime_dep_a0.1_b1.4285714285714286e-06_c0.5.mat
    Returns:
        - base_name (constant part): 'gaussian_speedXtime_dep'
        - extracted_values (dict): {'A': 0.1, 'B': 1.42857e-06, 'C': 0.5}
    """
    filename = filename.replace(".mat", "")  # Remove .mat extension

    # Extract everything before "_a" (constant part of the filename)
    base_match = re.match(r"(.+?)_[a-zA-Z]", filename)
    base_name = base_match.group(1) if base_match else filename

    # Extract all parameter-value pairs (e.g., a0.1, b1.42e-06, c0.5)
    pattern = r"_([a-zA-Z])([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
    matches = re.findall(pattern, filename)

    # Convert parameter names to uppercase (A, B, C) and store values as floats
    extracted_values = {match[0].upper(): float(match[1]) for match in matches}

    return base_name, extracted_values


def save_to_excel(functions, directory, output_xlsx, limits=None):
    filenames = None
    all_results = []
    headers = ["Filename", "Base_Name"]
    extracted_data = []

    if limits is None:
        limits = np.array([None])  # Use NumPy array instead of a list

    limits = np.asarray(limits)  # Ensure limits is a NumPy array

    # Get all filenames to extract variable names
    filenames = [f for f in os.listdir(directory) if f.endswith(".mat")]

    if not filenames:
        print("No .mat files found in the directory.")
        return

    # Extract variable names dynamically from the first file
    _, first_extracted_values = extract_filename_info(filenames[0])
    variable_names = sorted(first_extracted_values.keys())  # ['A', 'B', 'C']
    headers.extend(variable_names)  # Add A, B, C as headers

    # Extract values from all filenames
    for filename in filenames:
        base_name, extracted_values = extract_filename_info(filename)
        extracted_data.append([filename, base_name] + [extracted_values.get(var, None) for var in variable_names])

    # Process each function
    for func in functions:
        func_name = func.__name__
        func_params = inspect.signature(func).parameters

        if "limit" in func_params:
            for limit in limits:
                limit_str = f"_{int(limit)}" if limit is not None else ""
                # headers.extend([f"{func_name}{limit_str}_rw{i + 1}" for i in range(4)])
                headers.append(f"{func_name}{limit_str}_sum")

                _, results = repeat_function(lambda f: func(f, limit=int(limit)), directory)
                sum_results = np.sum(results, axis=1, keepdims=True)
                # results_with_sum = np.hstack([results, sum_results])
                all_results.append(sum_results)
        else:
            # headers.extend([f"{func_name}_rw{i + 1}" for i in range(4)])
            headers.append(f"{func_name}_sum")

            _, results = repeat_function(func, directory)
            sum_results = np.sum(results, axis=1, keepdims=True)
            # results_with_sum = np.hstack([results, sum_results])
            all_results.append(sum_results)

    all_results = np.hstack(all_results)  # Combine all results horizontally

    # Create a DataFrame
    df = pd.DataFrame(np.column_stack([extracted_data, all_results]), columns=headers)

    # Save to Excel
    df.to_excel(output_xlsx, index=False, engine='openpyxl')

    print(f"Results saved to {output_xlsx} (Excel format)")


def count_zero_crossings(dataset):
    data = loadmat(dataset)
    omega = data['all_w_sol']  # Shape: (N, 4)
    # Check for sign changes (zero crossings)
    zero_crossings = np.sum(np.diff(np.sign(omega), axis=0) != 0, axis=0)

    return zero_crossings

def extract_time(dataset):
    data = loadmat(dataset)
    solve_time = data['total_time']
    result = np.sum(solve_time, axis=1)
    return result

def extract_iterations(dataset):
    data = loadmat(dataset)
    iterations = data['iter_count']
    result = np.sum(iterations, axis=1)
    return result

def extract_solve_time(dataset):
    data = loadmat(dataset)
    solve_time = data['solve_time']
    result = np.sum(solve_time, axis=1)
    return result

def extract_components(dataset):
    data = loadmat(dataset)
    omega = data['omega_start']
    nullspace_component = NULL_R @ (NULL_R_T @ omega)
    orthogonal_component = omega - nullspace_component
    return float(np.linalg.norm(nullspace_component)), float(np.linalg.norm(orthogonal_component))
    # return nullspace_component, orthogonal_component

# evaluation_functions = [count_zero_crossings, energy, omega_squared_avg, time_stiction_accurate, extract_time, extract_iterations]
# zone = np.array([100])
# save_to_excel(evaluation_functions, 'Data/Auto/slew2_ab', 'Data/Auto/eval2_ab.xlsx', zone)

# print(extract_solve_time('Data/optimisation/guess/500a.mat'))
# print(extract_solve_time('Data/optimisation/guess/500b.mat'))
# print(extract_solve_time('Data/optimisation/guess/500c.mat'))
# print(extract_iterations('Data/optimisation/guess/500a.mat'))
# print(extract_iterations('Data/optimisation/guess/500b.mat'))
# print(extract_iterations('Data/optimisation/guess/500c.mat'))
#
# print(extract_solve_time('Data/optimisation/guess/500a_bis.mat'))
# print(extract_solve_time('Data/optimisation/guess/500b_bis.mat'))
# print(extract_solve_time('Data/optimisation/guess/500c_bis.mat'))
# print(extract_iterations('Data/optimisation/guess/500a_bis.mat'))
# print(extract_iterations('Data/optimisation/guess/500b_bis.mat'))
# print(extract_iterations('Data/optimisation/guess/500c_bis.mat'))

if __name__ == "__main__":


    data = 'Data/conventional/pseudo_omega/slew1/19.mat'
    data = 'Data/OPT/02_zero/slew1/18.mat'
    yeet = loadmat(data)
    # print(np.sum(yeet['iter_count']))
    # print(np.sum(yeet['solve_time']))
    print(count_zero_crossings(data))
    print(omega_squared_avg(data))
    print(energy(data))
    print(time_stiction_accurate(data, limit=100))
    print(extract_components(data))


    print(sum(count_zero_crossings(data)))
    print(sum(omega_squared_avg(data)))
    print(sum(energy(data)))
    print(sum(time_stiction_accurate(data, limit=100)))