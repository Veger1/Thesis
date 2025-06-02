import numpy as np
from sympy import symbols, lambdify
from config import R_PSEUDO, IRW, R, MAX_TORQUE
from itertools import combinations
from collections import defaultdict
from scipy.io import savemat
import time as clock
import inspect

def rpm_to_rad(rpm):
    return rpm * 2 * np.pi / 60

def rad_to_rpm(rad):
    return rad * 60 / (2 * np.pi)

def convert_expr(ocp, objective_expr, w):
    ocp_t = ocp.t
    w_sym, t_sym = symbols('w t')
    objective_expr_casadi = lambdify((w_sym, t_sym), objective_expr, 'numpy')
    objective_expr_casadi = objective_expr_casadi(w, ocp_t)
    return objective_expr_casadi

def exponential_moving_average(signal, alpha=0.1):
    """Apply exponential moving average to smooth the signal."""
    smoothed = np.zeros_like(signal)
    smoothed[:, 0] = signal[:, 0]  # Initialize with first value
    for i in range(1, signal.shape[1]):
        smoothed[:, i] = alpha * signal[:, i] + (1 - alpha) * smoothed[:, i-1]
    return smoothed

def hysteresis_filter(signal, low_threshold, high_threshold):
    """Apply hysteresis to prevent rapid switching."""
    flag = np.zeros(signal.shape[1], dtype=bool)
    active = False  # Start with no low torque detection

    for i in range(signal.shape[1]):
        if np.all(abs(signal[:, i]) < low_threshold):
            active = True
        elif np.all(abs(signal[:, i]) > high_threshold):
            active = False
        flag[i] = active
    return flag

def detect_transitions(signal):
    signal = np.array(signal, dtype=bool)  # Ensure it's a boolean NumPy array
    diff_signal = np.diff(signal.astype(int))  # Convert to int and compute difference

    rising_edges = np.where(diff_signal == 1)[0] + 1  # +1 to shift index correctly
    falling_edges = np.where(diff_signal == -1)[0] + 1

    return rising_edges, falling_edges

def pseudo_sol(data, omega_start):
    length = data.shape[1]
    omega_sol = np.zeros((4, length + 1))
    omega_sol[:, 0] = omega_start.flatten()
    for i in range(length):
        omega_sol[:, i + 1] = (omega_sol[:, i].reshape((4,1)) + 0.1 * R_PSEUDO @ data[:, i].reshape(3, 1) / IRW).flatten()
    return omega_sol

def total_momentum(data):
    omega_sol = pseudo_sol(data)
    momentum = R @ omega_sol
    return momentum

def nullspace_alpha(data):
    length = data.shape[1]
    alpha = np.zeros(length)
    for i in range(length):
        alpha[i] = (- data[0, i] + data[1, i] - data[2, i] + data[3, i]) / 4
    return alpha

def best_alpha(merged_intervals, alpha=0, use_bounds=False):
    inverted_intervals = []
    for i, interval in enumerate(merged_intervals):
        if i == 0 and not use_bounds:
            inverted_intervals.append([-np.inf, interval[0]])  # [-inf, -3]

        if i == len(merged_intervals) - 1 and not use_bounds:
            inverted_intervals.append([interval[1], np.inf])  # [4, inf]

        if i < len(merged_intervals) - 1:
            inverted_intervals.append([interval[1], merged_intervals[i + 1][0]])

    # Step 2: Find the smallest absolute value in each inverted interval
    result = []
    for interval in inverted_intervals:
        if interval[0] == -np.inf:
            result.append(interval[1])  # First interval boundary
        elif interval[1] == np.inf:
            result.append(interval[0])  # Last interval boundary
        else:
            if interval[0] < alpha < interval[1]:
                result.append(alpha)
            else:
                dist_to_0 = abs(interval[0] - alpha)
                dist_to_1 = abs(interval[1] - alpha)

                if dist_to_0 < dist_to_1:
                    result.append(interval[0])
                else:
                    result.append(interval[1])
    return result

def compute_band_intersections(stacked_bands):
    assert stacked_bands.shape[0] == 8
    N = stacked_bands.shape[1]
    num_bands = 4
    sorted_bands = stacked_bands.reshape((num_bands, 2, N))  # shape (4, 2, N)
    overlaps_with_crossing = {}
    overlaps_without_crossing = {}
    for i, j in combinations(range(num_bands), 2):
        lower_i, upper_i = sorted_bands[i, 0], sorted_bands[i, 1]
        lower_j, upper_j = sorted_bands[j, 0], sorted_bands[j, 1]

        lower = np.maximum(lower_i, lower_j)
        upper = np.minimum(upper_i, upper_j)
        is_overlap = lower < upper

        intervals = []
        if np.any(is_overlap):
            starts = np.where(np.diff(np.concatenate(([0], is_overlap.astype(int)))) == 1)[0]
            ends = np.where(np.diff(np.concatenate((is_overlap.astype(int), [0]))) == -1)[0]
            for s, e in zip(starts, ends):
                pre = max(0, s - 1)
                post = min(N - 1, e)

                mid_i_pre = 0.5 * (sorted_bands[i, 0, pre] + sorted_bands[i, 1, pre])
                mid_j_pre = 0.5 * (sorted_bands[j, 0, pre] + sorted_bands[j, 1, pre])

                mid_i_post = 0.5 * (sorted_bands[i, 0, post] + sorted_bands[i, 1, post])
                mid_j_post = 0.5 * (sorted_bands[j, 0, post] + sorted_bands[j, 1, post])

                sign_pre = np.sign(mid_i_pre - mid_j_pre)
                sign_post = np.sign(mid_i_post - mid_j_post)

                # Check if it's crossing or just overlap
                if sign_pre == sign_post and sign_pre != 0:  # No crossing, just overlap
                    overlaps_without_crossing.setdefault((i, j), []).append((s, e))
                elif sign_pre != sign_post:  # Crossing detected
                    overlaps_with_crossing.setdefault((i, j), []).append((s, e))

        if (i, j) not in overlaps_with_crossing:
            overlaps_with_crossing[(i, j)] = []
        if (i, j) not in overlaps_without_crossing:
            overlaps_without_crossing[(i, j)] = []
    return overlaps_with_crossing, overlaps_without_crossing

def invert_intervals(overlap_intervals, total_start_idx, total_end_idx):
    inverted_dict = {}

    for key, intervals in overlap_intervals.items():
        if not intervals:
            inverted_dict[key] = [(total_start_idx, total_end_idx)]
            continue

        # Sort and merge
        intervals = sorted(intervals)
        merged = [intervals[0]]
        for current in intervals[1:]:
            prev = merged[-1]
            if current[0] <= prev[1]:  # overlapping or adjacent
                merged[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged.append(current)

        # Invert merged intervals
        inverted = []
        prev_end = total_start_idx
        for start, end in merged:
            if prev_end < start:
                inverted.append((prev_end, start))
            prev_end = end
        if prev_end < total_end_idx:
            inverted.append((prev_end, total_end_idx))

        inverted_dict[key] = inverted

    return inverted_dict

def select_minimum_covering_nodes(inverted_intervals_dict, total_range, initial_nodes=None):
    interval_list = []
    idx_to_key = {}
    idx = 0
    for key, intervals in inverted_intervals_dict.items():
        for interval in intervals:
            interval_list.append(interval)
            idx_to_key[idx] = (key, interval)
            idx += 1

    interval_list = [(a, b) for (a, b) in interval_list if b > a + 1]

    initial_nodes = set(initial_nodes) if initial_nodes else set()
    uncovered_intervals = set(range(len(interval_list)))
    for node in initial_nodes:
        for i, (start, end) in enumerate(interval_list):
            if i in uncovered_intervals and start <= node < end:
                uncovered_intervals.remove(i)

    start, end = total_range
    coverage = defaultdict(set)
    for t in range(start, end):
        for i in uncovered_intervals:
            interval = interval_list[i]
            if interval[0] < t < interval[1]:
                coverage[t].add(i)

    selected_nodes = set(initial_nodes)

    while uncovered_intervals:
        valid_candidates = {t: ids for t, ids in coverage.items() if ids & uncovered_intervals}
        if not valid_candidates:
            print("Remaining uncovered intervals:")
            for idx in uncovered_intervals:
                print(f"Interval: {interval_list[idx]}")
            raise ValueError("No valid candidates to cover remaining intervals.")
        best_t = max(valid_candidates.items(), key=lambda x: len(x[1] & uncovered_intervals))[0]
        selected_nodes.add(best_t)
        for i in coverage[best_t]:
            uncovered_intervals.discard(i)

    return sorted(selected_nodes)

def get_random_start(seed=None):
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    omega_start = np.random.uniform(-300, 300, (4, 1))
    return omega_start, seed

def save_classes_to_mat(class_list, filename="classes_output.mat"):
    """
    Save a list of class instances to a .mat file.
    Each instance becomes a struct (folder), and its attributes are fields.

    Parameters:
    - class_list: list of class instances
    - filename: output .mat filename
    """
    mat_data = {}

    for i, cls in enumerate(class_list):
        class_name = getattr(cls, 'name', f'class_{i}')
        if class_name in mat_data:
            class_name += f'_{i}'

        attributes = {}
        for attr_name in dir(cls):
            if attr_name.startswith("_"):
                continue  # skip private/internal
            attr_value = getattr(cls, attr_name)
            if inspect.ismethod(attr_value) or inspect.isfunction(attr_value):
                continue  # skip methods

            try:
                if attr_value is None:
                    attributes[attr_name] = np.nan  # or use: [] or 'None'
                else:
                    attributes[attr_name] = attr_value
            except Exception as e:
                print(f"Skipping {attr_name} of {class_name}: {e}")

        mat_data[class_name] = attributes

        savemat(filename, mat_data)
    print(f"Saved {len(class_list)} class instances to {filename}")


def calc_alpha_torque_limits(torque_data, n0, N):
    T_fixed = R_PSEUDO @ torque_data[:, n0:n0 + N]
    T_fixed[[1, 3], :] *= -1
    T_fixed_pos = T_fixed + MAX_TORQUE
    T_fixed_neg = T_fixed - MAX_TORQUE
    mask_max_neg = T_fixed_neg == np.max(T_fixed_neg, axis=0, keepdims=True)
    mask_min_pos = T_fixed_pos == np.min(T_fixed_pos, axis=0, keepdims=True)
    T_critical_neg = np.where(mask_max_neg, T_fixed, 0)
    T_critical_pos = np.where(mask_min_pos, T_fixed, 0)
    # Collapse each column to one row: the critical values
    T_critical_neg = np.sum(T_critical_neg, axis=0)  # shape: (N,)
    T_critical_pos = np.sum(T_critical_pos, axis=0)  # shape: (N,)
    alpha_upper_bound = MAX_TORQUE - T_critical_neg  # shape: (N,)
    alpha_lower_bound = -MAX_TORQUE - T_critical_pos  # shape: (N,)
    return alpha_lower_bound, alpha_upper_bound