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
    """
    Convert the objective expression to a CasADi-compatible function.
    """
    ocp_t = ocp.t
    w_sym, t_sym = symbols('w t')
    objective_expr_casadi = lambdify((w_sym, t_sym), objective_expr, 'numpy')
    objective_expr_casadi = objective_expr_casadi(w, ocp_t)
    return objective_expr_casadi

def hysteresis_filter(signal, low_threshold, high_threshold):
    """
    Apply hysteresis filtering to the signal to detect low torque intervals.
    :param signal: numpy array (M, N)
    :param low_threshold: (float)
    :param high_threshold: (float)
    :return: boolean array (N) indicating low torque intervals
    """
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
    """
    Detect rising and falling edges in a boolean signal.
    :param signal: numpy array (N,) of boolean values
    :return:
        - rising_edges: List of indices of rising edges
        - falling_edges: List of indices of falling edges
    """
    signal = np.array(signal, dtype=bool)  # Ensure it's a boolean NumPy array
    diff_signal = np.diff(signal.astype(int))  # Convert to int and compute difference

    rising_edges = np.where(diff_signal == 1)[0] + 1  # +1 to shift index correctly
    falling_edges = np.where(diff_signal == -1)[0] + 1

    return rising_edges, falling_edges

def pseudo_sol(data, omega_start):
    """
    Calculate the pseudoinverse solution for the given data and initial omega vector.
    :param data: numpy array of shape (3, N) representing the torque data
    :param omega_start: numpy array of shape (4, 1) representing the initial omega vector
    :return: omega_sol: numpy array of shape (4, N+1) with the solution
    """
    length = data.shape[1]
    omega_sol = np.zeros((4, length + 1))
    omega_sol[:, 0] = omega_start.flatten()
    for i in range(length):
        omega_sol[:, i + 1] = (omega_sol[:, i].reshape((4,1)) + 0.1 * R_PSEUDO @ data[:, i].reshape(3, 1) / IRW).flatten()
    return omega_sol

def nullspace_alpha(data):
    length = data.shape[1]
    alpha = np.zeros(length)
    for i in range(length):
        alpha[i] = (- data[0, i] + data[1, i] - data[2, i] + data[3, i]) / 4
    return alpha

def best_alpha(merged_intervals, alpha=0, use_bounds=False):
    """
    Invert the merged intervals and find the best alpha value within each inverted interval.
    :param merged_intervals: List of merged intervals, e.g. [[-3, 0], [1, 4]]
    :param alpha: Float, the best value for alpha
    :param use_bounds: Boolean, to include saturation or not
    :return: result: List of best alpha values
    """
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
    """
    Compute the intersections of bands in a stacked array of bands.
    :param stacked_bands: numpy array of shape (8, N) where each 2 rows represents upper and lower bound of stiction band
    :return:
        - overlaps_with_crossing: dict of overlapping bands with crossing
        - overlaps_without_crossing: dict of overlapping bands without crossing
    """
    assert stacked_bands.shape[0] == 8
    N = stacked_bands.shape[1]
    num_bands = 4
    sorted_bands = stacked_bands.reshape((num_bands, 2, N))  # shape (4, 2, N)
    overlaps_with_crossing = {}
    overlaps_without_crossing = {}
    for i, j in combinations(range(num_bands), 2):
        # Pairwise comparison between bands i and j, for a total of 6 comparisons
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
    """
    Invert the intervals in the overlap_intervals dictionary.
    :param overlap_intervals: Dict with keys as tuples of band indices and values as lists of overlapping intervals.
    :param total_start_idx: 0
    :param total_end_idx: 8004
    :return: inverted_dict: Dict with keys as band indices and values as lists of inverted intervals.
    """
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
    """
    Select the minimum set of nodes that cover all intervals in the inverted_intervals_dict.
    :param inverted_intervals_dict: dictionary where keys are band indices and values are lists of inverted intervals.
    :param total_range: 0
    :param initial_nodes: 8004
    :param initial_nodes: List of initial nodes to start the coverage from.
    :return: selected_nodes: List of selected nodes that cover all intervals.
    """
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
    """
    Generate a random start for the omega vector.
    :param seed: integer, optional
    :return:
        - omega_start: numpy array of shape (4, 1) with random values
        - seed: integer, the seed used for random generation
    """
    if seed is None:
        seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    omega_start = np.random.uniform(-300, 300, (4, 1))
    return omega_start, seed

def calc_alpha_torque_limits(torque_data, n0, N):
    """
    Calculate reduced set of alpha constraints based on torque data.
    :param torque_data: numpy array of shape (3, 8004) of torque data
    :param n0: beginning index for the torque data slice
    :param N: length of the torque data slice
    :return:
        - alpha_lower_bound: numpy array of shape (N,) with lower bounds for alpha
        - alpha_upper_bound: numpy array of shape (N,) with upper bounds for alpha
    """
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