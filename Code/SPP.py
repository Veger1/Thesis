import numpy as np
import casadi as ca
from config import *
from helper import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from rockit import Ocp, MultipleShooting

def calc_segments(omega_input):
    length = omega_input.shape[1]
    segments = np.zeros((8, length))  # Constraints: 4 bands
    for j in range(length):
        omega = momentum4[:, j]
        for i in range(4):
            segments[2 * i, j] = (OMEGA_MIN - omega[i]) * (-1) ** i
            segments[1 + 2 * i, j] = (-OMEGA_MIN - omega[i]) * (-1) ** i
    return segments

def alpha_options(omega_input, reference=0):
    all_alphas = []
    for j in range(omega_input.shape[1]):
        intervals = np.zeros((4, 2))
        omega_begin = omega_input[:, j]
        for i in range(4):
            first = (OMEGA_MIN - omega_begin[i]) * (-1) ** i
            second = (-OMEGA_MIN - omega_begin[i]) * (-1) ** i
            intervals[i] = np.sort([first, second]).flatten()  # Potentially sort manually
        sorted_intervals = intervals[np.argsort(intervals[:, 0])]
        merged_intervals = []
        for interval in sorted_intervals:
            if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
        alphas = best_alpha(merged_intervals, reference)
        all_alphas.append(alphas)
    return all_alphas

def alpha_to_sign(all_alphas, omega_input, indexes):
    all_result = []
    for i in range(len(all_alphas)):
        result = []
        for j in range(len(all_alphas[i])):
            out = omega_input[:, indexes[i]] - (all_alphas[i][j] * NULL_R).flatten()
            result.append(np.sign(np.array(out)))
        all_result.append(result)
    return all_result

def count_sign_changes(list1, list2):
    results = []
    # TO ADD: if sign does not change, check if it does not cross a band of wheel that also does not change sign
    # Iterate through each corresponding level-2 list
    for sublist1, sublist2 in zip(list1, list2):
        sublist_results = np.zeros((len(sublist1), len(sublist2)), dtype=int)

        for i, arr1 in enumerate(sublist1):
            sign1 = np.sign(arr1)  # Convert to sign representation

            for j, arr2 in enumerate(sublist2):
                sign2 = np.sign(arr2)  # Convert to sign representation

                # Count sign changes by comparing element-wise
                sign_changes = np.sum(sign1 != sign2)
                sublist_results[i, j] = sign_changes

        results.append(sublist_results)

    return results

def compute_constraints(wheel_signs_before, wheel_signs_after, bands, position=0):
    N = bands.shape[1]
    min_constraints = np.full(N, -np.inf)
    max_constraints = np.full(N, np.inf)
    crossings_max = []
    crossings_min = []
    for i in range(len(wheel_signs_before)):
        band_idx = 2 * i  # Row index corresponding to the wheel's bands
        band_max = np.max(bands[band_idx:band_idx + 2], axis=0)
        band_min = np.min(bands[band_idx:band_idx + 2], axis=0)
        if wheel_signs_before[i] == wheel_signs_after[i]:  # Sign change detected
            if position < band_max[0]:
                max_constraints = np.minimum(max_constraints, band_min)

            if position > band_min[0]:
                min_constraints = np.maximum(min_constraints, band_max)
        else:
            crossings_max.append(band_max)
            crossings_min.append(band_min)


    return min_constraints, max_constraints, crossings_min, crossings_max

def find_crossings(signals):
    num_signals, N = signals.shape
    crossings = {}

    for i in range(num_signals):
        for j in range(i+1, num_signals):
            diff = signals[i] - signals[j]
            crossing_indices = np.where(np.diff(np.sign(diff)) != 0)[0]
            crossings[(i, j)] = crossing_indices

    return crossings

def calc_mid_indices(possible_alphas):
    segment_counts = np.array([len(a) for a in possible_alphas])
    change_indices = np.where(np.diff(segment_counts) != 0)[0] + 1
    mid_indices = [(change_indices[i] + change_indices[i+1]) // 2 for i in range(len(change_indices) - 1)]
    return mid_indices


def plot(n0=0, n=8005, path=None, scatter=True, limits=True, optimal=False):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # ax.plot(time, segments.T, color='gray')
    ax.plot(time, alpha, color='black', linestyle='--', label='Ideal path')
    if scatter:
        for i in range(len(alphas)):
            ax.scatter( np.ones_like(alphas[i])*time[i], alphas[i], color='b', zorder=5)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0, 8, 2):
        color = colors[i // 2 % len(colors)]  # Cycle through colors
        avg = (segments[i] + segments[i + 1]) / 2
        plt.plot(time, avg, color=color)
        plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    if path is not None:
        plt.plot(time[n0:n0+n], path, color='red', label='Optimal Path')
    plt.xlabel("Time (s)")
    plt.ylabel("Nullspace component")
    plt.title("Zero speed bands vs time")
    # plt.legend()
    plt.show()

data = load_data('Data/Slew1.mat')
low_torque_flag = hysteresis_filter(data, 0.000005, 0.000015)
low_torque_flag[0:2] = False
rising, falling = detect_transitions(low_torque_flag)
falling = np.insert(falling, 0, 0)

momentum4 = pseudo_sol(data)
momentum3 = R @ momentum4
momentum0 = R_PSEUDO @ momentum3
sections = []
if len(rising) == len(falling):
    for i in range(len(rising)):
        sections.append(momentum4[:, falling[i]:rising[i]])
else:
    print("Error: rising and falling edges do not match")
    print(rising, falling)
    exit()

time = np.linspace(0, 800, 8005)
alpha = nullspace_alpha(momentum4)

segments = calc_segments(momentum4)
all_segments = []
for i in range(len(sections)):
    all_segments.append(segments[:, falling[i]:rising[i]])

all_options = alpha_options(all_segments[0],alpha[0])
options_count = np.array([len(a) for a in all_options])
change_indices = np.where(np.diff(options_count) != 0)[0] + 1
mid_indices = [(change_indices[i] + change_indices[i+1]) // 2 for i in range(len(change_indices) - 1)]
full_indices = [falling[0]] + mid_indices + [rising[0]]
plot(scatter=False)








