import numpy as np
import casadi as ca
from config import *
from helper import *
from scipy.io import loadmat
import matplotlib.pyplot as plt
from rockit import Ocp, MultipleShooting


def alpha_options(part,omega_input, reference=0):
    all_alphas = []
    for j in range(len(part)):
        intervals = np.zeros((4, 2))
        omega_begin = omega_input[:, part[j]]
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

def optimize_variable(x_start, x_end, x_min0, x_max0, x_target, zone_min, zone_max):
    if np.any(x_min0 > x_max0):
        raise ValueError("Infeasible problem detected! Add zero crossings")
    x_min = np.copy(x_min0)
    x_max = np.copy(x_max0)
    N = len(x_min)
    # Fix start and end points
    x_min[0], x_max[0] = x_start, x_start
    x_min[-1], x_max[-1] = x_end, x_end
    bound_limit = 1e3
    x_min[np.isneginf(x_min)] = -bound_limit
    x_max[np.isposinf(x_max)] = bound_limit

    # Define optimization variables
    x = ca.MX.sym('x', N)
    # Objective: Minimize squared error to target
    objective = ca.sumsqr(x - x_target)

    # Constraints
    constraints = []
    for i in range(N):
        constraints.append(x[i] - x_min[i])  # x >= x_min
        constraints.append(x_max[i] - x[i])  # x <= x_max

    # Define NLP problem
    nlp = {'x': x, 'f': objective, 'g': ca.vertcat(*constraints)}

    # Solver settings
    opts = {"ipopt.print_level": 0, "print_time": 0}
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Solve with strict bounds
    solution = solver(lbg=-np.inf, ubg=0)  # Allow inequality constraints

    # Extract solution
    x_opt = solution['x'].full().flatten()

    return x_opt

def rockit_optimize(x_start, x_end, x_min0, x_max0, x_target, zone_min, zone_max):
    if np.any(x_min0 > x_max0):
        raise ValueError("Infeasible problem detected! Add zero crossings")
    x_min = np.copy(x_min0)
    x_max = np.copy(x_max0)
    N = len(x_min)
    # Fix start and end points
    x_min[0], x_max[0] = x_start, x_start
    x_min[-1], x_max[-1] = x_end, x_end
    bound_limit = 1e3
    x_min[np.isneginf(x_min)] = -bound_limit
    x_max[np.isposinf(x_max)] = bound_limit

    ocp = Ocp(t0=0, T=N/10)
    alpha_opt = ocp.control()
    max_band = ocp.parameter(1, grid='control')
    min_band = ocp.parameter(1, grid='control')
    ocp.set_value(max_band, x_max)
    ocp.set_value(min_band, x_min)
    ocp.subject_to(min_band <= (alpha_opt <= max_band))
    max_diff = 5
    ocp.subject_to(ocp.next(alpha_opt) - alpha_opt <= max_diff)
    ocp.subject_to(ocp.next(alpha_opt) - alpha_opt >= -max_diff)

    # **Soft Constraint for Band Crossing**
    # Define a penalty term that activates when alpha_opt crosses forbidden regions
    crossing_penalty = 1e9
    crossing_cost = 0
    a = 10
    for k in range(len(zone_min)):
        band_reference = ocp.parameter(1, grid='control')
        band_top = zone_max[k]
        band_bottom = zone_min[k]
        reference = band_top - band_bottom
        ocp.set_value(band_reference, reference)

        crossing_cost += np.exp(-a * (alpha_opt - band_reference) ** 2)
        ocp.add_objective(crossing_penalty*ocp.integral(crossing_cost))


    time_objective = ocp.integral((alpha_opt - x_target) ** 2)
    ocp.add_objective(time_objective)

    print("crossing_cost:", crossing_cost)

    ocp.solver('ipopt', SOLVER_OPTS)
    ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
    sol = ocp.solve()

    _, alpha_sol = sol.sample(alpha_opt, grid='control')  # (N,)

    return alpha_sol[:-1]

def plot(n0=0, n=8005, path=None, scatter=True, limits=True, optimal=False, background=False):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # ax.plot(time, segments.T, color='gray')

    ax.plot(time, alpha, color='black', linestyle='--', label='Ideal path')
    if scatter:
        for i in range(4):
            ax.scatter(np.ones_like(begin_alphas[i]) * falling[i] / 10, begin_alphas[i], color='b', zorder=5)
            ax.scatter(np.ones_like(end_alphas[i]) * rising[i] / 10, end_alphas[i], color='r', zorder=5)
    if background:
        ax.axvspan(0, 200, facecolor='lightgrey', alpha=0.3, zorder=0)
        ax.axvspan(400, 600, facecolor='lightgrey', alpha=0.3, zorder=0)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0, 8, 2):
        color = colors[i // 2 % len(colors)]  # Cycle through colors
        avg = (segments[i] + segments[i + 1]) / 2
        plt.plot(time, avg, color=color)
        plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    if limits:
        plt.plot(time[n0:n0+n], max_constraint, color='black', linestyle='--', label='Max Constraint')
        plt.plot(time[n0:n0+n], min_constraint, color='black', label='Min Constraint')
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
sections = []
if len(rising) == len(falling):
    for i in range(len(rising)):
        sections.append(data[:, falling[i]:rising[i]])
else:
    print("Error: rising and falling edges do not match")
    print(rising, falling)
    exit()
momentum4_with_nullspace = pseudo_sol(data)
momentum3 = R @ momentum4_with_nullspace
alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:,0:1])
momentum4 = R_PSEUDO @ momentum3

time = np.linspace(0, 800, 8005)
# alpha = nullspace_alpha(momentum4)
alpha, alpha_ref = np.zeros_like(time, dtype=int), 0

begin_alphas = alpha_options(falling, momentum4, reference=alpha_ref)
end_alphas = alpha_options(rising, momentum4, reference=alpha_ref)

begin_signs = alpha_to_sign(begin_alphas, momentum4, falling)
end_signs = alpha_to_sign(end_alphas, momentum4, rising)
changes = count_sign_changes(begin_signs, end_signs)

length = momentum4.shape[1]
segments = np.zeros((8, length )) # Constraints: 4 bands
for j in range(length):
    omega = momentum4[:, j]
    for i in range(4):
        segments[2*i, j] = (OMEGA_MIN - omega[i]) * (-1) ** i
        segments[1+2*i, j] = (-OMEGA_MIN - omega[i]) * (-1) ** i

# segments = segments - alpha[0]
# begin_alphas = np.array(begin_alphas) - alpha[0]
# end_alphas = np.array(end_alphas) - alpha[0]
# alpha = alpha - alpha[0]

for k in range(len(begin_alphas)):
    begin = falling[k]
    size = rising[k] - falling[k]
    for i in range(len(begin_alphas[k])):
        for j in range(len(end_alphas[k])):
            constrained_segments = segments[:, begin:begin+size]
            min_constraint, max_constraint, crossing_min, crossing_max = compute_constraints(begin_signs[k][i], end_signs[k][j], constrained_segments, position=begin_alphas[k][i])
            # x_opt = rockit_optimize(begin_alphas[k][i], end_alphas[k][j], min_constraint, max_constraint, alpha_ref, crossing_min, crossing_max)
            plot(begin, size, limits=False, scatter=True, background=True)
