import os

import matplotlib.pyplot as plt
from scipy.io import savemat
from sympy import symbols, exp, Abs, tanh, atan, pi
from config import *
from repeat_solver import solve_ocp_index
from helper import *


def minmax_torque(torque_sc, omega):
    T_rw = R_PSEUDO @ torque_sc  # + Null_R @ alpha
    alpha_null = -T_rw / NULL_R
    alpha_best = (max(alpha_null) + min(alpha_null)) / 2
    return alpha_best

def minmax_omega(torque_sc, omega):
    nominator = - omega - 0.1*I_INV @ R_PSEUDO @ torque_sc
    denominator = 0.1*I_INV @ NULL_R
    alpha_null = nominator / denominator
    alpha_best = (max(alpha_null) + min(alpha_null)) / 2
    return alpha_best

def squared_omega(torque_sc, omega):
    omega_new = omega + 0.1*I_INV @ R_PSEUDO @ torque_sc
    alpha =  constrained_alpha(omega_new)
    return -alpha/(0.1*41802.61224523921)  # replace with IRW

def pseudo(torque_sc, omega):
    return np.zeros((1, 1))

def optimal_alpha(omega):
    opt_alpha  = (- omega[0] + omega[1] - omega[2] + omega[3])/4
    return opt_alpha

def line_constraint(omega):
    segments = np.zeros((4, 2))
    for i in range(4):
        first = (OMEGA_MIN - omega[i]) * (-1) ** i
        second = (-OMEGA_MIN - omega[i]) * (-1) ** i
        segments[i] = np.sort([first[0], second[0]]).flatten() # Potentially sort manually
    return segments

def overlap_constraint(segments):
    segments = segments[np.argsort(segments[:, 0])]
    merged = [segments[0]]

    for current in segments[1:]:
        last_merged = merged[-1]

        if current[0] <= last_merged[1]:
            merged[-1] = (last_merged[0], max(last_merged[1], current[1]))
        else:
            merged.append(current)
    return np.array(merged)

def constrained_alpha(omega):
    segments = line_constraint(omega)
    segments = overlap_constraint(segments)
    opt_alpha = optimal_alpha(omega)
    for segment in segments:
        start, end = segment[0], segment[1]

        if start <= opt_alpha <= end:
            dist_start = abs(opt_alpha - start)
            dist_end = abs(opt_alpha - end)

            if dist_start == dist_end:
                return start if abs(start) < abs(end) else end
            else:
                return start if dist_start < dist_end else end

    return opt_alpha

def ideal_omega(omega):
    return omega + np.array([[1], [-1], [1], [-1]]) * constrained_alpha(omega)

def solve(torque_data, calc_alpha_func=pseudo, second_func=None):
    global w_current
    low_torque_flag = hysteresis_filter(torque_data, 0.000005, 0.000015)
    for i in range(total_points):
        T_sc = torque_data[:, i].reshape(3, 1)
        if second_func is not None and low_torque_flag[i]:
            alpha = second_func(T_sc, w_current).reshape(1, 1)
        else:
            alpha = calc_alpha_func(T_sc, w_current).reshape(1, 1)
        alpha_sol[:, i] = alpha.flatten()
        T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha

        der_state = I_INV @ T_rw
        w_current = w_current + der_state * 0.1
        w_sol[:, i+1] = w_current.flatten()
        torque_sol[:, i] = T_rw.flatten()
    # torque_sol[:, i] = T_rw.flatten()


def solve_ideal():
    global w_current
    global w_sol
    solve(squared_omega)
    low_torque_flag = hysteresis_filter(full_data, 0.000005, 0.000015)
    # rising, falling = detect_transitions(low_torque_flag)
    # print(rising, falling)

    w, t = symbols('w t')
    a = 0.01
    gaus_expr = exp(-a * w ** 2)  # Gaussian function
    cost_expr = gaus_expr

    current_idx = 0

    while current_idx < len(low_torque_flag):
        if low_torque_flag[current_idx]:  #  (real-time control)
            start_idx = current_idx
            while current_idx < len(low_torque_flag) and low_torque_flag[current_idx]:
                current_idx += 1
            end_idx = current_idx  # End of the True section

            # Loop through each point in the True section for real-time control
            for point in range(start_idx, end_idx):
                T_sc = full_data[:, point].reshape(3, 1)
                alpha = pseudo(T_sc, w_current).reshape(1, 1)

                T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha
                der_state = I_INV @ T_rw
                w_current = w_current + der_state * 0.1

                w_sol[:, point+1] = w_current.flatten()
                alpha_sol[:, point] = alpha.flatten()
                torque_sol[:, point] = T_rw.flatten()


        else:  # (optimization)
            start_idx = current_idx
            while current_idx < len(low_torque_flag) and not low_torque_flag[current_idx]:
                current_idx += 1
            end_idx = current_idx  # End of the False section

            N_points = end_idx - start_idx
            w_end = w_sol[:, end_idx].reshape(4, 1)
            print(f"Optimizing for {N_points} points")
            omega_sol, alpha, T_sol = solve_ocp_index(cost_expr, start_idx, N_points, w_final=w_end)
            print("Done")
            omega_sol = omega_sol.T
            T_sol = T_sol.T
            w_sol[:, start_idx+1:end_idx+1] = omega_sol[:, :-1]
            w_current = omega_sol[:, -1].reshape(4, 1)
            alpha_sol[:, start_idx:end_idx] = alpha[:-1].reshape(1, N_points)
            torque_sol[:, start_idx:end_idx] = T_sol[:, :-1]


def sum_squared(omega):
    return np.sum(omega**2)

def repeat_ideal_omega(omega):
    for i in range(omega.shape[1]):
        omega[:, i] = ideal_omega(omega[:, i].reshape(4, 1)).flatten()
    return omega


def convert_to_index(matrix):
    signs = (matrix >= 0).astype(int)  # Convert positive numbers to 1, negative to 0
    indices = signs[0] * 8 + signs[1] * 4 + signs[2] * 2 + signs[3] * 1  # Binary to decimal conversion
    return indices

def save(name='output'):
    data_to_save = {
        'all_w_sol': w_sol.T,
        'all_alpha_sol': alpha_sol,
        'all_T_sol': torque_sol.T,
        'all_t': np.linspace(0, total_points/10, total_points+1)
    }
    savemat(f'Data/Realtime/{name}.mat', data_to_save)

def save_to_mat(path_base, solve_time=0.0):
    if not path_base.endswith(".mat"):
        path_base += ".mat"

    os.makedirs(os.path.dirname(path_base), exist_ok=True)

    mat_data = {
        "seed": seed,
        "omega_start": w_initial,
        "all_w_sol": w_sol.T,
        "all_T_sol": torque_sol.T,
        "all_alpha_sol": alpha_sol,
        "all_t": np.linspace(0, 800, 8005),
        "solve_time": solve_time,
        "null_sol": nullspace_alpha(w_sol)
    }
    savemat(path_base, mat_data)

def plot():
    plt.plot(w_sol.T)
    plt.show()
    plt.plot(alpha_sol.T)
    plt.show()


def plot_radians(data):
    all_w_sol = data.T
    all_t = np.linspace(0, all_w_sol.shape[0] / 10, all_w_sol.shape[0])
    plt.axhline(y=600, color='r', linestyle='--', label=f'rad/s=600')
    plt.axhline(y=-600, color='r', linestyle='--', label=f'rad/s=-600')
    plt.fill([all_t[0], all_t[0], all_t[-1], all_t[-1]], [-OMEGA_MIN, OMEGA_MIN, OMEGA_MIN, -OMEGA_MIN], 'r', alpha=0.1)
    plt.plot(all_t, all_w_sol)
    # plt.ylim([-300, 300])
    plt.xlabel('Time (s)')
    plt.ylabel('Rad/s')
    plt.title('Rad/s vs Time')


def plot_index(solution):
    index = convert_to_index(solution)
    plt.plot(index)
    plt.title('Ideal Momentum Envelope vs Time')
    plt.ylim([-1, 16])
    plt.ylabel('Index')
    plt.xlabel('Point')
    plt.show()

def target_nullspace(alpha_target, torque, omega_start, specific_torque_limits=False):
    N = 8005
    dt = 0.1
    w = omega_start
    alpha_control_guess = np.zeros(N-1)
    w_guess = np.zeros((4, N))
    w_guess[:, 0] = w.flatten()
    torque_guess = np.zeros((4, N))

    if specific_torque_limits:
        alpha_lower, alpha_upper = calc_alpha_torque_limits(torque, 0, N-1)
        def get_upper_torque_limits(l):
            return alpha_upper[l]
        def get_lower_torque_limits(l):
            return alpha_lower[l]
    else:
        def get_upper_torque_limits(l):
            return 0.1*MAX_TORQUE
        def get_lower_torque_limits(l):
            return -0.1*MAX_TORQUE

    for k in range(N-1):
        T_sc = torque[:, k:k+1]
        T_rw = R_PSEUDO @ T_sc + NULL_R @ np.array([[0]])
        der_state = I_INV @ T_rw
        w_next = w + der_state * dt
        alpha_null = NULL_R_T @ w_next / 4
        alpha_null_wanted = alpha_target
        alpha_diff = alpha_null_wanted - alpha_null

        # T_rw = NULL_R * alpha_diff*IRW/dt
        alpha_guess = alpha_diff*IRW/dt
        alpha_guess = np.clip(alpha_guess, get_lower_torque_limits(k), get_upper_torque_limits(k))

        # T_rw = np.clip(T_rw, -0.1*MAX_TORQUE, 0.1*MAX_TORQUE)
        # der_state = I_INV @ T_rw * alpha_guess
        der_state = I_INV @ NULL_R * alpha_guess
        w = w_next + der_state * dt
        w_guess[:, k+1] = w.flatten()
        torque_guess[:, k] = T_rw.flatten() + (NULL_R @ alpha_guess).flatten()
        alpha_control_guess[k] = alpha_diff.item()

    w_sol[:,:] = w_guess
    torque_sol[:,:] = torque_guess
    alpha_sol[:, :] = alpha_control_guess
    return w_guess, alpha_control_guess, torque_guess

full_data = load_data('Data/Slew1.mat')  # (3, 8004)
total_points = 8004
w_current, w_initial = OMEGA_START, OMEGA_START
# w_current = np.random.uniform(-100, 100, (4, 1))

w_sol = np.zeros((4, total_points+1))
torque_sol = np.zeros((4, total_points+1))
w_sol[:, 0] = w_current.flatten()
alpha_sol = np.zeros((1, total_points))

solve(full_data, pseudo)
run_dir = f"Data/Realtime/pseudo"
os.makedirs(run_dir, exist_ok=True)
#
save_to_mat(f"{run_dir}/torque")

if __name__ == "__main__":
    full_data = load_data('Data/Slew1.mat')
    for seed in range(1, 51):
        break
        continue
        print(f"--- Run {seed} ---")
        total_points = full_data.shape[1]

        # Random initial condition
        w_current, return_seed = get_random_start(seed=seed)
        if return_seed != seed:
            print(f"Seed mismatch: {return_seed} != {seed}")
            break
        w_initial = w_current.copy()

        # Reset arrays
        w_sol = np.zeros((4, total_points + 1))
        torque_sol = np.zeros((4, total_points + 1))
        alpha_sol = np.zeros((1, total_points))
        w_sol[:, 0] = w_current.flatten()

        # === Pseudoinverse ===
        begin_time = clock.time()
        target_nullspace(0, full_data, w_initial, specific_torque_limits=True)
        solve_time = clock.time() - begin_time
        save_to_mat(f"Data/conventional/pseudo_omega/slew1/{seed}.mat", solve_time)

        # === Min-Max ===
        begin_time = clock.time()
        solve(full_data, minmax_omega)
        solve_time = clock.time() - begin_time
        save_to_mat(f"Data/conventional/minmax_omega/slew1/{seed}.mat", solve_time)

