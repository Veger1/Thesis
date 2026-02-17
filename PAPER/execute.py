import numpy as np

from algorithm import *
from metrics import *

def run():
    t0 = time.perf_counter()
    w_pseudo, torque_pseudo = forward_integration(OMEGA_START_PSEUDO, data, dt=0.1)

    t1 = time.perf_counter()
    overlap_mask_sorted, band_pairs_mask_sorted, band_order, band_center, band_radii = make_direct_overlap_masks(w_pseudo)

    t2 = time.perf_counter()
    free, not_crossing, crossing = generate_all_intervals(overlap_mask_sorted, band_pairs_mask_sorted, band_center)

    t3 = time.perf_counter()
    selected_layers = [0, 8004]
    free_flat = flatten_list_of_lists(free)
    not_crossing_flat = flatten_list_of_lists(not_crossing)
    crossing_flat = flatten_list_of_lists(crossing)
    all = free_flat + not_crossing_flat + crossing_flat

    t4 = time.perf_counter()
    new_layers = set_covering(all, selected_layers)

    t5 = time.perf_counter()
    new_layers = set_covering_greedy(all, selected_layers)

    t6 = time.perf_counter()
    node_list, sign_list = determine_nodes(w_pseudo, band_center, band_radii, new_layers)

    t7 = time.perf_counter()
    K = 10
    restricted_intervals = []
    G = build_graph(node_list, sign_list, K, restricted_intervals)

    t8 = time.perf_counter()
    shortest_path = solve_graph(G, start_layer=0, end_layer=8004, omega_start_sign=np.sign(OMEGA_START.flatten()))

    t9 = time.perf_counter()
    path_constraint = calc_alpha_limits(band_center, band_radii, shortest_path, node_list, new_layers)

    t10 = time.perf_counter()
    torque_constraint = calc_alpha_torque_limits(torque_pseudo)

    t11 = time.perf_counter()
    w_sol, alpha_sol, torque_sol = forward_integration_optimal(OMEGA_START, path_constraint, torque_constraint,
                                                               torque_pseudo, dt=0.1)
    t12 = time.perf_counter()

    timing = {
        "forward_integration": t1 - t0,
        "overlap_masks": t2 - t1,
        "generate_intervals": t3 - t2,
        "set_covering": t4 - t3,
        "set_covering_greedy": t5 - t4,
        "determine_nodes": t6 - t5,
        "build_graph": t7 - t6,
        "solve_graph": t8 - t7,
        "calc_alpha_limits": t9 - t8,
        "calc_alpha_torque_limits": t10 - t9,
        "forward_integration_optimal": t11 - t10,
        "total_time": t12 - t0
    }

    return timing, (w_sol, alpha_sol, torque_sol), new_layers

def safe_run():

    try:
        timing_run, result_run, layers = run()
        print("Execution completed successfully. Timing:", timing_run["total_time"])
    except Exception as e:
        print("An error occurred during execution:", str(e))
        return None
    return timing_run, result_run, layers

def evaluate(w_sol, alpha_sol, torque_sol, layers):
    stiction_time = time_stiction_accurate(w_sol, OMEGA_MIN, dt=0.1)
    energy_metrics = energy(w_sol, torque_sol, dt=0.1)
    zero_crossing = count_zero_crossings(w_sol)
    omega_squared = omega_squared_sum(w_sol, OMEGA_MIN)
    number_of_layers = len(layers)
    return stiction_time, energy_metrics, zero_crossing, omega_squared, number_of_layers

def run_pseudo():
    time0 = time.perf_counter()
    opt_sol = np.zeros(8004)
    torque_psd = R_PSEUDO @ data  # (4, N)
    torque_lim = calc_alpha_torque_limits(torque_psd)
    time1 = time.perf_counter()
    forward_integration_optimal(OMEGA_START, opt_sol, torque_lim, torque_psd, dt=0.1)
    time2 = time.perf_counter()
    timing = {"total_time": time2 - time0, "forward_integration_optimal": time2 - time1, "constraints": time1 - time0}
    return timing, (w_sol, alpha_sol, torque_sol), None


def minmax_alpha_from_wpseudo(w_pseudo):
    """
    Fast min-max alpha assuming |NULL_R_i| are identical.

    :param w_pseudo: (4,N)
    :return: alpha_optimal (N,)
    """

    N_vec = NULL_R.ravel()

    # alpha_null = -w / N
    alpha_null = -w_pseudo / N_vec[:, None]

    alpha_optimal = 0.5 * (
            np.max(alpha_null, axis=0)
            + np.min(alpha_null, axis=0)
    )

    return alpha_optimal

def solve():
    N = 8004
    w_sol = np.zeros((4, N+1))
    alpha_sol = np.zeros(N)
    torque_sol = np.zeros((4, N))

    w_current = OMEGA_START.flatten()
    w_sol[:, 0] = w_current

    for i in range(N):
        T_sc = data[:, i]
        alpha = minmax_omega(T_sc, w_current, dt=0.1)
        alpha_sol[i] = alpha
        T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha

        der_state = I_INV @ T_rw
        w_current += der_state * 0.1
        w_sol[:, i+1] = w_current.flatten()
        torque_sol[:, i] = T_rw.flatten()

def minmax_omega(torque_sc, omega, dt=0.1):
    nominator = - omega - dt * I_INV @ R_PSEUDO @ torque_sc
    denominator = dt * I_INV @ NULL_R
    alpha_null = nominator / denominator
    alpha_best = (max(alpha_null) + min(alpha_null)) / 2
    return alpha_best

def pseudo_omega(torque_sc, omega, dt=0.1):
    opt_alpha  = (- omega[0] + omega[1] - omega[2] + omega[3])/4

    return - dt * I_INV @ R_PSEUDO @ torque_sc + NULL_R * alpha

timer = 0.0
success = 0
for _ in range(1):

    OMEGA_START = np.random.uniform(-300, 300, (4, 1))
    OMEGA_START_PSEUDO = R_PSEUDO @ R @ OMEGA_START
    OMEGA_START_NULL = OMEGA_START - OMEGA_START_PSEUDO
    timing, result, layer_list = safe_run()
    metrics = evaluate(result[0], result[1], result[2], layer_list)
    print("Metrics: Stiction Time:", metrics[0], "Energy:", metrics[1], "Zero Crossings:", metrics[2],
          "Omega Squared Sum:", metrics[3], "Number of Layers:", metrics[4])
    if timing is not None:
        timer += timing["total_time"]
        success += 1
print(f"Total time accumulated: {timer:.2f} seconds over {success} successful runs.")
print(f"Average time: {timer / success:.2f}")