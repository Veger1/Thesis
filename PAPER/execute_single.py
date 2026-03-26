from algorithm import *
from metrics import *
import pandas as pd
from tabulate import tabulate

def safe_run(w_start, k_given):
    try:
        w_start_pseudo = R_PSEUDO @ R @ w_start
        t0 = time.perf_counter()
        w_pseudo, torque_pseudo = forward_integration(w_start_pseudo, data, dt=0.1)

        t1 = time.perf_counter()
        overlap_mask_sorted, band_pairs_mask_sorted, band_order, band_center, band_radii = make_direct_overlap_masks(
            w_pseudo)

        t2 = time.perf_counter()
        free, not_crossing, crossing = generate_all_intervals(overlap_mask_sorted, band_pairs_mask_sorted, band_center)

        t3 = time.perf_counter()
        selected_layers = [0, 689, 2003, 2690, 4004, 5100, 6005, 7101, 8004]
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
        K = k_given
        restricted_intervals = []
        G = build_graph(node_list, sign_list, K, new_layers, restricted_intervals)

        t8 = time.perf_counter()
        shortest_path = solve_graph(G, start_layer=0, end_layer=8004, omega_start_sign=np.sign(w_start.flatten()))

        t9 = time.perf_counter()
        path_constraint = calc_alpha_limits(band_center, band_radii, shortest_path, node_list, new_layers)

        t10 = time.perf_counter()
        torque_constraint = calc_alpha_torque_limits(torque_pseudo)

        t11 = time.perf_counter()
        w_sol, alpha_sol, torque_sol = forward_integration_optimal(w_start, path_constraint, torque_constraint,
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

        result_run = (w_sol, alpha_sol, torque_sol)
        print("Execution completed successfully. Timing:", timing["total_time"])
        return timing, result_run, new_layers
    except Exception as e:
        print("An error occurred during execution:", str(e))
        return None

def evaluate(result_tuple, layers):
    w_sol, alpha_sol, torque_sol = result_tuple
    stiction_time = time_stiction_accurate(w_sol, OMEGA_MIN, dt=0.1)
    energy_metrics = energy(w_sol, torque_sol, dt=0.1)
    zero_crossing = count_zero_crossings(w_sol)
    omega_squared = omega_squared_sum(w_sol, OMEGA_MIN)/8005  # normalize by number of time steps
    number_of_layers = len(layers)
    result_dict = {
        "stiction_time": np.sum(stiction_time),
        "energy": np.sum(energy_metrics),
        "zero_crossing": np.sum(zero_crossing),
        "omega_squared": np.sum(omega_squared),
        "number_of_layers": number_of_layers}
    return result_dict

def run_pseudo(w_start):
    time0 = time.perf_counter()
    opt_sol = np.zeros(8005)
    torque_psd = R_PSEUDO @ data  # (4, N)
    torque_lim = calc_alpha_torque_limits(torque_psd)
    time1 = time.perf_counter()
    w_sol, alpha_sol, torque_sol = forward_integration_optimal(w_start, opt_sol, torque_lim, torque_psd, dt=0.1)
    time2 = time.perf_counter()
    timing = {"total_time": time2 - time0, "forward_integration_optimal": time2 - time1, "constraints": time1 - time0}
    return timing, (w_sol, alpha_sol, torque_sol), []

def run_minmax(w_start):
    time0 = time.perf_counter()
    w_start_pseudo = R_PSEUDO @ R @ w_start
    w_pseudo, torque_pseudo = forward_integration(w_start_pseudo, data, dt=0.1)
    opt_sol = minmax_alpha_from_wpseudo(w_pseudo)
    torque_psd = R_PSEUDO @ data  # (4, N)
    torque_lim = calc_alpha_torque_limits(torque_psd)
    time1 = time.perf_counter()
    w_sol, alpha_sol, torque_sol = forward_integration_optimal(w_start, opt_sol, torque_lim, torque_psd, dt=0.1)
    time2 = time.perf_counter()
    timing = {"total_time": time2 - time0, "forward_integration_optimal": time2 - time1, "constraints": time1 - time0}
    return timing, (w_sol, alpha_sol, torque_sol), []

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


def run_and_evaluate(sims=1, k_given=10):
    results = []
    np.random.seed(1)
    for i in range(sims):  # Adjust the range for the desired number of runs
        OMEGA_START = np.random.uniform(-300, 300, (4, 1))
        # print("omega_start:", OMEGA_START.flatten())
        timing_safe, result_safe, layers_safe = safe_run(OMEGA_START, k_given=k_given)
        if timing_safe is not None:
            metrics_safe = evaluate(result_safe, layers_safe)
        else:
            metrics_safe = None

        # timing_minmax, result_minmax, layers_minmax = run_minmax(OMEGA_START)
        # metrics_minmax = evaluate(result_minmax, layers_minmax)

        # timing_pseudo, result_pseudo, layers_pseudo = run_pseudo(OMEGA_START)
        # metrics_pseudo = evaluate(result_pseudo, layers_pseudo)

        method, metrics, timings = "graph_10000000", metrics_safe, timing_safe

        results.append({
            "Iteration": i,
            "Omega Start": OMEGA_START.flatten(),
            "Stiction Time": metrics["stiction_time"],
            "Energy": metrics["energy"],
            "Zero Crossings": metrics["zero_crossing"],
            "Omega² Avg": metrics["omega_squared"],
            "Number of Layers": metrics["number_of_layers"],
            "Timing": timings
        })

    results_df = pd.DataFrame(results)
    results_df.to_excel("simulation_results.xlsx", index=False)
    print("All simulations completed. Results saved to 'simulation_results.xlsx'.")


run_and_evaluate(sims=50, k_given=1000)