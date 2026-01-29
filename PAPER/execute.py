from algorithm import *


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


timer = 0.0
success = 0
for _ in range(500):

    OMEGA_START = np.random.uniform(-300, 300, (4, 1))
    OMEGA_START_PSEUDO = R_PSEUDO @ R @ OMEGA_START
    OMEGA_START_NULL = OMEGA_START - OMEGA_START_PSEUDO
    timing, result, layer_list = safe_run()
    if timing is not None:
        timer += timing["total_time"]
        success += 1
print(f"Total time accumulated: {timer:.2f} seconds over {success} successful runs.")
print(f"Average time: {timer / success:.2f}")