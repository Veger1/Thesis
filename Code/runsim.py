from scipy.io import savemat
from DAC import *

def run_with_config(omega_start):
    start_time = clock.time()
    torque_data = load_data('Data/Slew1.mat')
    low_torque_flag = hysteresis_filter(torque_data, 0.000005, 0.000015)
    low_torque_flag[0:2] = False
    rising, falling = detect_transitions(low_torque_flag)
    stationary = np.insert(falling - 1, 0, 0)
    stationary = np.append(stationary, len(low_torque_flag))
    falling = np.insert(falling, 0, 0)
    falling = np.append(falling, len(low_torque_flag))  # +1 but works?

    momentum4_with_nullspace = pseudo_sol(torque_data, OMEGA_START)
    alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:, 0:1])
    momentum3 = R @ momentum4_with_nullspace
    momentum4 = R_PSEUDO @ momentum3
    segments = calc_segments(momentum4)
    indices, solution_space, overlap_space = calc_indices(segments, low_torque_flag, rising)

    problem = []
    options = alpha_options(momentum4)
    for i, end_index in enumerate(rising):
        start_index = falling[i]
        stationary_index = stationary[i + 1]
        section_name = f"Section_{i + 1}"
        section = Section(section_name, start_index, end_index, stationary_index, options, momentum4)
        section.populate_section(start_index, end_index, stationary_index, indices)
        problem.append(section)

    source_index = find_start_node(problem, OMEGA_START)  # source_ids = ["Section_1_0_3"]
    target_index = ["goal_8100_0"]
    graph = build_graph(problem)
    assign_mixed_costs(graph, penalty=10000)
    path, mixed_cost, sign_cost, vib_cost = shortest_path(graph, source_index, target_index, cost_type="mixed_cost")
    mid1_time = clock.time()

    alpha_points = extract_info_from_path(graph, path)
    alpha_min, alpha_max = build_alpha_constraints(alpha_points, segments)
    guess_alpha, guess_omega = find_initial_guess(alpha_min, alpha_max, momentum4)
    w_sol, alpha_sol = init_guesser(guess_alpha, torque_data)
    null_sol = nullspace_alpha(w_sol)
    mid2_time = clock.time()

    omega_min, omega_max = build_omega_constraints(alpha_min, alpha_max, momentum4)
    solver = Solver(torque_data, omega_limits=(omega_min, omega_max))
    # solver = Solver(torque_data, omega_limits=(omega_min, omega_max), omega_guess=w_sol, control_guess = alpha_sol)
    w_cas_sol, alpha_cas_sol, torque_cas_sol, t1, t2, t3 = solver.oneshot_casadi(n0=0, N=8004, torque_flag=True)
    null_cas_sol = nullspace_alpha(w_cas_sol)
    end_time = clock.time()

    extra = (alpha_min, alpha_max, guess_alpha, guess_omega, omega_min, omega_max)
    extra_time = (mid1_time-start_time, mid2_time-mid1_time, end_time-mid2_time, t1, t2, t3)
    return w_sol, alpha_sol, null_sol, w_cas_sol, alpha_cas_sol, null_cas_sol, torque_cas_sol, extra, extra_time

results = {}
for z in range(31):
    OMEGA_START, seed = get_random_start()
    print(seed)
    try:
        error_code = None
        begin_time = clock.time()
        w_sol, alpha_sol, null_sol, w_cas_sol, alpha_cas_sol, null_cas_sol, torque_cas_sol, extra, extra_time = run_with_config(OMEGA_START)
        end_time = clock.time()
        total_time = end_time - begin_time
        print(f"Iteration {z + 1} completed in {total_time} seconds.")
        results[seed] = {
            "seed": seed,
            "w_sol": w_sol,
            "alpha_sol": alpha_sol,
            "null_sol": null_sol,
            "w_cas_sol": w_cas_sol,
            "alpha_cas_sol": alpha_cas_sol,
            "null_cas_sol": null_cas_sol,
            "torque_cas_sol": torque_cas_sol,
            "extra_time": extra_time,
            "total_time": total_time,
            "omega_start": OMEGA_START
        }
    except Exception as e:
        error_code = e
        print(f"Error occurred in iteration {z + 1}: {error_code}")
        results[seed] = {
            "seed": seed,
            "error": str(error_code)
        }

    savemat(f'Data/DAC/slew1/reference/{seed}.mat', results[seed])
savemat(f'Data/DAC/slew1/reference.mat', results[seed])


