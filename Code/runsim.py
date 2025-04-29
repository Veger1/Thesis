from DAC import *

def run_with_config(omega_start):
    torque_data = load_data('Data/Slew1.mat')
    low_torque_flag = hysteresis_filter(torque_data, 0.000005, 0.000015)
    low_torque_flag[0:2] = False
    rising, falling = detect_transitions(low_torque_flag)
    stationary = np.insert(falling - 1, 0, 0)
    stationary = np.append(stationary, len(low_torque_flag))
    falling = np.insert(falling, 0, 0)
    falling = np.append(falling, len(low_torque_flag) + 1)

    momentum4_with_nullspace = pseudo_sol(torque_data, omega_start)
    alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:,0:1])
    momentum3 = R @ momentum4_with_nullspace
    momentum4 = R_PSEUDO @ momentum3
    segments = calc_segments(momentum4)
    indices, solution_space, overlap_space = calc_indices(segments, low_torque_flag, rising)

    problem = []
    options = alpha_options(momentum4)
    for i, end_index in enumerate(rising):
        start_index = falling[i]
        stationary_index = stationary[i+1]
        section_name = f"Section_{i+1}"
        section = Section(section_name, start_index, end_index, stationary_index, options, momentum4)
        section.populate_section(start_index, end_index, stationary_index, indices)
        problem.append(section)

    source_index = find_start_node(problem, omega_start)  # source_ids = ["Section_1_0_3"]
    target_index = ["goal_8100_0"]
    graph = build_graph(problem)
    assign_mixed_costs(graph, penalty=10000)
    path, mixed_cost, sign_cost, vib_cost = shortest_path(graph, source_index, target_index, cost_type="mixed_cost")

    alpha_points = extract_info_from_path(graph, path)
    alpha_min, alpha_max = build_alpha_constraints(alpha_points, segments)
    omega_min, omega_max = build_omega_constraints(alpha_min, alpha_max, momentum4)
    guess_alpha, guess_omega = find_initial_guess(alpha_min, alpha_max, momentum4)
    w_sol, alpha_sol = init_guesser(guess_alpha, torque_data)
    null_sol = nullspace_alpha(w_sol)
    return w_sol, alpha_sol, null_sol, guess_alpha, guess_omega

for z in range(50):
    OMEGA_START, seed = get_random_start()
    error_code = None
    try:
        begin_time = clock.time()
        w_sol, alpha_sol, null_sol, guess_alpha, guess_omega = run_with_config(OMEGA_START)
        end_time = clock.time()
        print(f"Iteration {z + 1} completed in {end_time - begin_time:.2f} seconds.")
    except Exception as e:
        error_code = e
        print(f"Error occurred in iteration {z + 1}: {error_code}")
        pass
