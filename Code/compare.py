from DAC import *

list_of_seeds = list(range(1, 51))
data = load_data('Data/Slew1.mat')

def save_results(temp_seed, temp_omega_start, temp_w_sol, temp_alpha_sol, temp_torque_sol, temp_null_sol,
                 temp_solve_time, temp_setup_time, temp_iteration_count, temp_time_sol):
    data_to_save = {
        'seed': temp_seed,
        'omega_start': temp_omega_start,
        'solve_time': temp_solve_time,
        'setup_time': temp_setup_time,
        'iteration_count': temp_iteration_count,
        'all_w_sol': temp_w_sol.T,
        'all_alpha_sol': temp_alpha_sol,
        'all_T_sol': temp_torque_sol.T,
        'null_sol': temp_null_sol,
        'all_t': temp_time_sol,
    }

    filename = f'Data/compare/slew2/5000/{temp_seed}.mat'
    savemat(filename, data_to_save)


for seed in list_of_seeds:
    omega_start, return_seed = get_random_start(seed=seed)
    if return_seed != seed:
        print(f"Seed mismatch: {return_seed} != {seed}")
        break
    begin_time = clock.time()
    try:
        (alpha_points, alpha_min, alpha_max, omega_min, omega_max, guess_alpha,
         guess_omega, w_sol, torque_sol, alpha_sol, null_sol, speeds, speed_segments, graph_problem)\
            = solve(omega_start, data, specific_torque_limits=True, penalty=5000)
    except Exception as e:
        print(f"Error during solve: {e}")
        continue
    solve_time = clock.time() - begin_time
    # Save results
    save_results(seed, omega_start, w_sol, alpha_sol, torque_sol,
                 null_sol,solve_time, 0, 0, np.linspace(0, 800, 8005))