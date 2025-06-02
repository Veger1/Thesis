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
    savemat(filename, data_to_save)

def save_class(class_instance, temp_seed, temp_omega_start):
    data_to_save = {
        'seed': temp_seed,
        'omega_start': temp_omega_start,
        'solve_time': class_instance.solve_time,
        'setup_time': class_instance.setup_time,
        'iteration_count': class_instance.iteration_count,
        'all_w_sol': class_instance.w_sol.T,
        'all_alpha_sol': class_instance.alpha_sol,
        'all_T_sol': class_instance.T_rw_sol.T,
        'null_sol': class_instance.null_sol,
        'all_t': class_instance.temp_time,
    }
    savemat(filename, data_to_save)

solver_times = []
for seed in list_of_seeds:
    print(f"Processing seed: {seed}")
    omega_start, return_seed = get_random_start(seed=seed)
    if return_seed != seed:
        print(f"Seed mismatch: {return_seed} != {seed}")
        break
    begin_time = clock.time()
    try:
        (alpha_points, alpha_min, alpha_max, omega_min, omega_max, guess_alpha,
         guess_omega, w_sol, torque_sol, alpha_sol, null_sol, speeds, speed_segments, graph_problem)\
            = solve(omega_start, data, specific_torque_limits=True, penalty=0)

        # og = Solver(data, omega_start , omega_selective_limits=(omega_min, omega_max)
        #             , reduce_torque_limits=True)
        # og.oneshot_casadi(n0=0, N=8004, torque_on_g=False, omega_on_g=False, penalise_stiction=True)

    except Exception as e:
        print(f"Error during solve: {e}")
        continue
    solve_time = clock.time() - begin_time
    solver_times.append(solve_time)
    # Save results
    # filename = f'Data/DAC/compare_convex_guess/slew2/10000/{seed}.mat'
    # filename = f'Data/DAC/full_objective/slew1/0/{seed}.mat'
    try:
        continue
        save_class(og, seed, omega_start)
        # save_results(seed, omega_start, w_sol, alpha_sol, torque_sol,
        #          null_sol,solve_time, 0, 0, np.linspace(0, 800, 8005))
    except Exception as e:
        print(f"Error during save: {e}")
        continue

# print(sum(solver_times) / len(solver_times))