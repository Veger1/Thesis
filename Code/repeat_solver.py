from casadi import sum1
from rockit import Ocp, MultipleShooting
import numpy as np
from init_helper import load_data, initialize_constants
from sympy import symbols, lambdify

def solve_ocp(objective_expr, num_intervals, N, time, scaling):
    test_data_T_full = load_data()  # This is the full test data (8004 samples)
    helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()

    w_initial = Omega_start  # Start with the initial condition
    section_length = 200  # Number of points per section

    all_t, all_w_sol, all_alpha_sol, all_T_sol = [], [], [], []

    for i in range(num_intervals):
        t0_actual = time * i
        t0_relative = t0_actual % 200
        t0_offset = t0_actual//200*200
        ocp = Ocp(t0=t0_relative, T=time)  # Create the OCP for the current interval
        w = ocp.state(4)  # 4 states for angular velocities
        alpha = ocp.control()  # control input (acceleration)
        T_sc = ocp.parameter(3, grid='control')  # Set torque as a parameter (3 torques, using 'control' grid)

        test_data_T = test_data_T_full[:, i*N:(i+1)*N]  # Extract N points from data
        ocp.set_value(T_sc, test_data_T)  # Assign the torque constraint

        T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw * scaling @ alpha
        der_state = I_inv @ T_rw
        ocp.set_der(w, der_state)

        ocp.subject_to(-T_max <= (T_rw <= T_max))  # Add torque constraints
        ocp.subject_to(-Omega_max <= (w <= Omega_max))  # Add saturation constraints
        # Set initial conditions
        ocp.subject_to(ocp.at_t0(w) == w_initial)
        ocp.set_initial(w, w_initial)  # Set initial guess

        # a = 0.001
        # b = 1/7000000
        # objective_expr_casadi = np.exp(-a * w ** 2) + (b*w ** 2) * (8 / (1 + np.exp(-0.3 * (ocp.t - 60))))
        ocp_t = ocp.t
        w_sym, t_sym = symbols('w t')
        objective_expr_casadi = lambdify((w_sym, t_sym), objective_expr, 'numpy')
        objective_expr_casadi = objective_expr_casadi(w, ocp_t)
        objective = ocp.integral(sum1(objective_expr_casadi))
        ocp.add_objective(objective)

        solver_opts = {
            "print_time": False,  # Suppress overall solver timing
            "ipopt": {
                "print_level": 0,  # Disable IPOPT output
                "sb": "yes"  # Suppress banner output
            }
        }

        ocp.solver('ipopt', solver_opts)  # Use IPOPT solver
        ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
        sol = ocp.solve()  # Solve the problem

        # Post-processing: Sample solutions for this interval
        ts, w_sol = sol.sample(w, grid='control')
        _, alpha_sol = sol.sample(alpha, grid='control')
        _, T_rw_sol = sol.sample(T_rw, grid='control')
        w_initial = w_sol[-1]  # Last value of the current interval

        ts = ts + t0_offset

        all_t.append(ts[:-1])
        all_w_sol.append(w_sol[:-1])  # Last value is unique
        all_alpha_sol.append(alpha_sol[:-1])
        all_T_sol.append(T_rw_sol[:-1])

    # Concatenate the data along the first axis (if they're all 1D arrays)
    all_t = np.concatenate(all_t)
    all_w_sol = np.concatenate(all_w_sol)
    all_alpha_sol = np.concatenate(all_alpha_sol)
    all_T_sol = np.concatenate(all_T_sol)

    return all_t, all_w_sol, all_alpha_sol, all_T_sol

def fast_solve_ocp(objective_expr, num_intervals, N, time, scaling):
    test_data_T_full = load_data()
    helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()
    w_initial = Omega_start

    ocp = Ocp(t0=0, T=time)
    w = ocp.state(4)
    alpha = ocp.control()
    T_sc = ocp.parameter(3, grid='control')
    w0 = ocp.parameter(4)
    ocp.set_value(w0, w_initial)
    test_data_T = test_data_T_full[:, :N]
    ocp.set_value(T_sc, test_data_T)  # Assign the torque constraint, to_function does not work if this is not done

    T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw * scaling @ alpha
    der_state = I_inv @ T_rw
    ocp.set_der(w, der_state)

    ocp.subject_to(-T_max <= (T_rw <= T_max))  # Add torque constraints
    ocp.subject_to(-Omega_max <= (w <= Omega_max))  # Add saturation constraints

    ocp.subject_to(ocp.at_t0(w) == 0)
    ocp.set_initial(w, w0)
    ocp.set_initial(alpha, 0)

    w_sym = symbols('w')
    objective_expr_casadi = lambdify(w_sym, objective_expr, 'numpy')
    objective_expr_casadi = objective_expr_casadi(w)
    objective = ocp.integral(sum1(objective_expr_casadi))
    ocp.add_objective(objective)

    ocp.solver('ipopt')  # Use IPOPT solver
    ocp.method(MultipleShooting(N=N, M=1, intg='rk'))

    states = ocp.sample(w, grid='control')[1]
    states2 = ocp.sample(w, grid='control-')[1]
    controls = ocp.sample(alpha, grid='control')[1]
    constraint = ocp.sample(T_sc, grid='control-')[1]
    torque = ocp.sample(T_rw, grid='control')[1]
    torque_input = ocp.sample(T_sc, grid='control')[1]
    torque_input2 = ocp.sample(T_sc, grid='control-')[1]

    prob_solve = ocp.to_function('prob_solve',
                                 [w0, constraint],
                                 [states, states2, controls, torque, torque_input, torque_input2],
                                 ["w_init","torque_in"],
                                 ["w_sol","w_sol2","alpha_sol","torque_sol","input_torque","input_torque2"])

    all_t = np.linspace(0, time*num_intervals, N*num_intervals)
    all_w_sol = []
    all_alpha_sol = []
    all_torque_sol = []
    for i in range(num_intervals):
        test_data_T = test_data_T_full[:, i*N:(i+1)*N]
        sol = prob_solve(w_init=w_initial, torque_in=test_data_T)
        w_sol, w_sol2, alpha_sol, torque_sol, input_sol, input_sol2 = sol['w_sol'], sol['w_sol2'], sol['alpha_sol'], sol['torque_sol'], sol['input_torque'], sol['input_torque2']
        w_initial = w_sol[:,-1]
        all_w_sol.append(w_sol[:,:-1])
        all_alpha_sol.append(alpha_sol[:,:-1])
        all_torque_sol.append(torque_sol[:,:-1])

    all_w_sol = np.concatenate(all_w_sol, axis=1).transpose()
    all_alpha_sol = np.concatenate(all_alpha_sol, axis=1)
    all_torque_sol = np.concatenate(all_torque_sol, axis=1).transpose()

    return all_t, all_w_sol, all_alpha_sol, all_torque_sol


def unconstrained_solve_ocp(objective_expr, num_intervals, N, time, scaling):
    test_data_T_full = load_data()  # This is the full test data (8004 samples)
    helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max = initialize_constants()

    w_initial = Omega_start  # Start with the initial condition

    all_t = []
    all_w_sol = []
    all_alpha_sol = []
    all_T_sol = []

    for i in range(num_intervals):
        ocp = Ocp(t0=time*i, T=time)  # Create the OCP for the current interval

        w = ocp.state(4)  # 4 states for angular velocities
        alpha = ocp.control()  # control input (acceleration)
        T_sc = ocp.parameter(3, grid='control')  # Set torque as a parameter (3 torques, using 'control' grid)
        s1 = ocp.variable(4, grid='control')
        s2 = ocp.variable(4, grid='control')

        test_data_T = test_data_T_full[:, i*N:(i+1)*N ]  # Extract N points from data
        ocp.set_value(T_sc, test_data_T)  # Assign the torque constraint

        T_rw = R_rwb_pseudo @ T_sc + Null_Rbrw * scaling @ alpha
        der_state = I_inv @ T_rw
        ocp.set_der(w, der_state)
        # ocp.set_der(a, j)
        # jerk_min, jerk_max = -0.005 , 0.005
        # ocp.subject_to(jerk_min <= (j <= jerk_max))

        ocp.subject_to(-Omega_max <= (w <= Omega_max))  # Add saturation constraints
        ocp.subject_to(0 <= s1)
        ocp.subject_to(0 <= s2)

        c1, c2 = T_rw - T_max, -T_rw - T_max
        lambda1 = 10000000
        mu1 = 100000

        # torque_constraint_expr = np.exp(T_rw+T_max) + np.exp(-T_rw+T_max)
        # speed_constraint_expr = np.exp(w+Omega_max) + np.exp(-w+Omega_max)
        # torque_constraint_expr = np.log(1+np.exp(T_rw-T_max)) + np.log(1+np.exp(-T_rw-T_max))
        # speed_constraint_expr = np.log(1+np.exp(w-Omega_max)) + np.log(1+np.exp(-w-Omega_max))
        torque_constraint_expr = lambda1*(c1+s1) + mu1/2*(c1+s1)**2 + lambda1*(c2+s2) + mu1/2*(c2+s2)**2
        ocp.add_objective(ocp.integral(sum1(torque_constraint_expr)))
        # ocp.add_objective(ocp.integral(sum1(speed_constraint_expr)))

        # Set initial conditions
        ocp.subject_to(ocp.at_t0(w) == w_initial)
        ocp.set_initial(w, w_initial)  # Set initial guess

        w_sym = symbols('w')
        objective_expr_casadi = lambdify(w_sym, objective_expr, 'numpy')
        objective_expr_casadi = objective_expr_casadi(w)

        objective = ocp.integral(sum1(objective_expr_casadi))
        ocp.add_objective(objective)

        ocp.solver('ipopt')  # Use IPOPT solver
        ocp.method(MultipleShooting(N=N, M=1, intg='rk'))

        sol = ocp.solve()  # Solve the problem

        # Post-processing: Sample solutions for this interval
        ts, w_sol = sol.sample(w, grid='control')
        _, alpha_sol = sol.sample(alpha, grid='control')
        _, T_rw_sol = sol.sample(T_rw, grid='control')

        # Update the initial condition for the next interval
        w_initial = w_sol[-1]  # Last value of the current interval

        all_t.append(ts[:-1])
        all_w_sol.append(w_sol[:-1])  # Last value is unique
        all_alpha_sol.append(alpha_sol[:-1])
        all_T_sol.append(T_rw_sol[:-1])

    # Concatenate the data along the first axis (if they're all 1D arrays)
    all_t = np.concatenate(all_t)
    all_w_sol = np.concatenate(all_w_sol)
    all_alpha_sol = np.concatenate(all_alpha_sol)
    all_T_sol = np.concatenate(all_T_sol)

    return all_t, all_w_sol, all_alpha_sol, all_T_sol


def calc_cost(w_sol, cost_expr, t_vals=None):
    w, t = symbols('w t')
    # Check if the cost expression depends on t
    if t in cost_expr.free_symbols:
        if t_vals is None:
            raise ValueError("t_vals must be provided when cost_expr depends on time.")

        # Convert symbolic expression to numpy function with two arguments (w, t)
        cost_func = lambdify((w, t), cost_expr, 'numpy')

        # Evaluate cost over w_sol and t_vals
        cost = np.array([cost_func(w_val, t_vals[i] % 200) for i, w_val in enumerate(w_sol)])

    else:
        # Convert symbolic expression to numpy function with only w
        cost_func = lambdify(w, cost_expr, 'numpy')
        cost = cost_func(w_sol)

    # Generate cost graph over an omega range
    omega_axis = np.linspace(-600, 600, 600)
    if t in cost_expr.free_symbols:
        cost_graph = np.array([cost_func(omega, t_vals[0]) for omega in omega_axis])  # Use first t value
    else:
        cost_graph = cost_func(omega_axis)

    # Compute total cost over all time steps
    total_cost = np.sum(cost, axis=1) if cost.ndim > 1 else np.sum(cost)
    total_cost = np.array(total_cost, dtype=float)

    return cost, total_cost, cost_graph, omega_axis
