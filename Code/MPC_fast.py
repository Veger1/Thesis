import numpy as np
from casadi import sum1
from rockit import Ocp, MultipleShooting
from config import *
import sys


def solve(obj_expr, total_points=8000, horizon=1, full_data=None, w_start = OMEGA_START):
    if full_data is None:
        full_data = load_data('Data/Slew1.mat')

    all_w = np.zeros((4, horizon + 1, total_points))  # (4, H+1, N)
    actual_w = np.zeros((4, total_points))  #  (4, N)
    all_alpha = np.zeros(total_points)  # (N)
    all_T_rw = np.zeros((4, total_points))  # (4, N)

    w_current, w_initial = w_start, w_start

    ocp = Ocp(t0=0, T=horizon / 10)
    w = ocp.state(4)
    alpha = ocp.control()
    T_sc = ocp.parameter(3, grid='control')

    w0 = ocp.parameter(4)  # Initial state/guess parameter
    ocp.set_value(w0, w_initial)

    # data = full_data[:, :horizon]  # Is this necessary?
    data = full_data[:, 0].reshape(3, 1).repeat(horizon, axis=1)

    ocp.set_value(T_sc, data)

    T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha  # Model/Dynamics
    der_state = I_INV @ T_rw
    ocp.set_der(w, der_state)

    ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))  # Constraints
    ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))
    ocp.subject_to(ocp.at_t0(w) == w0)
    ocp.set_initial(w, w0)  # Only work for FIRST interval, does not update for subsequent intervals

    # a = 0.01
    # b = 1 / 700000
    # offset = 60
    # k = 1.0
    # time_dependent = (1 + np.tanh(k * (ocp.t - offset))) / 2
    # objective_expr_casadi = np.exp(-a * w ** 2) #+ time_dependent*b * w ** 2
    objective = ocp.integral(sum1(obj_expr))
    ocp.add_objective(objective)

    ocp.solver('ipopt', SOLVER_OPTS)
    ocp.method(MultipleShooting(N=horizon, M=1, intg='rk'))

    constraint = ocp.sample(T_sc, grid='control-')[1]  # input/output

    states = ocp.sample(w, grid='control')[1]  # output
    controls = ocp.sample(alpha, grid='control-')[1]
    torque = ocp.sample(T_rw, grid='control')[1]

    prob_solve = ocp.to_function('prob_solve', [w0, constraint],
                                 [states, controls, torque, constraint])

    for i in range(total_points):
        sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
        sys.stdout.flush()

        #data = full_data[:, i:i + horizon_length]  # Horizon torque data
        data = np.hstack([full_data[:, i:i + 1]] * horizon)
        w_sol, alpha_sol, torque_sol, _ = prob_solve(w_current, data)
        w_current = w_sol[:, 1]
        ocp.set_value(w0, w_current)
        # ocp.set_initial(w, w_current)  # Does nothing

        w_sol = np.array(w_sol)
        torque_sol = np.array(torque_sol)

        actual_w[:, i] = w_sol[:, 0]  # Wrong?
        all_w[:, :, i] = w_sol
        all_alpha[i] = alpha_sol[0]
        all_T_rw[:, i] = torque_sol[:, 0]

    return all_w, actual_w, all_alpha, all_T_rw


def solve_interval(obj_expr, total_points=8000, horizon=1, interval = 1, full_data=None, w_start = OMEGA_START):
    if full_data is None:
        full_data = load_data('Data/Slew1.mat')

    all_w = np.zeros((4, horizon + 1, total_points))  # (4, H+1, N)
    actual_w = np.zeros((4, total_points))  # (4, N)
    all_alpha = np.zeros(total_points)  # (N)
    all_T_rw = np.zeros((4, total_points))  # (4, N)
    w_sol = None
    w_current, w_initial = w_start, w_start
    alpha_buffer = np.zeros(interval)  # Buffer to hold 'update_interval' control actions (U)
    alpha_buffer_next = np.zeros(interval)  # (U)

    ocp = Ocp(t0=0, T=horizon / 10)
    w = ocp.state(4)
    alpha = ocp.control()
    T_sc = ocp.parameter(3, grid='control')

    w0 = ocp.parameter(4)  # Initial state/guess parameter
    ocp.set_value(w0, w_initial)

    data = full_data[:, :horizon]
    data = full_data[:, 0].reshape(3, 1).repeat(horizon, axis=1)
    ocp.set_value(T_sc, data)

    T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha  # Model/Dynamics
    der_state = I_INV @ T_rw
    ocp.set_der(w, der_state)

    ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))  # Constraints
    ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))

    ocp.subject_to(ocp.at_t0(w) == w0)
    ocp.set_initial(w, w0)

    a = 0.01
    objective_expr_casadi = np.exp(-a * w ** 2)
    objective = ocp.integral(sum1(obj_expr))
    ocp.add_objective(objective)

    ocp.solver('ipopt', SOLVER_OPTS)
    ocp.method(MultipleShooting(N=horizon, M=1, intg='rk'))

    constraint = ocp.sample(T_sc, grid='control-')[1]
    states = ocp.sample(w, grid='control')[1]
    controls = ocp.sample(alpha, grid='control-')[1]
    torque = ocp.sample(T_rw, grid='control')[1]

    prob_solve = ocp.to_function('prob_solve', [w0, constraint],
                                 [states, controls, torque, constraint])

    # MPC loop
    for i in range(total_points):
        sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
        sys.stdout.flush()

        if i % interval == 0:  # Solve MPC every 3 steps
            alpha_buffer = alpha_buffer_next
            data = np.hstack([full_data[:, i:i + 1]] * horizon)
            w_sol, alpha_sol, torque_sol, _ = prob_solve(w_current, data)  # (4, H+1) (1
            alpha_buffer_next[:] = alpha_sol[:interval]
            ocp.set_value(w0, w_current)
            # Store first 3 control actions


        j = i % interval
        data_live = full_data[:, i]
        alpha = alpha_buffer[j].reshape(1)
        torque = R_PSEUDO @ data_live + NULL_R @ alpha
        w_current = w_current + I_INV @ torque.reshape(-1, 1) * 0.1

        all_w[:, :, i] = w_sol  # check
        actual_w[:, i] = w_current.reshape(-1)  # Store actual trajectory
        all_alpha[i] = alpha_buffer[j]  # Apply buffered control action
        all_T_rw[:, i] = torque  # Store first torque action

    return all_w, actual_w, all_alpha, all_T_rw

