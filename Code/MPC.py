import numpy as np
from casadi import sum1
from rockit import Ocp, MultipleShooting
import sys
from config import *
from helper import convert_expr


def solve(obj_expr, total_points=8000, horizon=1, full_data=None, w_start = OMEGA_START):
    if full_data is None:
        full_data = load_data('Data/Slew1.mat')

    all_w = np.zeros((4, horizon + 1, total_points))  # Store predictions over time (4, H+1, N)
    actual_w = np.zeros((4, total_points))  # Store actual past trajectory (4, N)
    all_alpha = np.zeros(total_points)  # Store all control actions (N)
    all_T_rw = np.zeros((4, total_points))  # Store all reaction wheel torques (4, N)
    w_sol = None
    w_current, w_initial = w_start, w_start

    for i in range(total_points):
        sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
        sys.stdout.flush()

        ocp = Ocp(t0=i/10, T=horizon / 10)
        w = ocp.state(4)
        alpha = ocp.control()
        T_sc = ocp.parameter(3, grid='control')

        data = full_data[:, i:i + horizon]
        ocp.set_value(T_sc, data)

        # Dynamics
        T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha
        der_state = I_INV @ T_rw
        ocp.set_der(w, der_state)

        ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))
        ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))

        # Initial condition for the current horizon
        ocp.subject_to(ocp.at_t0(w) == w_current)
        if i != 0:
            ocp.set_initial(w, w_sol.T)

        obj_expr_casadi = convert_expr(ocp, obj_expr, w)  # Convert to CasADi expression
        objective = ocp.integral(sum1(obj_expr_casadi))
        ocp.add_objective(objective)

        # Solve
        ocp.solver('ipopt', SOLVER_OPTS)
        ocp.method(MultipleShooting(N=horizon, M=1, intg='rk'))
        sol = ocp.solve()

        # Extract solutions
        ts, w_sol = sol.sample(w, grid='control')  # (N,) (H+1, N)
        _, alpha_sol = sol.sample(alpha, grid='control')  # (N,)
        _, T_rw_sol = sol.sample(T_rw, grid='control')  # (H+1, N)

        # Store results
        w_current = w_sol[1, :]
        actual_w[:, i] = w_sol[0, :]  # Wrong?
        all_w[:, :, i] = w_sol.T  # (4, H+1)
        all_alpha[i] = alpha_sol[0]
        all_T_rw[:, i] = T_rw_sol[0, :]

    return all_w, actual_w, all_alpha, all_T_rw


def solve_interval(obj_expr, total_points=8000, horizon=1, interval = 1, full_data=None, w_start = OMEGA_START):
    if full_data is None:
        full_data = load_data('Data/Slew1.mat')

    all_w = np.zeros((4, horizon + 1, total_points))  # (4, H+1, N)
    actual_w = np.zeros((4, total_points))  #  (4, N)
    all_alpha = np.zeros(total_points)  # (N)
    all_T_rw = np.zeros((4, total_points))  #  (4, N)
    w_sol = None
    w_current, w_initial = w_start, w_start

    alpha_buffer = np.zeros(interval)  # Buffer to hold control actions
    alpha_buffer_next = np.zeros(interval)

    for i in range(total_points):
        sys.stdout.write(f"\rSolving MPC for time step {i + 1}/{total_points}")
        sys.stdout.flush()
        section = i // 2000
        time0 = i/10 - section * 200
        if i % interval == 0:
            alpha_buffer = alpha_buffer_next
            ocp = Ocp(t0=time0, T=horizon / 10)
            w = ocp.state(4)
            alpha = ocp.control()
            T_sc = ocp.parameter(3, grid='control')

            # Set input torques
            data = np.hstack([full_data[:, i:i + 1]] * horizon)
            ocp.set_value(T_sc, data)

            # Dynamics
            T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha
            der_state = I_INV @ T_rw
            ocp.set_der(w, der_state)

            ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))
            ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))
            ocp.subject_to(ocp.at_t0(w) == w_current)

            # Objective function
            obj_expr_casadi = convert_expr(ocp, obj_expr, w)  # Convert to CasADi expression
            objective = ocp.integral(sum1(obj_expr_casadi))
            ocp.add_objective(objective)

            # Solve
            ocp.solver('ipopt', SOLVER_OPTS)
            ocp.method(MultipleShooting(N=horizon, M=1, intg='rk'))
            sol = ocp.solve()

            # Extract solutions
            ts, w_sol = sol.sample(w, grid='control')
            _, alpha_sol = sol.sample(alpha, grid='control')
            _, T_rw_sol = sol.sample(T_rw, grid='control')

            alpha_buffer_next[:] = alpha_sol[:interval]
            all_w[:, :, i] = w_sol.T  # Store prediction

        # Apply buffered control actions
        j = i % interval
        data_live = full_data[:, i]
        alpha = alpha_buffer[j].reshape(1)
        torque = R_PSEUDO @ data_live + NULL_R @ alpha
        w_current = w_current + I_INV @ torque.reshape(-1, 1) * 0.1

        actual_w[:, i] = w_current.reshape(-1)
        all_alpha[i] = alpha_buffer[j]
        all_T_rw[:, i] = torque

    return all_w, actual_w, all_alpha, all_T_rw