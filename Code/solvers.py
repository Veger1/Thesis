from rockit import Ocp, MultipleShooting
from casadi import sum1
from config import *
from helper import *

class Solver:
    def __init__(self, data, alpha_limits=None, omega_limits=None, omega_guess=None, control_guess=None):
        self.torque_data = data
        self.alpha_null_sol = None
        self.w_sol = None
        self.alpha_sol = None
        self.T_rw_sol = None
        self.alpha_limits = alpha_limits
        self.omega_limits = omega_limits
        self.omega_guess = omega_guess
        self.control_guess = control_guess
        self.nullspace_solution = None
        self.N = self.torque_data.shape[1]
        self.w_initial = OMEGA_START
        self.temp_time = np.linspace(0, 800, 8005)


    def oneshot(self, n0=0, N=1000, omega_initial=None):
        ocp = Ocp(t0=self.temp_time[n0], T=self.temp_time[N]-self.temp_time[n0])
        w = ocp.state(4)
        alpha_control = ocp.control()
        T_sc = ocp.parameter(3, grid='control')
        ocp.set_value(T_sc, self.torque_data[:, n0:n0+N])

        T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha_control
        der_state = I_INV @ T_rw
        ocp.set_der(w, der_state)

        if self.omega_guess is not None:
            print('Omega guess provided:')
            ocp.set_initial(w, self.omega_guess)
        if self.control_guess is not None:
            print('Control guess provided:')
            ocp.set_initial(alpha_control, self.control_guess[n0:n0+N])

        if self.alpha_limits is not None:
            min_constraint, max_constraint = self.alpha_limits
            min_constraint = np.where(np.isneginf(min_constraint), -1e6, min_constraint)
            max_constraint = np.where(np.isposinf(max_constraint), 1e6, max_constraint)
            alpha_null = ocp.variable(grid='control')
            ocp.subject_to(alpha_null == (-w[0] + w[1] - w[2] + w[3]) / 4)
            alpha_min_constraint = ocp.parameter(grid='control')
            alpha_max_constraint = ocp.parameter(grid='control')
            ocp.set_value(alpha_min_constraint, min_constraint[n0:n0+N])
            ocp.set_value(alpha_max_constraint, max_constraint[n0:n0+N])
            ocp.subject_to(alpha_min_constraint <= (alpha_null <= alpha_max_constraint))  # Add saturation constraints

        if self.omega_limits is not None:
            min_constraint, max_constraint = self.omega_limits
            ocp.subject_to(min_constraint[:,n0:n0+N] <= (w <= max_constraint[:, n0:n0+N]))  # Add saturation constraints
        else:
            ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))  # Remove later, can be inserted into segments

        ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))  # Add torque constraints
        if omega_initial is not None:
            ocp.subject_to(ocp.at_t0(w) == omega_initial)
        else:
            ocp.subject_to(ocp.at_t0(w) == self.w_initial)
        ocp.set_initial(w, self.w_initial)  # Set initial guess

        a = 0.1
        b = 1e-4
        objective_expr_casadi = np.exp(-a * w ** 2)
        objective_expr_casadi = b * w ** 2
        objective = ocp.integral(sum1(objective_expr_casadi))
        ocp.add_objective(objective)

        ocp.solver('ipopt', SOLVER_OPTS)  # Use IPOPT solver
        ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
        sol = ocp.solve()  # Solve the problem

        ts, w_sol = sol.sample(w, grid='control')
        _, alpha_sol = sol.sample(alpha_control, grid='control')
        _, T_rw_sol = sol.sample(T_rw, grid='control')

        return w_sol, alpha_sol, T_rw_sol

    def sequential(self, begin_ids, end_ids, omega_guess=None, alpha_guess=None, pseudo_sol=None):
        self.w_sol = np.zeros((4, self.N+1))
        self.w_sol[:, 0] = OMEGA_START.flatten()
        for i in range(2):
            try:
                n0 = begin_ids[i]
                n1 = end_ids[i]
                N = n1 - n0
                n2 = begin_ids[i+1]
                print(f"Segment {i+1}: n0 = {n0}, n1 = {n1}, N = {N}")
                # initial = pseudo_sol[:, n0:n0+1] + NULL_R * alpha_guess[n0]
                if i == 0:
                    initial =OMEGA_START
                else:
                    initial = omega_guess[:, n0:n0+1]
                # w_sol, alpha_sol, T_rw_sol = self.oneshot(n0=n0, N=N, omega_initial=initial)
                w_sol, alpha_sol, T_rw_sol = self.oneshot(n0=n0, N=N, omega_initial=self.w_sol[:, n0:n0+1])
                self.w_sol[:, n0:n1+1] = w_sol.T
                if omega_guess is not None:
                    self.w_sol[:, n1:n2+1] = omega_guess[:, n1:n2+1]
            except RuntimeError:
                print(f"Solver failed for segment {i+1}.")
        return self.w_sol
