import numpy as np
from rockit import Ocp, MultipleShooting
from casadi import sum1
from config import *
from helper import *
import casadi as ca

class Solver:
    def __init__(self, data, omega_start, alpha_limits=None, omega_limits=None, omega_guess=None, control_guess=None,
                 omega_selective_limits=None, reduce_torque_limits=False):
        self.torque_data = data
        self.N = self.torque_data.shape[1]
        self.null_sol = np.zeros((self.N+1,))
        self.w_sol = np.zeros((4, self.N + 1))
        self.alpha_sol = np.zeros((self.N,))
        self.T_rw_sol = np.zeros((4, self.N))
        self.alpha_limits = alpha_limits
        self.omega_limits = omega_limits
        self.omega_selective_limits = omega_selective_limits
        self.omega_guess = omega_guess
        self.control_guess = control_guess
        self.reduce_torque_limits = reduce_torque_limits
        self.w_initial = omega_start
        self.temp_time = np.linspace(0, 800, 8005)
        self.setup_time = None
        self.solve_time = None
        self.extrac_time = None
        self.iteration_count = None

    def oneshot_rockit(self, n0=0, N=1000, omega_initial=None):
        ocp = Ocp(t0=self.temp_time[n0], T= self.temp_time[N])
        w = ocp.state(4)
        alpha = ocp.control()
        T_sc = ocp.parameter(3, grid='control')

        torque_data = self.torque_data[:, n0:n0 + N]
        ocp.set_value(T_sc, torque_data)  # Assign the torque constraint

        T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha
        der_state = I_INV @ T_rw
        ocp.set_der(w, der_state)

        ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))  # Add torque constraints
        ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))  # Add saturation constraints
        if omega_initial is None:
            ocp.subject_to(ocp.at_t0(w) == self.w_initial)
            print('First:',self.w_initial)
        else:
            print('Second:', omega_initial)
            ocp.subject_to(ocp.at_t0(w) == omega_initial.reshape((4,1)))
        # ocp.set_initial(w, self.w_initial)  # Set initial guess

        objective_expr_casadi = np.exp(-0.05 * w ** 2)
        objective = ocp.integral(sum1(objective_expr_casadi))
        ocp.add_objective(objective)

        objective_expr_casadi = w ** 2 / 100000
        objective = ocp.integral(sum1(objective_expr_casadi))
        ocp.add_objective(objective)


        ocp.solver('ipopt', SOLVER_OPTS)  # Use IPOPT solver
        ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
        sol = ocp.solve()  # Solve the problem

        # Post-processing: Sample solutions for this interval
        ts, w_sol = sol.sample(w, grid='control')
        _, alpha_sol = sol.sample(alpha, grid='control')
        _, T_rw_sol = sol.sample(T_rw, grid='control')

        self.w_sol[:, n0:n0 + N + 1] = w_sol.T
        self.alpha_sol[n0:n0 + N] = alpha_sol.T[:N]
        self.T_rw_sol[:, n0:n0 + N] = T_rw_sol.T[:,:N]
        self.null_sol[n0:n0 + N + 1] = nullspace_alpha(self.w_sol[:, n0:n0 + N + 1])

    @profile
    def oneshot_casadi(self, n0=0, N=1000, omega_initial=None, omega_final=None, torque_on_g=False, omega_on_g=False, penalise_stiction=False):
        start_time = clock.time()
        dt = self.temp_time[n0 + 1] - self.temp_time[n0]
        T = dt * N

        # Symbolic variables
        w_vars = [ca.MX.sym(f'w_{k}', 4) for k in range(N + 1)]
        alpha_vars = [ca.MX.sym(f'alpha_{k}', NULL_R.shape[1]) for k in range(N)]

        # Symbolic parameter for external torque
        T_sc = ca.MX.sym('T_sc', 3, N)
        # Accumulate constraints and objective
        g = []
        lbg = []
        ubg = []
        obj = 0

        if penalise_stiction:
            def penalty_stiction(w_k, k):
                # Example penalty function for stiction
                return ca.sumsqr(ca.exp(-0.02 * w_k**2)) * dt
        else :
            def penalty_stiction(w_k, k):
                return 0

        if torque_on_g:
            if self.reduce_torque_limits:
                alpha_lower, alpha_upper = calc_alpha_torque_limits(self.torque_data, n0, N)
                print("Selective torque limits provided on g:")
                def torque_constraint(alpha_k, T_rw_k, k):
                    g.append(alpha_k)
                    lbg.append(alpha_lower[k])
                    ubg.append(alpha_upper[k])
            else:
                print("Torque limits provided on g:")
                def torque_constraint(alpha_k, T_rw_k, k):
                    g.append(T_rw_k)
                    lbg.append(-MAX_TORQUE * np.ones(4))
                    ubg.append(MAX_TORQUE * np.ones(4))
        else:
            def torque_constraint(alpha_k, T_rw_k, k):
                pass

        # Dynamics integration (Euler or RK — here we use simple Euler)
        for k in range(N):
            w_k = w_vars[k]
            w_kp1 = w_vars[k + 1]
            alpha_k = alpha_vars[k]
            T_rw_k = R_PSEUDO @ T_sc[:, k] + NULL_R @ alpha_k
            dw_k = I_INV @ T_rw_k

            # Euler integration
            w_next = w_k + dt * dw_k
            g.append(w_kp1 - w_next)
            lbg.append(np.zeros(4))
            ubg.append(np.zeros(4))

            # Objective (example: quadratic on angular velocity)
            obj += ca.sumsqr(w_k) * dt /100000 # Quadratic on angular velocity
            # obj += ca.sum1(ca.fabs(w_k)) * dt  # Linear on angular velocity
            # obj += ca.sumsqr(ca.exp(-0.02 * w_k**2)) * dt  #  Zero crossing
            obj += penalty_stiction(w_k, k)  # Stiction penalty
            # obj += ca.sum1(ca.exp(-0.01 * w_k**2)) * dt  #  Zero crossing
            # obj += 10000000*ca.sumsqr(w_k * T_rw_k) * dt  # Power
            #  Linear plus power is potentially a good idea for high activity phase

            # Optional: torque constraints
            torque_constraint(alpha_k, T_rw_k, k)

        # Optional: omega limits
        if omega_on_g:
            if self.omega_limits is not None:
                omega_min, omega_max = self.omega_limits
                print('Omega limits provided on g:')
                for k in range(N + 1):
                    g.append(w_vars[k])
                    lbg.append(omega_min[:, n0 + k])
                    ubg.append(omega_max[:, n0 + k])
            elif self.omega_selective_limits:
                print('Selective omega limits provided on g:')
                omega_min, omega_max = self.omega_selective_limits
                mask_lower = np.isclose(np.abs(omega_min), np.abs(OMEGA_MIN), atol=1e-4)
                mask_upper = np.isclose(np.abs(omega_max), np.abs(OMEGA_MIN), atol=1e-4)
                for k in range(N + 1):
                    for i in range(4):
                        has_lower = mask_lower[i, n0 + k]
                        has_upper = mask_upper[i, n0 + k]
                        if has_lower and has_upper:  # Maybe skip, maybe has never both?
                            g.append(w_vars[k][i])
                            lbg.append(omega_min[i, n0 + k])
                            ubg.append(omega_max[i, n0 + k])
                        elif has_lower:
                            g.append(w_vars[k][i])
                            lbg.append(omega_min[i, n0 + k])
                            ubg.append(ca.inf)
                        elif has_upper:
                            g.append(w_vars[k][i])
                            lbg.append(-ca.inf)
                            ubg.append(omega_max[i, n0 + k])
            else:
                print('Regulars omega limits provided on g:')
                for k in range(N + 1):
                    g.append(w_vars[k])
                    lbg.append(-OMEGA_MAX * np.ones(4))
                    ubg.append(OMEGA_MAX * np.ones(4))

        # Initial condition
        w0 = self.w_initial if omega_initial is None else omega_initial
        g.append(w_vars[0] - w0)
        lbg.append(np.zeros(4))
        ubg.append(np.zeros(4))

        if omega_final is not None:
            g.append(w_vars[-1] - omega_final)
            lbg.append(np.zeros(4))
            ubg.append(np.zeros(4))

        # Collect all variables
        x = ca.vertcat(*w_vars, *alpha_vars)
        g = ca.vertcat(*g)
        p = ca.vec(T_sc)

        # NLP setup
        nlp = {'x': x, 'f': obj, 'g': g, 'p': p}
        solver = ca.nlpsol('solver', 'ipopt', nlp, SOLVER_OPTS)

        # Initial guess and bounds
        lbx = []
        ubx = []

        if self.omega_limits is not None and not omega_on_g:
            omega_min, omega_max = self.omega_limits
            print('Omega limits provided on x:')
            for k in range(N + 1):
                lbx.append(omega_min[:, n0 + k])
                ubx.append(omega_max[:, n0 + k])
        elif self.omega_selective_limits is not None and not omega_on_g:
            print('Selective omega limits provided on x:')
            omega_min, omega_max = self.omega_selective_limits
            mask_lower = np.isclose(np.abs(omega_min), np.abs(OMEGA_MIN), atol=1e-4)
            mask_upper = np.isclose(np.abs(omega_max), np.abs(OMEGA_MIN), atol=1e-4)
            for k in range(N + 1):
                for i in range(4):
                    has_lower = mask_lower[i, n0 + k]
                    has_upper = mask_upper[i, n0 + k]
                    if has_lower and has_upper:  # Maybe skip, maybe has never both?
                        lbx.append(omega_min[i, n0 + k])
                        ubx.append(omega_max[i, n0 + k])
                    elif has_lower:
                        lbx.append(omega_min[i, n0 + k])
                        ubx.append(ca.inf)
                    elif has_upper:
                        lbx.append(-ca.inf)
                        ubx.append(omega_max[i, n0 + k])
                    else:
                        lbx.append(-ca.inf)
                        ubx.append(ca.inf)
        else:
            print('Regular omega limits provided on x:')
            for k in range(N + 1):
                lbx.append(-OMEGA_MAX * np.ones(4))
                ubx.append(OMEGA_MAX * np.ones(4))

        # Control limits
        if self.reduce_torque_limits and not torque_on_g:
            alpha_lower, alpha_upper = calc_alpha_torque_limits(self.torque_data, n0, N)
            print("Selective alpha limits provided on x:")
            for k in range(N):
                lbx.append(alpha_lower[k])
                ubx.append(alpha_upper[k])
        else:
            print('Regular alpha limits provided on x:')
            for k in range(N):
                lbx.extend(-1e6 * np.ones(NULL_R.shape[1]))
                ubx.extend(1e6 * np.ones(NULL_R.shape[1]))


        lbx = ca.vertcat(*lbx)
        ubx = ca.vertcat(*ubx)

        # Initial guess
        # Build omega guess
        if self.omega_guess is not None:
            print("Omega guess provided")
            w_guess_segment = self.omega_guess[:, n0:n0 + N + 1]  # shape (4, N+1)
            w0_guess = w_guess_segment.T.reshape((-1,))  # flatten in correct order
        else:
            w0_guess = np.zeros((4 * (N + 1),))  # fallback
            # best_guess = R_PSEUDO @ R @ self.w_initial
            # w0_guess_matrix = np.hstack([best_guess] * (N + 1))  # shape: (4, N+1)
            # w0_guess = w0_guess_matrix.T.reshape(-1)

        # Build alpha guess
        if self.control_guess is not None:
            print("Control guess provided")
            alpha_guess_segment = self.control_guess[n0:n0 + N]  # shape (N, n)
            alpha0_guess = alpha_guess_segment.reshape((-1,))
        else:
            alpha0_guess = np.zeros((NULL_R.shape[1] * N,))  # fallback

        # Combine both into x0
        x0 = ca.DM(np.concatenate([w0_guess, alpha0_guess]))

        mid1_time = clock.time()
        # Solve
        sol = solver(x0=x0, lbx=lbx, ubx=ubx,
                     lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg),
                     p=ca.vec(self.torque_data[:, n0:n0 + N]))
        mid2_time = clock.time()
        self.iteration_count = solver.stats()['iter_count']

        # Extract solution
        x_sol = sol['x'].full().flatten()
        w_sol = np.array([x_sol[i * 4:(i + 1) * 4] for i in range(N + 1)])
        alpha_start = (N + 1) * 4
        alpha_sol = np.array([
            x_sol[alpha_start + i * NULL_R.shape[1]: alpha_start + (i + 1) * NULL_R.shape[1]]
            for i in range(N)
        ])

        # Recompute T_rw
        T_rw_sol = R_PSEUDO @ self.torque_data[:, n0:n0 + N] + NULL_R @ alpha_sol.T
        end_time = clock.time()
        self.setup_time = mid1_time - start_time
        self.solve_time = mid2_time - mid1_time
        self.extrac_time = end_time - mid2_time

        self.w_sol[:,n0:n0+N+1] = w_sol.T
        print(alpha_sol.T.shape)
        print(self.alpha_sol.shape)
        self.alpha_sol[n0:n0+N] = alpha_sol.T
        self.T_rw_sol[:,n0:n0+N] = T_rw_sol
        self.null_sol[n0:n0+N+1] = nullspace_alpha(self.w_sol[:, n0:n0 + N + 1])

    def sequential_casadi_bis(self, torque_on_g=False, omega_on_g=False):
        start_idx = 0
        length = 1000
        for i in range(8):
            print(f"Segment {i+1}: start_idx = {start_idx}, length = {length}")
            if i ==0:
                self.oneshot_casadi(n0=start_idx, N=length, torque_on_g=torque_on_g, omega_on_g=omega_on_g)
            else :
                w_start = self.w_sol[:, start_idx:start_idx+1]
                self.oneshot_casadi(n0=start_idx, N=length, omega_initial=w_start, torque_on_g=torque_on_g, omega_on_g=omega_on_g)
            start_idx += length
            print('code reached')

    def sequential_rockit_bis(self):
        start_idx = 0
        length = 1000
        for i in range(8):
            if i ==0:
                self.oneshot_rockit(n0=start_idx, N=length)
            else :
                w_start = self.w_sol[:, start_idx:start_idx+1]
                self.oneshot_rockit(n0=start_idx, N=length, omega_initial=w_start)
            start_idx += length

    def sequential_casadi(self, split_indices, torque_on_g=False, omega_on_g=False):
        for i in range(len(split_indices) - 1):
            start_idx = split_indices[i]
            end_idx = split_indices[i + 1]
            torque_len = end_idx - start_idx
            omega_len = torque_len + 1
            print(f"Segment {i+1}: start_idx = {start_idx}, end_idx = {end_idx}, torque_len = {torque_len}")
            if i ==0:
                self.oneshot_casadi(n0=start_idx, N=torque_len, torque_on_g=torque_on_g, omega_on_g=omega_on_g)
            else :
                w_start = self.w_sol[:, start_idx:start_idx+1]
                self.oneshot_casadi(n0=start_idx, N=torque_len, omega_initial=w_start, torque_on_g=torque_on_g, omega_on_g=omega_on_g)


