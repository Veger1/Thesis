from rockit import Ocp, MultipleShooting
from casadi import sum1
from config import *
from helper import *
import casadi as ca

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

    def oneshot_casadi(self, n0=0, N=1000, omega_initial=None, omega_final=None):
        # Problem dimensions and parameters
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
            obj += ca.sumsqr(w_k) * dt
            # obj += 10000 * ca.sumsqr(ca.exp(-0.01 * w_k**2)) * dt # Example: exponential decay

            # Optional: torque constraints
            g.append(T_rw_k)
            lbg.append(-MAX_TORQUE * np.ones(4))
            ubg.append(MAX_TORQUE * np.ones(4))

            # Optional: alpha_null constraint and saturation
            if self.alpha_limits is not None:
                alpha_null = (-w_k[0] + w_k[1] - w_k[2] + w_k[3]) / 4
                alpha_min, alpha_max = self.alpha_limits
                g.append(alpha_null)
                lbg.append(alpha_min[n0 + k] if not np.isneginf(alpha_min[n0 + k]) else -1e6)
                ubg.append(alpha_max[n0 + k] if not np.isposinf(alpha_max[n0 + k]) else 1e6)

        # Initial condition
        w0 = self.w_initial if omega_initial is None else omega_initial
        g.append(w_vars[0] - w0)
        lbg.append(np.zeros(4))
        ubg.append(np.zeros(4))

        # Optional: omega saturation
        if self.omega_limits is not None:
            omega_min, omega_max = self.omega_limits
            print('Omega limits provided:')
            for k in range(N + 1):
                g.append(w_vars[k])
                lbg.append(omega_min[:, n0 + k])
                ubg.append(omega_max[:, n0 + k])
        else:
            print('No omega limits provided:')
            for k in range(N + 1):
                g.append(w_vars[k])
                lbg.append(-OMEGA_MAX * np.ones(4))
                ubg.append(OMEGA_MAX * np.ones(4))

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

        for k in range(N + 1):
            lbx.extend(-OMEGA_MAX * np.ones(4))
            ubx.extend(OMEGA_MAX * np.ones(4))

        for k in range(N):
            lbx.extend(-1e6 * np.ones(NULL_R.shape[1]))
            ubx.extend(1e6 * np.ones(NULL_R.shape[1]))

        lbx = ca.vertcat(*lbx)
        ubx = ca.vertcat(*ubx)


        # Initial guess
        w0_guess = []
        alpha0_guess = []

        # Build omega guess
        if self.omega_guess is not None:
            print("Omega guess provided")
            w_guess_segment = self.omega_guess[:, n0:n0 + N + 1]  # shape (4, N+1)
            w0_guess = w_guess_segment.T.reshape((-1,))  # flatten in correct order
        else:
            w0_guess = np.zeros((4 * (N + 1),))  # fallback

        # Build alpha guess
        if self.control_guess is not None:
            print("Control guess provided")
            alpha_guess_segment = self.control_guess[n0:n0 + N]  # shape (N, n)
            alpha0_guess = alpha_guess_segment.reshape((-1,))
        else:
            alpha0_guess = np.zeros((NULL_R.shape[1] * N,))  # fallback

        # Combine both into x0
        x0 = ca.DM(np.concatenate([w0_guess, alpha0_guess]))

        # Solve
        sol = solver(x0=x0, lbx=lbx, ubx=ubx,
                     lbg=ca.vertcat(*lbg), ubg=ca.vertcat(*ubg),
                     p=ca.vec(self.torque_data[:, n0:n0 + N]))

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

        return w_sol.T, alpha_sol.T, T_rw_sol

    def sequential_bis(self, begin_ids, end_ids, omega_guess=None, alpha_guess=None, pseudo_sol=None):
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

    def sequential(self, split_indices):
        full_w_sol = []
        full_alpha_sol = []
        full_T_rw_sol = []

        # First index
        start_idx = 0

        for end_idx in split_indices:  # Ensure the final segment goes until 8004
            N = end_idx - start_idx
            print(f"Solving from {start_idx} to {end_idx} (N={N})")

            # Set initial omega
            omega_init = self.omega_guess[:, start_idx] if self.omega_guess is not None else None

            # Set desired final omega (only if omega_guess is provided)
            omega_end = self.omega_guess[:, end_idx] if self.omega_guess is not None else None
            print(f"Segment {start_idx}->{end_idx}: omega_init={omega_init}, omega_end={omega_end}")
            # omega_null = nullspace_alpha(omega_init)
            # print(f"Nullspace of omega_init: {omega_null}")
            # alpha_null = nullspace_alpha(omega_end)
            # print(f"Nullspace of omega_end: {alpha_null}")

            # Solve the subproblem
            w_sol, alpha_sol, T_rw_sol = self.oneshot_casadi(
                n0=start_idx, N=N,
                omega_initial=omega_init,
                omega_final=omega_end
            )

            # Stitch results
            if len(full_w_sol) > 0:
                # Skip the first column of the new w_sol to avoid double point at boundary
                full_w_sol.append(w_sol[:, 1:])
            else:
                full_w_sol.append(w_sol)

            full_alpha_sol.append(alpha_sol)
            full_T_rw_sol.append(T_rw_sol)

            # Move to the next segment
            start_idx = end_idx

        # Concatenate all segments
        final_w = np.concatenate(full_w_sol, axis=1)  # shape (4, 8005)
        final_alpha = np.concatenate(full_alpha_sol, axis=1)  # shape (null_dim, 8004)
        final_T_rw = np.concatenate(full_T_rw_sol, axis=1)  # shape (4, 8004)

        return final_w, final_alpha, final_T_rw

