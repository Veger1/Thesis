from casadi import sum1
from matplotlib import pyplot as plt
from rockit import Ocp, MultipleShooting
import numpy as np
from init_helper import load_data, initialize_constants

full_data = load_data()
helper, I_inv, R_pseudo, Null_R, Omega_max, w_initial, T_max = initialize_constants()
num_intervals, points = 4, 30

ocp = Ocp(t0=0, T=points/10)
w = ocp.state(4)
alpha = ocp.control()
T_sc = ocp.parameter(3, grid='control')

w0 = ocp.parameter(4)
ocp.set_value(w0, w_initial)
data = full_data[:, :points]
ocp.set_value(T_sc, data)

T_rw = R_pseudo @ T_sc + Null_R @ alpha
der_state = I_inv @ T_rw
ocp.set_der(w, der_state)

ocp.subject_to(-T_max <= (T_rw <= T_max))
ocp.subject_to(-Omega_max <= (w <= Omega_max))

ocp.subject_to(ocp.at_t0(w) == w0)
ocp.set_initial(w, w0) # Only work for FIRST interval, does not update for subsequent intervals

a = 0.001
objective_expr_casadi = np.exp(-a * w ** 2)
objective = ocp.integral(sum1(objective_expr_casadi))
ocp.add_objective(objective)

ocp.solver('ipopt')
ocp.method(MultipleShooting(N=points, M=1, intg='rk'))

constraint = ocp.sample(T_sc, grid='control-')[1]  # input/output

states = ocp.sample(w, grid='control')[1]  # output
controls = ocp.sample(alpha, grid='control-')[1]
torque = ocp.sample(T_rw, grid='control')[1]

prob_solve = ocp.to_function('prob_solve', [w0, constraint],
                             [states, controls, torque, constraint])

for i in range(num_intervals):
    data = full_data[:, i*points:(i+1)*points]
    w_sol, alpha_sol, torque_sol, _ = prob_solve(w_initial, data)
    w_initial = w_sol[:, -1]
    ocp.set_value(w0, w_initial)
    # ocp.set_initial(w, w_initial) # Does nothing if uncommented
    plt.plot(np.array(w_sol).transpose())
    plt.show()

