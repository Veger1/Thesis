import MPC
import MPC_fast
import numpy as np
from sympy import symbols, exp, Abs, tanh, atan, pi


w, t = symbols('w t')
a = 0.01
b = 1 / 700000
offset = 60
k = 1.0
# time_dependent = (1 + tanh(k * (t - offset))) / 2
objective = exp(-a * w ** 2) #+ time_dependent*b * w ** 2

_, w_sol1, _, _ = MPC.solve_interval(objective, total_points = 1000)
_, w_sol2, _, _ = MPC_fast.solve_interval(objective, total_points = 1000)

_, w_sol3, _, _ = MPC.solve(objective, total_points = 1000)
_, w_sol4, _, _ = MPC_fast.solve(objective, total_points = 1000)
