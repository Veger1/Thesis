"""
This file contains helper functions that are used in the main code.
"""
import numpy as np
from sympy import symbols, lambdify



def rpm_to_rad(rpm):
    return rpm * 2 * np.pi / 60


def rad_to_rpm(rad):
    return rad * 60 / (2 * np.pi)

def convert_expr(ocp, objective_expr, w):
    ocp_t = ocp.t
    w_sym, t_sym = symbols('w t')
    objective_expr_casadi = lambdify((w_sym, t_sym), objective_expr, 'numpy')
    objective_expr_casadi = objective_expr_casadi(w, ocp_t)
    return objective_expr_casadi

