import scipy.io
import numpy as np
from scipy.linalg import null_space
from helper import *

IRW = 0.00002392195
I = np.diag([IRW]*4)
I_INV = np.linalg.inv(I)
BETA = np.radians(60)

# Matrices
R = np.array([[np.sin(BETA), 0, -np.sin(BETA), 0], # (3, 4)
                  [0, np.sin(BETA), 0, -np.sin(BETA)],
                  [np.cos(BETA), np.cos(BETA), np.cos(BETA), np.cos(BETA)]])

R_PSEUDO = np.linalg.pinv(R)  # Pseudo-inverse (4, 3)
NULL_R = null_space(R) * 2  # Null space (4, 1)
NULL_R_T = NULL_R.T  # Transpose of the null space (1, 4)

RPM_MAX = 6000
RPM_MIN = 300
RPM_TGT = 1000

OMEGA_MAX = rpm_to_rad(RPM_MAX)
OMEGA_MIN = rpm_to_rad(RPM_MIN)
OMEGA_TGT = rpm_to_rad(RPM_TGT)

OMEGA_REF = OMEGA_TGT * NULL_R  # (4, 1)
OMEGA_START = OMEGA_REF / 2  # (4, 1)
OMEGA_RANDOM = np.random.uniform(-100, 100, (4, 1))


MAX_TORQUE = 2.5 * 10**-3  # Torque maximum in Nm
LOW_TORQUE_THRESHOLD = 0.000005  # 5 e-6
HIGH_TORQUE_THRESHOLD = 0.000015 # 15 e-6

SOLVER_OPTS = {
        "print_time": False,  # Suppress overall solver timing
        "ipopt": {
            "print_level": 0,  # Disable IPOPT output
            "sb": "yes"  # Suppress banner output
        }
    }

def load_data(path):
    data_pack = scipy.io.loadmat(path)
    data = data_pack['Test']
    return data.T

