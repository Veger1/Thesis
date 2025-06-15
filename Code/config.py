import scipy.io
import numpy as np
from scipy.linalg import null_space

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
RPM_MIN = 100
RPM_TGT = 1000

OMEGA_MAX = RPM_MAX * 2 * np.pi / 60
OMEGA_MIN = RPM_MIN * 2 * np.pi / 60
OMEGA_TGT = RPM_TGT * 2 * np.pi / 60

seed = np.random.randint(0, 1000)
# np.random.seed(56)
# np.random.seed(seed)
seed = 1  # 1  /3
np.random.seed(seed)
OMEGA_START = np.random.uniform(-300, 300, (4, 1))
# OMEGA_START = np.zeros((4, 1))

MAX_TORQUE = 2.5 * 10**-3  # Torque maximum in Nm
LOW_TORQUE_THRESHOLD = 0.000005  # 5 e-6
HIGH_TORQUE_THRESHOLD = 0.000015 # 15 e-6

SOLVER_OPTS = {
        "print_time": False,  # Suppress overall solver timing
        "ipopt": {
           # "print_level": 0,  # Disable IPOPT output
            "sb": "yes"  # Suppress banner output
        }
    }

def load_data(path):
    data_pack = scipy.io.loadmat(path)
    data = data_pack['Test']
    return data.T



