import numpy as np
import scipy.io
from scipy.linalg import null_space
from helper import Helper  # Assuming Helper is defined elsewhere


# Load and initialize data
def load_data():
    data = scipy.io.loadmat('Data/Slew1.mat')
    test_data = data['Test']
    test_data_T = test_data.T
    return test_data_T


# Initialize constants and matrices
def initialize_constants():
    helper = Helper()

    # Constants
    Irw = 0.00002392195
    I = np.diag([Irw]*4)
    I_inv = np.linalg.inv(I)
    beta = np.radians(60)

    # Matrices
    R_brw = np.array([[np.sin(beta), 0, -np.sin(beta), 0],
                      [0, np.sin(beta), 0, -np.sin(beta)],
                      [np.cos(beta), np.cos(beta), np.cos(beta), np.cos(beta)]])

    R_rwb_pseudo = np.linalg.pinv(R_brw)  # Pseudo-inverse
    Null_Rbrw = null_space(R_brw) * 2  # Null space
    Null_Rbrw_T = Null_Rbrw.T  # Transpose of the null space

    # Angular velocities in radians per second
    Omega_max = helper.rpm_to_rad(6000)
    Omega_min = helper.rpm_to_rad(300)
    Omega_tgt = helper.rpm_to_rad(1000)

    Omega_ref = Omega_tgt * Null_Rbrw  # Matrix operation similar to MATLAB
    Omega_start = Omega_ref / 2

    T_max = 2.5 * 10**-3  # Torque maximum in Nm

    return helper, I_inv, R_rwb_pseudo, Null_Rbrw, Omega_max, Omega_start, T_max

