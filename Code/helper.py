"""
This file contains helper functions that are used in the main code.
"""
import numpy as np


def rpm_to_rad(rpm):
    return rpm * 2 * np.pi / 60


def rad_to_rpm(rad):
    return rad * 60 / (2 * np.pi)

