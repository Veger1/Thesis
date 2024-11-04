"""
This file contains helper functions that are used in the main code.
"""
import numpy as np

class Helper:
    def __init__(self):
        pass

    def rpm_to_rad(self, rpm):
        return rpm * 2 * np.pi / 60

    def rad_to_rpm(self, rad):
        return rad * 60 / (2 * np.pi)

