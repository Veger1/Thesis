import numpy as np
from scipy.interpolate import interp1d

def time_stiction_accurate(omega, limit, dt=0.1):
    N = omega.shape[1]
    time = np.linspace(0, N * dt, N)  # Assuming 0.1s intervals
    stiction_time = np.zeros(4)

    omega_max = limit
    omega_min = -limit

    for j in range(4):
        for i in range(N - 1):
            if abs(omega[i, j]) < omega_max or abs(omega[i + 1, j]) < omega_max:
                if abs(omega[i, j]) < omega_max and abs(omega[i + 1, j]) < omega_max:
                    stiction_time[j] += dt
                else:
                    if abs(omega[i, j]) < omega_max:
                        if omega[i+1, j] > omega_max:
                            crossing_value = omega_max
                        else:
                            crossing_value = omega_min
                    else:
                        if omega[i, j] > omega_max:
                            crossing_value = omega_max
                        else:
                            crossing_value = omega_min

                    # ✅ Insert safe interpolation here ↓
                    w0, w1 = omega[i, j], omega[i + 1, j]
                    t0, t1 = time[i], time[i + 1]
                    eps = 1e-12

                    if min(w0, w1) - eps <= crossing_value <= max(w0, w1) + eps:
                        interpolator = interp1d([w0, w1], [t0, t1])
                        crossing_time = interpolator(crossing_value)
                    else:
                        crossing_time = (t0 + t1) / 2  # or skip or log warning

                    # interpolator = interp1d([omega[i, j], omega[i+1, j]], [time[i], time[i + 1]])
                    # crossing_time = interpolator(crossing_value)

                    if abs(omega[i, j]) < omega_max:
                        # omega[i] is inside, add time from time[i] to crossing
                        stiction_time[j] += crossing_time - time[i]
                    else:
                        # omega[i+1] is inside, add time from crossing to time[i+1]
                        stiction_time[j] += time[i+1] - crossing_time
    return stiction_time

def energy(omega, torque, dt=0.1):
    """
    Calculate the energy for each band in a vectorized manner.

    :param omega: numpy array of shape (4, N), angular velocities.
    :param torque: numpy array of shape (4, N), torques.
    :param dt: float, time step.
    :return: numpy array of shape (4,), energy value for each band.
    """
    # Element-wise power calculation
    power = omega * torque * dt

    # Only consider positive power contributions
    positive_power = np.maximum(power, 0)

    # Sum over the time axis (axis=1) to get energy for each band
    energy_result = np.sum(positive_power, axis=1)

    return energy_result

def count_zero_crossings(omega):
    zero_crossings = np.sum(np.diff(np.sign(omega), axis=1) != 0, axis=1)

    return zero_crossings

def omega_squared_sum(omega, limit):
    """
        Calculate the sum of squared omega values normalized by OMEGA_MIN for each band.

        :param omega: numpy array of shape (4, N), angular velocities.
        :param limit: float, minimum omega value for normalization.
        :return: numpy array of shape (4,), sum of squared normalized omega values for each band.
        """
    # Compute the squared normalized omega values
    omega_squared = (omega / limit) ** 2

    # Sum over the time axis (axis=1) to get the result for each band
    omega_sqrd_sum = np.sum(omega_squared, axis=1)

    return omega_sqrd_sum

def layers_used(layer_list):
    return len(layer_list)