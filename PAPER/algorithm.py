from scipy.io import loadmat
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

OMEGA_MAX = RPM_MAX * 2 * np.pi / 60
OMEGA_MIN = RPM_MIN * 2 * np.pi / 60
OMEGA_START = np.random.uniform(-300, 300, (4, 1))

MAX_TORQUE = 2.5 * 10**-3  # Torque maximum in Nm

data = loadmat("Slew1.mat")["Test"].transpose()

OMEGA_START_PSEUDO = R_PSEUDO @ R @ OMEGA_START
OMEGA_START_NULL = OMEGA_START - OMEGA_START_PSEUDO

def forward_integration(w_start, torque_3d, dt):
    torque_4d = R_PSEUDO @ torque_3d  # (4, N)
    dw = I_INV @ torque_4d * dt  # (4, N)
    w = np.hstack([w_start, w_start + np.cumsum(dw, axis=1)])  # (4, N+1)
    return w

def make_pair_overlap_masks(w_data:np.ndarray):
    n_bands = w_data.shape[0]
    radii = OMEGA_MIN / np.abs(NULL_R).reshape(-1)  # (4, )  OMEGA_MIN could also be 4,N
    center = -w_data / NULL_R  # (4, N)
    i_idx, j_idx = np.triu_indices(n_bands, k=1)
    _pairs = np.stack((i_idx, j_idx), axis=1)  # (n_pairs, 2)

    # Extract centers and radii for each pair
    ci = center[i_idx]  # (n_pairs, N)
    cj = center[j_idx]  # (n_pairs, N)

    ri = radii[i_idx]  # (n_pairs,)
    rj = radii[j_idx]  # (n_pairs,)

    # Overlap condition
    _overlap_mask = np.abs(ci - cj) < (ri + rj)[:, None]

    return _overlap_mask, _pairs

def make_direct_overlap_masks(w_data:np.ndarray):
    center = -w_data / NULL_R  # (4, N)
    order = np.argsort(center, axis=0)  # (4, N)

    center_sorted = np.take_along_axis(center, order, axis=0)  # (4, N)
    radii_sorted = np.take_along_axis(
        (OMEGA_MIN / np.abs(NULL_R)),
        order,
        axis=0
    )  # (4, N)

    dist = np.abs(center_sorted[1:] - center_sorted[:-1])  # (3, N)
    r_sum = radii_sorted[1:] + radii_sorted[:-1]  # (3, N)

    overlap_sorted = dist < r_sum  # (3, N)

    band_pairs_sorted = np.stack(
        [order[:-1], order[1:]],
        axis=1
    )  # (3, 2, N) Recover which original bands formed each intersection

    return overlap_sorted, band_pairs_sorted, order

def overlap_or_intersect(overlap_sorted:np.ndarray, order:np.ndarray):
    overlap_prev = overlap_sorted[:, :-1]
    overlap_next = overlap_sorted[:, 1:]

    enter = (~overlap_prev) & overlap_next  # (3, N-1)
    exit = overlap_prev & (~overlap_next)  # (3, N-1)
    enter = np.hstack([overlap_sorted[:, :1], enter])
    exit = np.hstack([exit, overlap_sorted[:, -1:]])

    interval_id = np.cumsum(enter, axis=1) * overlap_sorted
    rel_order = np.sign(order[1:] - order[:-1])  # (3, N)

    intersection_flag = []

    for k in range(3):  # only 3 rows — negligible
        ids = interval_id[k]
        starts = enter[k]
        ends = exit[k]

        s = rel_order[k][starts]
        e = rel_order[k][ends]

        intersection_flag.append(s != e)

    intersection_flag = np.array(intersection_flag, dtype=bool)

    intersection_mask = np.zeros_like(overlap_sorted, dtype=bool)
    for k in range(3):
        ids = interval_id[k]
        valid = ids > 0
        intersection_mask[k, valid] = intersection_flag[k][ids[valid] - 1]

    overlap_only_mask = overlap_sorted & ~intersection_mask

    return overlap_only_mask, intersection_mask



def intersection_from_order(order):
    # order: (4, N)
    # returns (3, N) for adjacent pairs
    return np.sign(order[1:] - order[:-1])


w_pseudo = forward_integration(OMEGA_START_PSEUDO, data, dt=0.1)
overlap_mask_sorted, band_pairs_mask_sorted, band_order = make_direct_overlap_masks(w_pseudo)
overlap_or_intersect(overlap_mask_sorted, band_order)
print()