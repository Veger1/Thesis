import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from scipy.linalg import null_space
from heapq import heappush, heappop
from typing import List, Tuple
import time
import networkx as nx

NUM_WHEELS = 4
IRW = 0.00002392195
I = np.diag([IRW]*NUM_WHEELS)
I_INV = np.linalg.inv(I)
BETA = np.radians(60)

# Matrices
R = np.array([[np.sin(BETA), 0, -np.sin(BETA), 0], # (3, 4)
                  [0, np.sin(BETA), 0, -np.sin(BETA)],
                  [np.cos(BETA), np.cos(BETA), np.cos(BETA), np.cos(BETA)]])

R_PSEUDO = np.linalg.pinv(R)  # Pseudo-inverse (4, 3)
NULL_R = null_space(R)  # Null space (4, 1)
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
    return w, torque_4d

def make_direct_overlap_masks(w_data:np.ndarray):
    center = -w_data / NULL_R  # (4, N)
    order = np.argsort(center, axis=0)  # (4, N)
    radii = OMEGA_MIN / np.abs(NULL_R) # (4, 1)  # OMEGA_MIN could also be 4,N

    center_sorted = np.take_along_axis(center, order, axis=0)  # (4, N)
    radii_sorted = np.take_along_axis(
        radii,
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

    return overlap_sorted, band_pairs_sorted, order, center, radii

def generate_intervals(overlap_k, pair_k, center):
    """
    CAN BE ACCELERATED WITH VECTOR MATH, LOOK FOR BOUNDARIES WITH DIFFS
    overlap_k : (N,) boolean
    pair_k    : (2, N) band indices (i,j) per time
    center    : (4, N)

    Returns:
        free_intervals
        overlap_not_crossing
        overlap_crossing
    """

    free_intervals = []
    overlap_not_crossing = []
    overlap_crossing = []

    N = overlap_k.size
    start = 0
    current = overlap_k[0]

    for t in range(1, N):
        if overlap_k[t] != current:
            end = t - 1

            if current:  # overlap interval
                i, j = pair_k[:, start]

                s0 = np.sign(center[i, start] - center[j, start])
                s1 = np.sign(center[i, end]   - center[j, end])

                if s0 == s1:
                    overlap_not_crossing.append((start, end))
                else:
                    overlap_crossing.append((start, end))
            else:  # free interval
                free_intervals.append((start, end))

            start = t
            current = overlap_k[t]

    # Handle final interval
    end = N - 1
    if current:
        i, j = pair_k[:, start]
        s0 = np.sign(center[i, start] - center[j, start])
        s1 = np.sign(center[i, end]   - center[j, end])

        if s0 == s1:
            overlap_not_crossing.append((start, end))
        else:
            overlap_crossing.append((start, end))
    else:
        free_intervals.append((start, end))

    return free_intervals, overlap_not_crossing, overlap_crossing

def generate_all_intervals(overlap_mask, band_pairs, center):
    all_free_intervals = []
    all_overlap_not_crossing = []
    all_overlap_crossing = []

    n_pairs = overlap_mask.shape[0]

    for k in range(n_pairs):
        overlap_k = overlap_mask[k]
        pair_k = band_pairs[k]

        free_intervals, overlap_not_crossing, overlap_crossing = generate_intervals(
            overlap_k,
            pair_k,
            center
        )

        all_free_intervals.append(free_intervals)
        all_overlap_not_crossing.append(overlap_not_crossing)
        all_overlap_crossing.append(overlap_crossing)

    return all_free_intervals, all_overlap_not_crossing, all_overlap_crossing


def set_covering(intervals: List[Tuple[int, int]], existing_layers: List[int]) -> List[int]:
    """
    Determines the indices at which to place layers to cover all intervals.

    Args:
        intervals: A list of lists containing (start, end) intervals.
        existing_layers: A list of already determined layer indices.

    Returns:
        A list of indices where layers should be placed.
    """
    # Flatten the intervals into a single list with their group index

    # Sort intervals by start index for efficient processing
    intervals.sort(key=lambda x: x[0])

    # Convert existing layers to a set for fast lookup
    layer_set = set(existing_layers)

    # Function to check if an interval is covered
    def is_covered(start, end):
        return any(start <= layer <= end for layer in layer_set)

    # Place layers to cover all intervals
    for start, end in intervals:
        if not is_covered(start, end):
            # Place a layer at the midpoint of the uncovered interval
            new_layer = (start + end) // 2
            layer_set.add(new_layer)

    # Return the updated list of layer indices
    return sorted(layer_set)


def set_covering_greedy(intervals: List[Tuple[int, int]], existing_layers: List[int]) -> List[int]:
    """
    Optimized greedy strategy to minimize the number of layers needed to cover all intervals.

    Args:
        intervals: List of (start, end) intervals.
        existing_layers: List of already determined layer indices.

    Returns:
        A list of indices where layers should be placed.
    """
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    # Convert existing layers to a set for fast lookup
    layer_set = set(existing_layers)
    uncovered_intervals = set(intervals)

    # Priority queue to store intervals by their end time
    heap = []
    for start, end in intervals:
        heappush(heap, (end, start))

    while uncovered_intervals:
        # Select the interval with the earliest end time
        end, start = heappop(heap)

        # Check if the interval is already covered
        if (start, end) not in uncovered_intervals:
            continue

        # Place a layer at the end of the interval to maximize coverage
        new_layer = end
        layer_set.add(new_layer)

        # Remove all intervals covered by the new layer
        uncovered_intervals = {
            (s, e) for s, e in uncovered_intervals if not (s <= new_layer <= e)
        }

    return sorted(layer_set)

def flatten_list_of_lists(list_of_lists):
    return [item for sublist in list_of_lists for item in sublist]


def determine_nodes(omega_pseudo: np.ndarray, centers: np.ndarray, radii: np.ndarray, layers: List[int]):
    """
    Determine nodes for each layer based on allowed regions.

    Args:
        centers: (4, N) array of centers for each band at each time step.
        radii: (4, N) array of radii for each band.
        layers: List of indices where layers are placed.

    Returns:
        List of nodes (alpha_k values) for each layer.
    """
    nodes_per_layer = []
    signs_per_layer = []

    for k in layers:
        # Extract centers and radii for the current layer
        layer_centers = centers[:, k]
        layer_radii = radii[:]  # Change to radii[:, k] if radii vary with time

        # Calculate disallowed regions
        disallowed_regions = []
        for i in range(len(layer_centers)):
            center = layer_centers[i]
            radius = layer_radii[i]
            disallowed_regions.append((center - radius, center + radius))

        # Merge overlapping disallowed regions
        disallowed_regions.sort()  # Sort by start of intervals
        merged_regions = []
        current_start, current_end = disallowed_regions[0]
        for start, end in disallowed_regions[1:]:
            if start <= current_end:  # Overlap
                current_end = max(current_end, end)
            else:
                merged_regions.append((current_start, current_end))
                current_start, current_end = start, end
        merged_regions.append((current_start, current_end))

        # Determine allowed regions
        allowed_regions = []
        alpha_min, alpha_max = -np.inf, np.inf  # Define the range of alpha_k
        prev_end = alpha_min
        for start, end in merged_regions:
            if prev_end < start:
                allowed_regions.append((prev_end, start))
            prev_end = end
        if prev_end < alpha_max:
            allowed_regions.append((prev_end, alpha_max))

        # Place nodes in allowed regions
        nodes = []
        signs = []
        for start, end in allowed_regions:
            if np.sign(start) != np.sign(end):
                node = 0
            elif abs(start) < abs(end):
                node = start
            elif abs(end) < abs(start):
                node = end
            omega = omega_pseudo[:, k] + (NULL_R @ np.array([[node]])).reshape(-1)
            if np.any(np.abs(omega) > OMEGA_MAX):
                continue
            nodes.append(node)
            signs.append(np.sign(omega))

        nodes_per_layer.append(nodes)
        signs_per_layer.append(signs)

    return nodes_per_layer, signs_per_layer


def build_graph(
    nodes_per_layer: List[List[float]],
    signs_per_layer: List[List[np.ndarray]],
    K: float,
    restricted_intervals: List[Tuple[int, int]]
) -> nx.DiGraph:
    """
    Build a graph with nodes and edges based on the given layers, nodes, and signs.

    Args:
        nodes_per_layer: List of nodes for each layer.
        signs_per_layer: List of signs for each node in each layer.
        K: Weight parameter for zero crossings.
        restricted_intervals: List of (layer1, layer2) pairs where sign changes are disallowed.

    Returns:
        A directed graph with nodes and edges.
    """
    G = nx.DiGraph()
    num_layers = len(nodes_per_layer)

    # Add nodes to the graph
    for layer_idx, nodes in enumerate(nodes_per_layer):
        for node_idx, alpha_k in enumerate(nodes):
            G.add_node((layer_idx, node_idx), alpha_k=alpha_k, sign=signs_per_layer[layer_idx][node_idx])

    # Add edges between consecutive layers
    for layer_idx in range(num_layers - 1):
        for i, alpha_k1 in enumerate(nodes_per_layer[layer_idx]):
            for j, alpha_k2 in enumerate(nodes_per_layer[layer_idx + 1]):
                # Calculate zero crossings
                sign1 = signs_per_layer[layer_idx][i]
                sign2 = signs_per_layer[layer_idx + 1][j]
                zero_crossings = np.sum(sign1 != sign2)

                # Composite cost
                cost = K * zero_crossings + alpha_k2**2

                # Check if the edge is restricted
                if (layer_idx, layer_idx + 1) in restricted_intervals and zero_crossings > 0:
                    continue  # Skip adding this edge

                G.add_edge((layer_idx, i), (layer_idx + 1, j), weight=cost)

    # Add a virtual sink node
    G.add_node("sink")

    # Connect all nodes in the last layer to the sink node
    last_layer_idx = num_layers - 1
    for node_idx in range(len(nodes_per_layer[last_layer_idx])):
        G.add_edge((last_layer_idx, node_idx), "sink", weight=0)
    return G

def solve_graph(G: nx.DiGraph, start_layer: int, end_layer: int, omega_start_sign: np.ndarray) -> List[Tuple[int, int]]:
    """
    Solve the graph to find the shortest path from the start layer to the end layer.

    Args:
        G: The graph.
        start_layer: The starting layer index.
        end_layer: The ending layer index.
        omega_start_sign: The sign of OMEGA_START to determine the starting node.

    Returns:
        The shortest path as a list of (layer_index, node_index) tuples.
    """
    # Find the starting node in layer 0 with the same sign as OMEGA_START
    start_nodes = [
        (start_layer, i)
        for i, data in enumerate(G.nodes(data=True))
        if data[0][0] == start_layer and np.array_equal(data[1]["sign"], omega_start_sign)
    ]
    if not start_nodes:
        raise ValueError("No starting node matches the sign of OMEGA_START.")
    start_node = start_nodes[0]

    # Solve the shortest path
    path = nx.shortest_path(G, source=start_node, target="sink", weight="weight")

    # Remove the virtual sink from the path
    path = [node for node in path if node != "sink"]

    return path

def calc_alpha_limits(centers, radii, path, nodes_per_layer, layer_list):
    """
        Computes global alpha limits across layers based on circular constraints.

        centers: shape (4, N)
        radii: shape (4, 1) or (4,)
        """

    N = centers.shape[1]
    alpha_max = np.full((4, N), np.inf)
    alpha_min = np.full((4, N), -np.inf)

    for idx in range(len(layer_list[:-1])):
        layer1 = layer_list[idx]
        layer2 = layer_list[idx+1]
        node_idx1 = path[idx][1]  # Get the node index for the current layer
        node_idx2= path[idx+1][1]  # Get the node index for the next layer
        alpha1 = nodes_per_layer[idx][node_idx1]
        alpha2 = nodes_per_layer[idx+1][node_idx2]

        for i in range(4):
            first_center_above = True if centers[i, layer1] > alpha1 else False
            second_center_above = True if centers[i, layer2] > alpha2 else False
            if first_center_above and second_center_above:
                alpha_max[i, layer1:layer2+1] = centers[i, layer1: layer2+1] - radii[i, 0]
                alpha_min[i, layer1:layer2+1] = -np.inf
            elif not first_center_above and not second_center_above:
                alpha_min[i, layer1:layer2+1] = centers[i, layer1: layer2+1] + radii[i, 0]
                alpha_max[i, layer1:layer2+1] = np.inf
            else:
                alpha_max[i, layer1:layer2+1] = np.inf
                alpha_min[i, layer1:layer2+1] = -np.inf
    alpha_max = np.min(alpha_max, axis=0)
    alpha_min = np.max(alpha_min, axis=0)
    if np.any(alpha_min > alpha_max):
        print("Infeasible alpha interval detected")
    alpha_optimal = np.clip(0, alpha_min, alpha_max)
    return alpha_optimal

def calc_alpha_torque_limits(torque_4d):
    """
    General torque → alpha bound conversion.

    Parameters
    ----------
    torque_4d : (4, N)
        Fixed torque component (pseudo torque)

    Returns
    -------
    alpha_lower_bound : (N,)
    alpha_upper_bound : (N,)
    """

    n = NULL_R.reshape(4, 1)          # (4,1)
    T = torque_4d                     # (4,N)

    # Avoid division by zero (should not happen in your case,
    # but makes function mathematically correct)
    if np.any(np.abs(n) < 1e-12):
        raise ValueError("NULL_R contains zero entry — invalid for torque constraint.")

    # Compute raw bounds per wheel
    alpha_min_k = (-MAX_TORQUE - T) / n
    alpha_max_k = ( MAX_TORQUE - T) / n

    # If n_k < 0, swap bounds
    alpha_lower_k = np.minimum(alpha_min_k, alpha_max_k)
    alpha_upper_k = np.maximum(alpha_min_k, alpha_max_k)

    # Global feasible alpha interval
    alpha_lower_bound = np.max(alpha_lower_k, axis=0)
    alpha_upper_bound = np.min(alpha_upper_k, axis=0)

    return (alpha_lower_bound, alpha_upper_bound)


def forward_integration_optimal(w_start, alpha_optimal, torque_limits, torque_4d, dt):
    """
        Tracks optimal nullspace coordinate alpha while respecting torque limits.
        """

    N = torque_4d.shape[1]  # 8004

    w_solution = np.zeros((4, N + 1))
    alpha_control = np.zeros(N)
    torque_solution = np.zeros((4, N))

    w_current = w_start.flatten()
    w_solution[:, 0] = w_current

    torque_min, torque_max = torque_limits

    # ---- Precompute constants ----
    N_vec = NULL_R.ravel()

    I_diag = np.diag(I_INV)  # (4,)
    I_INV_N = I_diag * N_vec  # elementwise

    gamma = np.dot(N_vec, I_INV_N)  # scalar
    inv_gamma_dt = 1.0 / (gamma * dt)

    for k in range(N):

        # -------------------------------------------------
        # 2) Current alpha_w
        # -------------------------------------------------
        alpha_current = N_vec @ w_current

        # -------------------------------------------------
        # 3) Compute required null torque to reach target alpha
        # alpha_next = alpha_current + gamma * alpha_T * dt
        # => alpha_T = (alpha_desired - alpha_current) / (gamma * dt)
        # -------------------------------------------------
        alpha_diff = alpha_optimal[k+1] - alpha_current
        null_torque_desired = alpha_diff * inv_gamma_dt

        # -------------------------------------------------
        # 5) Saturate
        # -------------------------------------------------
        null_torque = np.clip(null_torque_desired, torque_min[k], torque_max[k])

        # -------------------------------------------------
        # 6) Apply total torque and integrate
        # -------------------------------------------------
        tau_pseudo = torque_4d[:, k]

        total_torque = tau_pseudo + N_vec * null_torque
        w_current += (I_diag * total_torque) * dt

        # -------------------------------------------------
        # 7) Store results
        # -------------------------------------------------
        w_solution[:, k + 1] = w_current
        alpha_control[k] = null_torque
        torque_solution[:, k] = total_torque

    return w_solution, alpha_control, torque_solution

def plot():

    band_top = band_center + band_radii
    band_bottom = band_center - band_radii

    w_null_sol = w_sol.T @ NULL_R
    w_pseudo_sol = w_pseudo.T @NULL_R
    time_sol = np.arange(w_sol.shape[1]) * 0.1

    plt.figure()
    plt.plot(time_sol, w_null_sol)
    plt.plot(time_sol, w_pseudo_sol)
    for i in range(NUM_WHEELS):
        plt.fill_between(time_sol, band_bottom[i], band_top[i], alpha=0.2)
    for i, layer in enumerate(new_layers):
        nodes = node_list[i]
        for node in nodes:
            alpha_k = node
            plt.scatter(time_sol[layer], alpha_k, color='red')

    plt.show()


if __name__ == "__main__":

    t0 = time.perf_counter()
    w_pseudo, torque_pseudo = forward_integration(OMEGA_START_PSEUDO, data, dt=0.1)
    t1 = time.perf_counter()
    overlap_mask_sorted, band_pairs_mask_sorted, band_order, band_center, band_radii = make_direct_overlap_masks(w_pseudo)
    t2 = time.perf_counter()
    free, not_crossing, crossing = generate_all_intervals(overlap_mask_sorted, band_pairs_mask_sorted, band_center)
    t3 = time.perf_counter()
    selected_layers = [0, 8004]
    # selected_layers = np.arange(0, 8005, 10).tolist()
    print(free)
    print(not_crossing)
    print(crossing)
    free_flat = flatten_list_of_lists(free)
    not_crossing_flat = flatten_list_of_lists(not_crossing)
    crossing_flat = flatten_list_of_lists(crossing)
    all = free_flat + not_crossing_flat + crossing_flat
    t4 = time.perf_counter()
    new_layers = set_covering(all, selected_layers)
    t5 = time.perf_counter()
    print(new_layers)
    t6 = time.perf_counter()
    new_layers = set_covering_greedy(all, selected_layers)
    t7 = time.perf_counter()
    print(new_layers)
    node_list, sign_list = determine_nodes(w_pseudo, band_center, band_radii, new_layers)
    t8 = time.perf_counter()


    K=10
    restricted_intervals = []
    G = build_graph(node_list, sign_list, K, restricted_intervals)
    t9 = time.perf_counter()

    # Solve the graph
    shortest_path = solve_graph(G, start_layer=0, end_layer=8004, omega_start_sign=np.sign(OMEGA_START.flatten()))
    t10 = time.perf_counter()
    path_constraint = calc_alpha_limits(band_center, band_radii, shortest_path, node_list, new_layers)
    t11 = time.perf_counter()
    torque_constraint = calc_alpha_torque_limits(torque_pseudo)
    t12 = time.perf_counter()
    w_sol, alpha_sol, torque_sol = forward_integration_optimal(OMEGA_START, path_constraint, torque_constraint, torque_pseudo, dt=0.1)
    t13 = time.perf_counter()



    print("Shortest Path:", shortest_path)

    print(f"Integration time: {t1 - t0:.4f} s")
    print(f"Mask generation time: {t2 - t1:.4f} s")
    print(f"Interval generation time: {t3 - t2:.4f} s")
    print(f"Set covering time: {t5 - t4:.4f} s")
    print(f"Set covering greedy time: {t7 - t6:.4f} s")
    print(f"Flattening time {t4 - t3:.4f} s")
    print(f"Node determination time: {t8 - t7:.4f} s")
    print(f"Graph building time: {t9 - t8:.4f} s")
    print(f"Graph solving time: {t10 - t9:.4f} s")
    print("Nodes per layer:", node_list)
    print(f"Make path constraints: {t11 - t10:.4f} s")
    print(f"Make torque constraints: {t12 - t11:.4f} s")
    print(f"Final integration time: {t13 - t12:.4f} s")

    print("OMEGA START:", OMEGA_START.flatten())
    print("OMEGA_START_PSEUDO:", OMEGA_START_PSEUDO.flatten())
    print("OMEGA_START_NULL:", OMEGA_START_NULL.flatten())
    plot()


