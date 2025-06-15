from config import *
from helper import *
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice
from solvers import Solver

def cut_data(torque_data):
    """
    Cuts the torque data into segments based on low torque thresholds and detects transitions.
    :param torque_data: numpy array of shape (3, N) representing torque data.
    :return:
        - temp_low_torque_flag: boolean array indicating low torque segments.
        - rising: indices where the low torque flag rises.
        - falling: indices where the low torque flag falls.
        - stationary: indices representing last index of each interval (~200s).
    """
    temp_low_torque_flag = hysteresis_filter(torque_data, 0.000005, 0.000015)
    temp_low_torque_flag[0:2] = False
    rising, falling = detect_transitions(temp_low_torque_flag)
    stationary = np.insert(falling - 1, 0, 0)
    stationary = np.append(stationary, len(temp_low_torque_flag))
    falling = np.insert(falling, 0, 0)
    falling = np.append(falling, len(temp_low_torque_flag))  # +1 but works?
    return temp_low_torque_flag, rising, falling, stationary

def calc_segments(omega_input):
    """
    Calculates speed bands based on the input omega.
    :param omega_input: numpy array of shape (4, N) representing wheel speeds.
    :return: segment_bis: numpy array of shape (8, N) representing speed bands. Row 1 is upper bound for band 1,
    row 2 is lower bound for band 1, etc.
    Code should work for any NULL_R.
    """
    sign_vector = NULL_R
    omega = omega_input

    seg1 = (OMEGA_MIN - omega) * sign_vector
    seg2 = (-OMEGA_MIN - omega) * sign_vector

    segments_bis = np.empty((8, omega.shape[1]))

    for i in range(4):
        if sign_vector[i, 0] > 0:
            # seg1 is upper, seg2 is lower → no sign flip, just ordered
            segments_bis[2 * i]     = seg2[i]  # lower
            segments_bis[2 * i + 1] = seg1[i]  # upper
        else:
            # seg1 is lower, seg2 is upper → still no sign flip
            segments_bis[2 * i]     = seg1[i]  # lower
            segments_bis[2 * i + 1] = seg2[i]  # upper

    return segments_bis

def alpha_bounds(omega_input):
    """
    Calculates the saturation bounds for alpha based on the input omega.
    :param omega_input: numpy array of shape (4, N) representing wheel speeds.
    :return:
        - alpha_lower: numpy array of shape (N,) representing lower bounds for alpha.
        - alpha_upper: numpy array of shape (N,) representing upper bounds for alpha.
    """
    sign_vector = -NULL_R  # shape (4,1)
    signed_omega = omega_input * sign_vector  # shape (4, N)

    lower_bounds = (signed_omega - OMEGA_MAX)  # shape (4, N)
    upper_bounds = (signed_omega + OMEGA_MAX)

    alpha_lower = np.max(lower_bounds, axis=0)  # (N,)
    alpha_upper = np.min(upper_bounds, axis=0)  # (N,)

    return alpha_lower, alpha_upper

def alpha_options(omega_input, reference=0, use_bounds=True):
    """
    Calculates the alpha options based on the input omega.
    :param omega_input: (4, N) numpy array representing wheel speeds.
    :param reference: Float, Represents the optimal value for alpha.
    :param use_bounds: Boolean, Use the saturation bounds to limit the alpha options.
    :return: all_alphas: (N) List of List, each containing the 1 to 5 alpha options for each time step.
    """
    first = (OMEGA_MIN - omega_input) * NULL_R  # shape (4, N)
    second = (-OMEGA_MIN - omega_input) * NULL_R  # shape (4, N)
    intervals = np.stack((first, second), axis=2)  # shape: (4, N, 2)
    sorted_intervals = np.sort(intervals, axis=2)
    # Each wheel produces 1 interval [x1, x2] indicating stiction zone. Sorted by start value (lowest value).

    if use_bounds:
        lower, upper = alpha_bounds(omega_input)
        alpha_min_bounds = np.vstack((-np.inf * np.ones_like(lower), lower)).T.reshape(1, -1, 2)
        alpha_max_bounds = np.vstack((upper, np.inf * np.ones_like(upper))).T.reshape(1, -1, 2)
        sorted_intervals = np.concatenate((sorted_intervals, alpha_min_bounds, alpha_max_bounds), axis=0)  # shape (6, N, 2)

    all_alphas = []
    for j in range(omega_input.shape[1]):
        # For each time step, merge the intervals if they overlap.
        current_intervals = sorted_intervals[:, j, :]  # shape (4, 2)
        sorted_by_start = current_intervals[np.argsort(current_intervals[:, 0])]
        merged_intervals = []
        for interval in sorted_by_start:
            # Merge interval if next interval starts before the last one ends.
            if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
        alphas = best_alpha(merged_intervals, reference, use_bounds=use_bounds) # See helper.py for best_alpha function
        # From stiction intervals, calculate the best alpha options
        all_alphas.append(alphas)
    return all_alphas

def calc_indices(bands, flag, rising_ids):
    """
    Calculates the indices of the nodes.
    :param bands: numpy array of shape (8, 8005) representing stiction bands.
    :param flag: numpy array of shape (8004,) representing low torque flags.
    :param rising_ids: List of indices where the low torque flag rises.
    :return:
        - indices: List of indices where the nodes are located.
        - inverted_intervals: Dict with keys as band indices and values as lists of inverted intervals.
        - overlapping_intervals: Dict with keys as tuples of band indices and values as lists of overlapping intervals.
    """
    crossing_intervals, overlapping_intervals = compute_band_intersections(bands)
    inverted_intervals = invert_intervals(crossing_intervals, 0, len(flag))
    selected_ids = select_minimum_covering_nodes(overlapping_intervals, (0, 8004), initial_nodes=[0] + list(rising_ids))
    selected_ids = select_minimum_covering_nodes(inverted_intervals, (0, 8004), initial_nodes=selected_ids)
    return selected_ids, inverted_intervals, overlapping_intervals

def build_graph(sections):
    """
    Builds a directed graph from the given sections.
    :param sections:
    :return: Directed graph with nodes and edges representing the sections and their layers.
    """
    G = nx.DiGraph()
    node_id = 0

    # Pass 1: Add all nodes with IDs
    for i, section in enumerate(sections):
        for layer in section.layers:
            for node in layer.nodes:
                node.id = f"{section.name}_{layer.time_index}_{node_id}"
                node.display_name = f"{node_id}"
                G.add_node(node.id, node=node)
                node_id += 1
            node_id = 0  # comment out for staggered view
        if i == len(sections) - 1:
            for node in section.stationary_layer.nodes:
                node.id = f"{section.name}_{section.stationary_layer.time_index}_{node_id}"
                node.display_name = f"{node_id}"
                G.add_node(node.id, node=node)
                node_id += 1
    goal_node = NodeOption(alpha=0, base_speed=0, node_type="goal")
    goal_node.id, goal_node.display_name = "goal_8200_0", "goal"
    G.add_node(goal_node.id, node=goal_node)

    # Pass 2: Add all edges
    for i, section in enumerate(sections):
        all_layers = section.layers
        for l1, l2 in zip(all_layers[:-1], all_layers[1:]):
            for n1 in l1.nodes:
                for n2 in l2.nodes:
                    sign_diff = sum(s1 != s2 for s1, s2 in zip(n1.signs, n2.signs))
                    # G.add_edge(n1.id, n2.id, sign_cost=sign_diff, vibration_cost=0)
                    G.add_edge(n1.id, n2.id, sign_cost=sign_diff, vibration_cost=n1.vibration_cost*1e-4)
                    # Slight optimisation for moving phase

        # End → Next Begin (same signs)
        if i < len(sections) - 1:
            next_begin = sections[i + 1].begin_layer
            for n_end in section.end_layer.nodes:
                for n_next in next_begin.nodes:
                    if np.array_equal(n_end.signs, n_next.signs):
                        for n_stat in section.stationary_layer.nodes:
                            if np.array_equal(n_end.signs, n_stat.signs):
                                G.add_edge(n_end.id, n_next.id, sign_cost=0, vibration_cost=n_stat.vibration_cost)

        else:
            # Last section → Goal
            for n_end in section.end_layer.nodes:
                for n_stat in section.stationary_layer.nodes:
                    if np.array_equal(n_end.signs, n_stat.signs):
                        G.add_edge(n_end.id, n_stat.id, sign_cost=0, vibration_cost=n_stat.vibration_cost)
                        G.add_edge(n_stat.id, goal_node.id, sign_cost=0, vibration_cost=0)

    return G

def shortest_path(G, source_ids, target_ids, cost_type="sign_cost"):
    """
    Finds the shortest path in a directed graph G from one or more source nodes to one or more target nodes.
    :param G: networkx.DiGraph, the directed graph to search.
    :param source_ids: List of source node IDs or a single source node ID.
    :param target_ids: List of target node IDs or a single target node ID.
    :param cost_type: String, the edge attribute to use for path cost calculation (default is "sign_cost").
    :return:
    - best_path: List of node IDs representing the best path found.
    - best_cost: Float, the cost of the best path found.
    - best_sign_cost: Integer, the sign cost of the best path found.
    - best_vibration_cost: Float, the vibration cost of the best path found.
    """
    best_path = None
    best_cost = float("inf")
    best_sign_cost = None
    best_vibration_cost = None

    for source in source_ids: # source_ids can be a list of source nodes
        for target in target_ids: # target_ids can be a list of target nodes
            try:
                time1 = clock.time()
                temp_path = nx.shortest_path(G, source=source, target=target, weight=cost_type)
                temp_cost = nx.path_weight(G, temp_path, weight=cost_type)
                if temp_cost < best_cost:
                    best_cost = temp_cost
                    best_path = temp_path
                    best_sign_cost = nx.path_weight(G, temp_path, weight="sign_cost")
                    best_vibration_cost = nx.path_weight(G, temp_path, weight="vibration_cost")
            except nx.NetworkXNoPath:
                continue
    return best_path, best_cost, best_sign_cost, best_vibration_cost

def plot_shortest_path(graph, path, get_coordinates=None):
    pos = {}
    xs = []
    for node in graph.nodes():
        if get_coordinates:
            x, y = get_coordinates(node)
            # x = x - (x // 2000) * 1000
            y = 2*y
        else:
            x, y = 0, 0
        pos[node] = (x, y)  # Flip y for nicer top-down plotting
        xs.append(x)

    plt.figure(figsize=(12, 4))
    nx.draw_networkx_edges(graph, pos, alpha=0.2)
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='lightgray')

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_nodes(graph, pos, nodelist=path, node_size=150, node_color='orange')
        nx.draw_networkx_edges(graph, pos, edgelist=path_edges, width=2, edge_color='red')
    # for node_id in graph.nodes:
    #     print(node_id)
    labels = {
        node_id: graph.nodes[node_id]['node'].display_name
        for node_id in graph.nodes
    }
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6)
    unique_xs = sorted(set(xs))
    for x in unique_xs:
        plt.text(x, -5, str(x), rotation=90, fontsize=8, ha='center', va='top')
        plt.axvline(x, color='lightgray', linestyle='--', linewidth=0.5)

    # plt.title(title)
    plt.axis('off')
    plt.xlabel("Time index or X coordinate")
    plt.tight_layout()
    plt.show()

def get_coord(node_id):
    parts = node_id.split('_')
    try:
        layer = int(parts[-2])  # second to last part = layer
        index = int(parts[-1])  # last part = index
        return layer, index
    except Exception as e:
        print(f"Error parsing node ID '{node_id}': {e}")
        return 0, 0

def k_shortest_paths(g, source_ids, target_ids, k, cost_type="sign_cost"):
    """
    Finds the k shortest simple paths in a directed graph g from one or more source nodes to one or more target nodes.
    :param k:
    :param G: networkx.DiGraph, the directed graph to search.
    :param source_ids: List of source node IDs or a single source node ID.
    :param target_ids: List of target node IDs or a single target node ID.
    :param k: Integer, the number of shortest paths to find.
    :param cost_type: String, the edge attribute to use for path cost calculation (default is "sign_cost").
    :return: List of tuples, each containing:
        - temp_path: List of node IDs representing the best path found.
        - temp_cost: Float, the cost of the best path found.
        - temp_sign_cost: Integer, the sign cost of the best path found.
        - temp_vibration_cost: Float, the vibration cost of the best path found.
    """
    all_paths = []

    for source in source_ids:
        for target in target_ids:
            try:
                paths_gen = nx.shortest_simple_paths(g, source, target, weight=cost_type)
                for temp_path in islice(paths_gen, k):
                    temp_mixed_cost = nx.path_weight(g, temp_path, weight=cost_type)
                    temp_sign_cost = nx.path_weight(g, temp_path, weight="sign_cost")
                    temp_vibration_cost = nx.path_weight(g, temp_path, weight="vibration_cost")
                    all_paths.append((temp_path, temp_mixed_cost, temp_sign_cost, temp_vibration_cost))
            except nx.NetworkXNoPath:
                continue

    # Sort all found paths by the mixed cost and return top k
    all_paths = sorted(all_paths, key=lambda x: x[1])[:k]
    return all_paths

def assign_mixed_costs(G, penalty=0):
    """
    Assigns a mixed cost to each edge in the graph G based on the sign cost and vibration cost.
    :param G: networkx.DiGraph, the directed graph to modify.
    :param penalty: zero-crossing penalty
    """
    for u, v, data in G.edges(data=True):
        temp_sign_cost = data.get("sign_cost", 0)
        temp_vibration_cost = data.get("vibration_cost", 0)
        mixed = temp_vibration_cost + penalty * temp_sign_cost
        data["mixed_cost"] = mixed

def extract_info_from_path(graph, path):
    """
    Extracts time index, alpha, and signs from the nodes in the given path.
    :param graph:
    :param path:
    :return: List of tuples containing (time_index, alpha, signs) for each node in the path.
    """
    results = []

    def get_node_data(node_id):
        node_data = graph.nodes[node_id].get('node')
        if node_data is None:
            return None, None
        return node_data.time_index, node_data.alpha, node_data.signs

    for node_id in path[:-1]:  # Handle all nodes except the last normally
        t, alpha, signs = get_node_data(node_id)
        if t is not None and alpha is not None:
            results.append((t, alpha, signs))

    # Handle the last node specially
    last_node = path[-1]
    last_data = graph.nodes[last_node].get('node')
    if last_data is None and len(path) >= 2:  # Useless code maybe
        t, alpha = get_node_data(path[-2])
        if t is not None and alpha is not None:
            results.append((t, alpha))

    return results

def build_alpha_constraints(alpha_nodes, bands):
    """
    Builds the alpha constraints based on the selected path and bands.
    :param alpha_nodes: List of tuples, each containing (time_index, alpha, signs).
    :param bands: numpy array of shape (8, N) representing stiction bands.
    :return:
        - min_constraints: numpy array of shape (N,) representing minimum constraints for alpha.
        - max_constraints: numpy array of shape (N,) representing maximum constraints for alpha.
    """
    N = bands.shape[1]
    min_constraints = np.full(N, -np.inf)
    max_constraints = np.full(N, np.inf)  # Replace with calculated bounds

    for j in range(len(alpha_nodes) - 1):
        t_start, alpha_start, signs_start = alpha_nodes[j]
        t_end, alpha_end, signs_end = alpha_nodes[j + 1]

        for i in range(4):  # For each wheel
            if signs_start[i] != signs_end[i]:
                continue  # Skip if the band is crossed (sign changes)

            band_idx = 2 * i  # Index for band (each wheel has 2 rows)
            band_max = bands[band_idx + 1, t_start:t_end]
            band_min = bands[band_idx, t_start:t_end]

            alpha_val = alpha_start

            if alpha_val < np.mean([band_max[0], band_min[0]]):  # Band is above alpha → it's a max constraint
                max_constraints[t_start:t_end] = np.minimum(max_constraints[t_start:t_end], band_min)

            elif alpha_val > np.mean([band_max[0], band_min[0]]):  # Band is below alpha → it's a min constraint
                min_constraints[t_start:t_end] = np.maximum(min_constraints[t_start:t_end], band_max)

    return min_constraints, max_constraints

def plot(n0=0, n=8005, limits=None, null_path=None, bounds=None,
         segments=None, scatter_points=None, start_point=None, momentum=None, legend=False, all_scatter_points=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    time = np.linspace(0, 800, 8005)
    zeros = np.zeros(8005)
    # ax.plot(time, zeros, color='black', linestyle='--', label='Ideal path')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if segments is not None:
        for i in range(0, 8, 2):
            color = colors[i // 2 % len(colors)]  # Cycle through colors
            avg = (segments[i] + segments[i + 1]) / 2
            plt.plot(time, avg, color=color)
            plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    if scatter_points is not None:
        for t, alpha, signs in scatter_points:
            plt.scatter(time[t], alpha, color='red', marker='o')
            # plt.scatter(time[2002:2003], [0], color='green', marker='x')
    if all_scatter_points is not None:
        blue_dots = [689, 2003, 2690, 4004, 5100, 6005, 7101]
        for section in all_scatter_points:
            for layer in section.layers:
                for alpha in layer.alphas:
                    if layer.time_index == 0:
                        continue
                    if layer.time_index in blue_dots:
                        plt.scatter(time[layer.time_index], alpha, color='blue', marker='o')
                    else:
                        plt.scatter(time[layer.time_index], alpha, color='green', marker='o')
    if start_point is not None:
        plt.scatter(0, start_point, color='blue', marker='x')
    if limits is not None:
        min_limit, max_limit = limits
        plt.plot(time, max_limit, color='black', linestyle='--', label='Max Constraint')
        plt.plot(time, min_limit, color='black', label='Min Constraint')
    if null_path is not None:
        N = null_path.shape[0]
        plt.plot(time[0:N], null_path, color='blue', linestyle='--', label='Nullspace Path')
    if bounds is not None:
        min_bounds, max_bounds = alpha_bounds(bounds)
        plt.plot(time, min_bounds, color='gray', linestyle='--', label='Saturation limit')
        plt.plot(time, max_bounds, color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel(r'Nullspace component $\left[\mathrm{kg{\cdot}m^2/s}\right]$')
    if legend:
        plt.legend()
    # plt.xlim(380, 500)  # Zoom in on x-axis between 20 and 40
    # plt.ylim(-110, 110)
    plt.show()

def find_start_node(sections, base_speed):
    """
    Finds the first node in the first section that matches the desired signs based on the starting speed.
    :param sections:
    :param base_speed: numpy array of shape (4, 1) representing the starting speed.
    :return: String sector name and node ID of the first matching node, or None if no match is found.
    """
    desired_signs = np.sign(base_speed).flatten()
    first_section = sections[0]
    first_begin_layer = first_section.begin_layer
    for node in first_begin_layer.nodes:
        if np.array_equal(node.signs, desired_signs):
            return [f"{first_section.name}_{node.id}"]
    return None  # No matching node found

def find_initial_guess(min_constraint, max_constraint, omega):
    """
    Finds an initial guess for the alpha (nullspace) and omega values based on the given constraints. This finds
    the optimal value for alpha within the given constraints WITHOUT account for dynamics (limited torque).
    :param min_constraint: numpy array of shape (N,) representing minimum constraints for alpha.
    :param max_constraint: numpy array of shape (N,) representing maximum constraints for alpha.
    :param omega: numpy array of shape (4, N) representing the initial omega values.
    :return:
        - alpha_guess: numpy array of shape (N,) representing the initial guess for alpha.
        - omega_guess: numpy array of shape (4, N) representing the initial guess for omega.
    """
    alpha_guess = np.clip(0, min_constraint, max_constraint)
    omega_guess = omega + NULL_R @ alpha_guess.reshape(1, -1)
    return alpha_guess, omega_guess

def plot_overlap_intervals(overlaps, node_ids=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    y_labels = []
    for idx, ((i, j), intervals) in enumerate(overlaps.items()):
        y = 6 - idx  # stacked from top to bottom
        for start, end in intervals:
            ax.hlines(y, start, end, colors='red', linewidth=6)
        y_labels.append(f"Band {i+1} ∩ Band {j+1}")
    if node_ids is not None:
        for x in node_ids:
            ax.axvline(x, color='blue', linestyle='--', linewidth=1)
    ax.set_yticks(range(1, 7))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Time")
    ax.set_title("Overlap Regions Between Forbidden Bands")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def init_guesser(alpha_target, torque, omega_start, specific_torque_limits=False):
    """
    Tracks a target alpha within torque and omega limits while respecting the dynamics of the system.
    :param alpha_target: numpy array of shape (N,) representing the target alpha momentum values.
    :param torque: numpy array of shape (3, N-1) representing the torque data.
    :param omega_start: numpy array of shape (4, 1) representing the starting omega values.
    :param specific_torque_limits: Boolean, whether to use specific torque limits or the maximum torque.
    :return:
        - w_guess: numpy array of shape (4, N) representing the omega values.
        - alpha_control_guess: numpy array of shape (N-1,) representing the alpha control values.
        - torque_guess: numpy array of shape (4, N-1) representing the torque values.
    """
    N = 8005
    dt = 0.1
    w = omega_start
    alpha_control_guess = np.zeros(N-1)
    w_guess = np.zeros((4, N))
    w_guess[:, 0] = w.flatten()
    torque_guess = np.zeros((4, N-1))

    if specific_torque_limits:
        alpha_lower, alpha_upper = calc_alpha_torque_limits(torque, 0, N-1)
        def get_upper_torque_limits(l):
            return alpha_upper[l]
        def get_lower_torque_limits(l):
            return alpha_lower[l]
    else:
        def get_upper_torque_limits(l):
            return MAX_TORQUE
        def get_lower_torque_limits(l):
            return -MAX_TORQUE

    for k in range(N-1):
        T_sc = torque[:, k:k+1]
        T_rw = R_PSEUDO @ T_sc + NULL_R @ np.array([[0]])
        der_state = I_INV @ T_rw
        w_next = w + der_state * dt
        alpha_null = NULL_R_T @ w_next / 4
        alpha_null_wanted = alpha_target[k+1]
        alpha_diff = alpha_null_wanted - alpha_null

        # T_rw = NULL_R * alpha_diff*IRW/dt
        alpha_guess = alpha_diff*IRW/dt
        alpha_guess = np.clip(alpha_guess, get_lower_torque_limits(k), get_upper_torque_limits(k))

        der_state = I_INV @ NULL_R * alpha_guess
        w = w_next + der_state * dt  # w_next of w
        w_guess[:, k+1] = w.flatten()
        alpha_control_guess[k] = alpha_diff.item()
        torque_guess[:, k] = (T_rw + NULL_R * alpha_guess).flatten()

    return w_guess, alpha_control_guess, torque_guess

def build_omega_constraints(min_constraint, max_constraint, omega_pseudo):
    """
    Builds the omega constraints based on the given minimum and maximum constraints.
    :param min_constraint: numpy array of shape (N,) representing minimum alpha momentum.
    :param max_constraint: numpy array of shape (N,) representing maximum alpha momentum.
    :param omega_pseudo: numpy array of shape (4, N) representing the pseudo omega values.
    :return:
        - w_sol_min: numpy array of shape (4, N) representing the minimum omega constraints.
        - w_sol_max: numpy array of shape (4, N) representing the maximum omega constraints.
    """
    pos_mask = (NULL_R > 0)

    # Broadcast masks to shape (4,8005)
    w_sol_max = np.where(pos_mask, omega_pseudo + max_constraint, omega_pseudo - min_constraint)
    w_sol_min = np.where(pos_mask, omega_pseudo + min_constraint, omega_pseudo - max_constraint)

    w_sol_max = np.clip(w_sol_max, -OMEGA_MAX, OMEGA_MAX)
    w_sol_min = np.clip(w_sol_min, -OMEGA_MAX, OMEGA_MAX)
    return w_sol_min, w_sol_max

class NodeOption:
    def __init__(self, alpha, base_speed, node_type="mid", layer_idx=None, local_id=None):
        self.alpha = alpha
        self.type = node_type  # 'begin', 'end', 'stationary' or 'mid'
        self.time_index = layer_idx
        self.layer_type = node_type
        self.wheel_speeds = base_speed + (NULL_R * alpha).flatten()
        self.vibration_cost = np.sum(self.wheel_speeds**2)
        self.signs = np.sign(self.wheel_speeds)
        self.id = f"{layer_idx}_{local_id}"
        self.config = None
        self.populate_node()

    def populate_node(self):
        pass

    def __repr__(self):
        return f"NodeOption(alpha={self.alpha}, sign={self.signs}, type={self.type})"

class Layer:
    def __init__(self, time_index, speeds, options_list, layer_type="normal"):
        self.time_index = time_index
        self.layer_type = layer_type
        self.wheel_speeds = speeds
        self.nodes = []
        self.alphas = None
        self.number_of_nodes = None
        self.populate_layer(options_list)

    def populate_layer(self, options_list):
        if options_list is None:
            return
        self.alphas = options_list[self.time_index]
        self.number_of_nodes = len(self.alphas)
        for k, alpha in enumerate(self.alphas):
            node = NodeOption(alpha=alpha, base_speed=self.wheel_speeds, layer_idx=self.time_index, local_id=k)
            self.add_node(node)

    def add_node(self, node: NodeOption):
        self.nodes.append(node)

    def __repr__(self):
        return f"Layer(t={self.time_index}, options={len(self.nodes)})"

class Section:
    def __init__(self, name, start_idx, end_idx, stationary_idx, options_list, speeds):
        self.name = name
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.stationary_idx = stationary_idx
        self.layers = []
        self.speeds = speeds
        self.alpha_options = options_list[start_idx:stationary_idx]
        self.options = options_list
        self.begin_layer = Layer(start_idx, self.speeds[:, self.start_idx], self.options,"begin")
        self.end_layer = Layer(end_idx, self.speeds[:, self.end_idx], self.options, "end")
        self.stationary_layer = Layer(stationary_idx, self.speeds[:, self.stationary_idx], self.options,"stationary")  # Not in the layers list

    def populate_section(self, start_idx, end_idx, stationary_idx, ids):
        mid_idx = [val for val in ids if start_idx < val < end_idx]
        self.add_layer(self.begin_layer)
        for j in range(len(mid_idx)):
            layer = Layer(mid_idx[j], self.speeds[:, mid_idx[j]], self.options, "normal")
            self.add_layer(layer)
        self.add_layer(self.end_layer)


    def add_layer(self, layer: Layer):
        self.layers.append(layer)


    def __repr__(self):
        return f"Section({self.name}, layers={len(self.layers)})"

def solve(omega_start, torque_data, specific_torque_limits=False, penalty=0):
    low_torque_flag, rising, falling, stationary = cut_data(torque_data)

    momentum4_with_nullspace = pseudo_sol(torque_data, omega_start)
    alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:, 0:1])
    momentum3 = R @ momentum4_with_nullspace
    momentum4 = R_PSEUDO @ momentum3
    segments = calc_segments(momentum4)
    indices, solution_space, overlap_space = calc_indices(segments, low_torque_flag, rising)

    problem = []
    options = alpha_options(momentum4)
    for i, end_index in enumerate(rising):
        start_index = falling[i]
        stationary_index = stationary[i + 1]
        section_name = f"Section_{i + 1}"
        section = Section(section_name, start_index, end_index, stationary_index, options, momentum4)
        section.populate_section(start_index, end_index, stationary_index, indices)
        problem.append(section)

    source_index = find_start_node(problem, omega_start)  # source_ids = ["Section_1_0_3"]
    target_index = ["goal_8200_0"]
    graph = build_graph(problem)
    assign_mixed_costs(graph, penalty=penalty)
    path, mixed_cost, sign_cost, vib_cost = shortest_path(graph, source_index, target_index, cost_type="mixed_cost")
    # plot_shortest_path(graph, path, get_coordinates=get_coord, title="Graph of the Shortest Path")

    temp_alpha_points = extract_info_from_path(graph, path)
    temp_alpha_min, temp_alpha_max = build_alpha_constraints(temp_alpha_points, segments)
    temp_omega_min, temp_omega_max = build_omega_constraints(temp_alpha_min, temp_alpha_max, momentum4)
    temp_guess_alpha, temp_guess_omega = find_initial_guess(temp_alpha_min, temp_alpha_max, momentum4)
    temp_w_sol, temp_alpha_sol, temp_torque_sol = init_guesser(temp_guess_alpha, torque_data, omega_start, specific_torque_limits=specific_torque_limits)
    temp_null_sol = nullspace_alpha(temp_w_sol)
    return (temp_alpha_points, temp_alpha_min, temp_alpha_max, temp_omega_min, temp_omega_max,
            temp_guess_alpha, temp_guess_omega, temp_w_sol, temp_torque_sol, temp_alpha_sol, temp_null_sol, momentum4, segments, problem)

def save_class_to_mat(classs_instance):
    data_to_save = {
        'solve_time': classs_instance.solve_time,
        'setup_time': classs_instance.setup_time,
        'iteration_count': classs_instance.iteration_count,
        'all_w_sol': classs_instance.w_sol.T,
        'all_alpha_sol': classs_instance.alpha_sol,
        'all_T_sol': classs_instance.T_rw_sol.T,
        'null_sol': classs_instance.null_sol,
        'all_t': classs_instance.temp_time,
    }
    savemat('Data/DAC/solver_results.mat', data_to_save)

if __name__ == "__main__":
    data = load_data('Data/Slew1.mat')
    (alpha_points, alpha_min, alpha_max, omega_min, omega_max, guess_alpha,
     guess_omega, w_sol, torque_sol, alpha_sol, null_sol, speeds, speed_segments, graph_problem) = solve(OMEGA_START, data, specific_torque_limits=True)
    plot(#scatter_points=alpha_points,
         limits=(alpha_min,alpha_max), segments=speed_segments, legend=True, null_path=null_sol)
    #plot(all_scatter_points=graph_problem, segments=speed_segments,
     #    start_point=nullspace_alpha(OMEGA_START), legend=True
         #scatter_points=alpha_points)

    # plot(segments=speed_segments, bounds=speeds, legend=True)
    # _, rising, falling, stationary = cut_data(data)
    # og = Solver(data, OMEGA_START, omega_selective_limits=(omega_min, omega_max), reduce_torque_limits=True)
    # og = Solver(data, OMEGA_START)
    #
    # begin_time = clock.time()
    # og.oneshot_casadi(n0=0, N=8004, torque_on_g=False, omega_on_g=False, penalise_stiction=True)
    # end_time =
    # clock.time()
    # print("Time taken for casadi:", end_time - begin_time)
    # plot(scatter_points=alpha_points, limits=(alpha_min,alpha_max), null_path=og.null_sol, segments=speed_segments)

