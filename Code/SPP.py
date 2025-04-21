import numpy as np
import casadi as ca
from config import *
from helper import *
import matplotlib.pyplot as plt
from rockit import Ocp, MultipleShooting
import heapq
from collections import defaultdict
from pprint import pprint
import time as clock
import networkx as nx


def calc_segments(omega_input):
    length = omega_input.shape[1]
    segments = np.zeros((8, length))  # Constraints: 4 bands
    for j in range(length):
        omega = momentum4[:, j]
        for i in range(4):
            segments[2 * i, j] = (OMEGA_MIN - omega[i]) * (-1) ** i
            segments[1 + 2 * i, j] = (-OMEGA_MIN - omega[i]) * (-1) ** i
    return segments

def alpha_options(omega_input, reference=0):
    all_alphas = []
    for j in range(omega_input.shape[1]):
        intervals = np.zeros((4, 2))
        omega_begin = omega_input[:, j]
        for i in range(4):
            first = (OMEGA_MIN - omega_begin[i]) * (-1) ** i
            second = (-OMEGA_MIN - omega_begin[i]) * (-1) ** i
            intervals[i] = np.sort([first, second]).flatten()  # Potentially sort manually
        sorted_intervals = intervals[np.argsort(intervals[:, 0])]
        merged_intervals = []
        for interval in sorted_intervals:
            if not merged_intervals or merged_intervals[-1][1] < interval[0]:
                merged_intervals.append(interval)
            else:
                merged_intervals[-1][1] = max(merged_intervals[-1][1], interval[1])
        alphas = best_alpha(merged_intervals, reference)
        all_alphas.append(alphas)
    return all_alphas

def calc_indices(possible_options):
    options_count = np.array([len(a) for a in possible_options])
    change_indices = np.where(np.diff(options_count) != 0)[0] + 1
    mid_indices = []
    for i in range(len(change_indices) - 1):
        start = change_indices[i]
        end = change_indices[i + 1]
        # Only include midpoint if the second region has *more* options than the first
        if options_count[start] > options_count[end]:
            mid_indices.append((start + end) // 2)
    # mid_indices = [(change_indices[i] + change_indices[i+1]) // 2 for i in range(len(change_indices) - 1)]
    return [0] + mid_indices + [rising[k] - falling[k] - 1]  # VERIFY -1

def alpha_to_sign(all_alphas, omega_input, indexes):
    all_result = []
    for i in range(len(indexes)):
        result = []
        for j in range(len(all_alphas[indexes[i]])):
            out = omega_input[:, indexes[i]] - (all_alphas[indexes[i]][j] * NULL_R).flatten()
            result.append(np.sign(np.array(out)))
        all_result.append(result)
    return all_result


def count_sign_changes(a, b):
    return sum(ai != bi for ai, bi in zip(a, b))

def build_adjacency_dict(layers):
    adjacency = {}
    for i in range(len(layers) - 1):  # for each layer except the last
        for idx_from, node_from in enumerate(layers[i]):
            from_key = (i, idx_from)
            adjacency[from_key] = {}
            for idx_to, node_to in enumerate(layers[i+1]):
                to_key = (i+1, idx_to)
                cost = count_sign_changes(node_from, node_to)
                adjacency[from_key][to_key] = cost
    return adjacency

def find_all_shortest_paths_from_adjacency(adjacency):
    # Determine all layers
    all_nodes = set(adjacency.keys()) | {n for targets in adjacency.values() for n in targets}
    layers = defaultdict(list)
    for layer_idx, node_idx in all_nodes:
        layers[layer_idx].append(node_idx)

    min_layer = min(layers)
    max_layer = max(layers)

    start_nodes = [(min_layer, idx) for idx in layers[min_layer]]
    end_nodes = [(max_layer, idx) for idx in layers[max_layer]]

    all_paths = {}

    for start in start_nodes:
        heap = [(0, start, [start])]
        visited = {}

        while heap:
            cost, current, path = heapq.heappop(heap)
            if current in visited and visited[current] <= cost:
                continue
            visited[current] = cost

            if current[0] == max_layer:
                all_paths[(start[1], current[1])] = (cost, path)
                continue

            for neighbor, edge_cost in adjacency.get(current, {}).items():
                heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))

    return all_paths

def find_all_equal_shortest_paths(adjacency):
    # Build layer info from adjacency keys
    all_nodes = set(adjacency.keys()) | {n for targets in adjacency.values() for n in targets}
    layers = defaultdict(list)
    for layer_idx, node_idx in all_nodes:
        layers[layer_idx].append(node_idx)

    min_layer = min(layers)
    max_layer = max(layers)

    start_nodes = [(min_layer, idx) for idx in layers[min_layer]]
    end_nodes = [(max_layer, idx) for idx in layers[max_layer]]

    # Store paths like: (start_idx, end_idx): [(cost, [path]), ...]
    all_paths = defaultdict(list)

    for start in start_nodes:
        heap = [(0, start, [start])]
        visited = defaultdict(lambda: float('inf'))

        while heap:
            cost, current, path = heapq.heappop(heap)

            if cost > visited[current]:
                continue
            elif cost < visited[current]:
                visited[current] = cost
            # If cost == visited[current], we still allow multiple paths to this node

            if current[0] == max_layer:
                all_paths[(start[1], current[1])].append((cost, path))
                continue

            for neighbor, edge_cost in adjacency.get(current, {}).items():
                heapq.heappush(heap, (cost + edge_cost, neighbor, path + [neighbor]))

    return all_paths

def create_graph(all_paths, alpha_first, alpha_second, all_signs, dummy_alpha):
    """
    Constructs a directed graph based on solution dictionaries, then appends
    a dummy transition layer before reaching the final goal.

    Parameters:
      all_paths: list of dictionaries (one per real solution). Each dictionary
                 has keys (begin_idx, end_idx) mapping to lists of tuples (cost, path).
      alpha_first: list of lists; alpha_first[sol][i] is the alpha of the i-th beginning node in solution sol.
      alpha_second: list of lists; alpha_second[sol][j] is the alpha of the j-th ending node in solution sol.
      all_signs: list of sign layers. For each solution sol (0 to N-1):
                 - all_signs[sol][0] gives the sign vectors for the beginning nodes.
                 - all_signs[sol][-1] gives the sign vectors for the ending nodes.
                 The last element, all_signs[-1], is the dummy layer sign vectors (one per dummy node).
      dummy_alpha: list; dummy_alpha[i] is the alpha for the i-th dummy node.

    Returns:
      G: The resulting directed graph, where each edge is labeled with two cost attributes:
           - "cross_cost": cost for internal solution crossings.
           - "vib_cost": cost for solution-to-solution (transition) vibrations.
      positions: Dictionary mapping nodes to positions (for visualization).
      edge_labels: Dictionary mapping edge tuples (u,v) to a tuple (vib_cost, cross_cost).
    """
    G = nx.DiGraph()
    positions = {}
    edge_labels = {}

    num_real_sols = len(all_signs) - 1  # Last entry is for the dummy layer.

    # ---------------------------------------------------
    # Step 1: Create nodes for each real solution from sign layers.
    # ---------------------------------------------------
    for sol in range(num_real_sols):
        # Create begin nodes from the first layer.
        begin_signs = all_signs[sol][0]
        for i, sign in enumerate(begin_signs):
            node_name = f"{sol}_b_{i}"
            alpha_val = alpha_first[sol][i] if i < len(alpha_first[sol]) else None
            G.add_node(node_name, alpha=alpha_val, sign=sign, type='begin')
            positions[node_name] = (sol * 3, i)

        # Create end nodes from the last layer.
        end_signs = all_signs[sol][-1]
        for j, sign in enumerate(end_signs):
            node_name = f"{sol}_e_{j}"
            alpha_val = alpha_second[sol][j] if j < len(alpha_second[sol]) else None
            G.add_node(node_name, alpha=alpha_val, sign=sign, type='end')
            positions[node_name] = (sol * 3 + 2, j)

    # ---------------------------------------------------
    # Step 2: Add internal solution edges (begin -> end) from all_paths.
    # These edges represent the crossing cost.
    # ---------------------------------------------------
    for sol, sol_dict in enumerate(all_paths):
        for (begin_idx, end_idx), paths_list in sol_dict.items():
            cost, _ = paths_list[0]  # Only use the cost from the first path.
            begin_node = f"{sol}_b_{begin_idx}"
            end_node = f"{sol}_e_{end_idx}"
            # For internal edges: cross_cost = cost, vib_cost = 0.
            G.add_edge(begin_node, end_node, cross_cost=cost, vib_cost=0)
            edge_labels[(begin_node, end_node)] = (0, cost)

    # ---------------------------------------------------
    # Step 3: Add transitions between consecutive real solutions.
    # These edges represent vibration costs.
    # ---------------------------------------------------
    for sol in range(num_real_sols - 1):
        num_end_nodes = len(all_signs[sol][-1])
        num_begin_nodes_next = len(all_signs[sol + 1][0])
        for j in range(num_end_nodes):
            en = f"{sol}_e_{j}"
            sign_en = G.nodes[en]['sign']
            for i in range(num_begin_nodes_next):
                bn = f"{sol + 1}_b_{i}"
                sign_bn = G.nodes[bn]['sign']
                if count_sign_changes(sign_en, sign_bn) == 0:
                    # Transition edge: vib_cost = alpha from the next solution's begin node, cross_cost = 0.
                    linking_cost = G.nodes[bn]['alpha']
                    G.add_edge(en, bn, vib_cost=linking_cost, cross_cost=0)
                    edge_labels[(en, bn)] = (linking_cost, 0)

    # ---------------------------------------------------
    # Step 4: Create dummy layer nodes from dummy signs (all_signs[-1])
    # ---------------------------------------------------
    dummy_signs = all_signs[-1]  # This is a list of sign vectors (one per dummy node)
    num_dummy = len(dummy_signs)
    for i, sign in enumerate(dummy_signs):
        node_name = f"dummy_{i}"
        # Use the provided dummy_alpha for the cost.
        alpha_val = dummy_alpha[i] if i < len(dummy_alpha) else None
        G.add_node(node_name, alpha=alpha_val, sign=sign, type='dummy')
        # Place dummy nodes to the right of the last real solution.
        positions[node_name] = (num_real_sols * 3, i)

    # ---------------------------------------------------
    # Step 5: Add transitions from the last real solution's end nodes to dummy nodes.
    # These edges represent the vibration cost.
    # ---------------------------------------------------
    last_sol = num_real_sols - 1
    num_end_last = len(all_signs[last_sol][-1])
    for j in range(num_end_last):
        en = f"{last_sol}_e_{j}"
        sign_en = G.nodes[en]['sign']
        for i in range(num_dummy):
            dummy_node = f"dummy_{i}"
            sign_dummy = G.nodes[dummy_node]['sign']
            if count_sign_changes(sign_en, sign_dummy) == 0:
                # Transition edge: vib_cost = dummy node's alpha, cross_cost = 0.
                linking_cost = G.nodes[dummy_node]['alpha']
                G.add_edge(en, dummy_node, vib_cost=linking_cost, cross_cost=0)
                edge_labels[(en, dummy_node)] = (linking_cost, 0)

    # ---------------------------------------------------
    # Step 6: Connect dummy nodes to the final "Goal" node.
    # Here we assign zero cost so that the vibration cost is not double-counted.
    # ---------------------------------------------------
    goal_node = "Goal"
    G.add_node(goal_node)
    for i in range(num_dummy):
        dummy_node = f"dummy_{i}"
        # Set both vib_cost and cross_cost to zero.
        G.add_edge(dummy_node, goal_node, vib_cost=0, cross_cost=0)
        edge_labels[(dummy_node, goal_node)] = (0, 0)
    # Position the goal node at the far right.
    positions[goal_node] = ((num_real_sols) * 3 + 2, 0)

    return G, positions, edge_labels, 0

def assign_combined_weights(G, crossing_multiplier):
    """
    For each edge in G, assign a combined weight:

        combined_weight = vib_cost + crossing_multiplier * cross_cost

    The crossing_multiplier allows us to trade off between crossing cost (typically small)
    and vibration cost (typically large).
    """
    for u, v, data in G.edges(data=True):
        vib = data.get("vib_cost", 0)
        cross = data.get("cross_cost", 0)
        data["combined_weight"] = vib + crossing_multiplier * cross

def compute_path_costs(path, G):
    """
    Given a path (as a list of nodes) and graph G, computes:
      - Total vibration cost (sum of vib_cost),
      - Total crossing cost (sum of cross_cost), and
      - Total combined cost (sum of combined_weight).
    Returns a tuple (total_vib, total_cross, total_combined).
    """
    total_vib = 0
    total_cross = 0
    total_combined = 0
    for i in range(len(path) - 1):
        data = G.get_edge_data(path[i], path[i + 1])
        total_vib += data.get("vib_cost", 0)
        total_cross += data.get("cross_cost", 0)
        total_combined += data.get("combined_weight", 0)
    return total_vib, total_cross, total_combined

def find_k_best_paths(G, source, target, k, weight_attribute="combined_weight"):
    """
    Finds up to k best simple paths from source to target in graph G, using the specified weight attribute.
    Returns a list of tuples (path, total_weight).
    """
    paths = []
    try:
        generator = nx.shortest_simple_paths(G, source, target, weight=weight_attribute)
        for i, path in enumerate(generator):
            if i >= k:
                break
            tot_weight = sum(G.get_edge_data(path[j], path[j + 1]).get(weight_attribute, 0)
                             for j in range(len(path) - 1))
            paths.append((path, tot_weight))
    except nx.NetworkXNoPath:
        pass
    return paths

def visualize_paths(G, position, edge_labels, path=None):
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos=position, with_labels=True, node_size=1000, node_color='skyblue', font_size=6, font_weight='bold', arrows=True)
    if path is not None:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos=position, nodelist=path, node_color='lightgreen', node_size=1000)
        nx.draw_networkx_edges(G, pos=position, edgelist=path_edges, edge_color='red', width=2)
    nx.draw_networkx_edge_labels(G, pos=position, edge_labels=edge_labels, label_pos=0.5)

    plt.title("Shortest Path Problem with Multiple Solutions")
    # plt.show()

def calc_cost(index1, index2, omega_input, reference=0):
    alpha_begin = []
    alpha_end = []
    cost_begin = []
    cost_end = []
    for i in range(len(index1)):
        result = alpha_options(omega_input[:, index1[i]].reshape(4,1), reference)[0] # Check
        # alpha_begin.append(f"{index1[i]}")
        alpha_begin.append(result)
        cost = []
        for j in range(len(result)):
            step1 = omega_input[:, index1[i]].reshape(4, 1) - NULL_R * result[j]
            sum_omega_square1 = np.sum( step1 ** 2)
            cost.append(sum_omega_square1)
        cost_begin.append(cost)

    for i in range(len(index2)):
        result = alpha_options(omega_input[:, index2[i]].reshape(4,1),reference)[0]
        # alpha_begin.append(f"{index1[i]}")
        alpha_end.append(result)
        cost = []
        for j in range(len(result)):
            step1 = omega_input[:, index1[i]].reshape(4, 1) - NULL_R * result[j]
            sum_omega_square1 = np.sum(step1 ** 2)
            cost.append(sum_omega_square1)
        cost_end.append(cost)
    return alpha_begin, alpha_end, cost_begin, cost_end

def plot(n0=0, n=8005, path=None, scatter=True, limits=True, optimal=False):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    # ax.plot(time, segments.T, color='gray')
    ax.plot(time, alpha, color='black', linestyle='--', label='Ideal path')
    if scatter:
        for i in range(len(full_indices)):
            ax.scatter( np.ones_like(options[full_indices[i]])*time[full_indices[i]+falling[k]], options[full_indices[i]], color='b', zorder=5)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0, 8, 2):
        color = colors[i // 2 % len(colors)]  # Cycle through colors
        avg = (segments[i] + segments[i + 1]) / 2
        plt.plot(time, avg, color=color)
        plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    if path is not None:
        plt.plot(time[n0:n0+n], path, color='red', label='Optimal Path')
    plt.xlabel("Time (s)")
    plt.ylabel("Nullspace component")
    plt.title("Zero speed bands vs time")
    # plt.legend()
    # plt.show()

old_time = clock.time()

torque_data = load_data('Data/Slew1.mat')
low_torque_flag = hysteresis_filter(torque_data, 0.000005, 0.000015)
low_torque_flag[0:2] = False
rising, falling = detect_transitions(low_torque_flag)
falling = np.insert(falling, 0, 0)

momentum4_with_nullspace = pseudo_sol(torque_data)
alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:,0:1])
momentum3 = R @ momentum4_with_nullspace
momentum4 = R_PSEUDO @ momentum3
sections = []
if len(rising) == len(falling):
    for i in range(len(rising)):
        sections.append(momentum4[:, falling[i]:rising[i]])
else:
    print("Error: rising and falling edges do not match")
    print(rising, falling)
    exit()

time = np.linspace(0, 800, 8005)
# alpha = nullspace_alpha(momentum4)
alpha, alpha_ref = np.zeros_like(time, dtype=int), 0

segments = calc_segments(momentum4)
all_options = []
all_indices = []
all_signs = []
all_adj = []
all_equal_paths = []

for k in range(len(sections)):
    options = alpha_options(sections[k],alpha_ref)
    full_indices = calc_indices(options)
    signs = alpha_to_sign(options, sections[k], full_indices)
    adj = build_adjacency_dict(signs)

    # all_paths = find_all_shortest_paths_from_adjacency(adj)
    equal_paths = find_all_equal_shortest_paths(adj)
    # pprint(equal_paths)

    # plot(scatter=True)
    # plt.show()
    all_options.append(options)
    all_indices.append(full_indices)
    all_signs.append(signs)
    all_adj.append(adj)
    all_equal_paths.append(equal_paths)

options = alpha_options(momentum4[:,-1].reshape(4,1),alpha_ref)
signs = alpha_to_sign(options, momentum4[:,-1].reshape(4,1), [0])
all_signs.append(signs[0])
alpha_rising, alpha_falling, cost_rising, cost_falling = calc_cost(rising, falling, momentum4, reference=alpha_ref)
alpha_end, _, cost_end, _ = calc_cost([-1], [], momentum4, reference=alpha_ref)


plot(scatter=False)

# graph, positions, labels, nodes = create_graph(all_equal_paths, alpha_falling, alpha_rising, all_signs, alpha_end[0])
graph, positions, labels, nodes = create_graph(all_equal_paths, cost_falling, cost_rising, all_signs, cost_end[0])
assign_combined_weights(graph, crossing_multiplier=0)
shortest_path = nx.dijkstra_path(graph, source='0_b_2', target='Goal', weight='combined_weight')
print("Path to goal:", shortest_path)
visualize_paths(graph, positions, labels, path=shortest_path)

# visualize_paths(graph, positions, labels)
source = "0_b_2"
target = "Goal"
best_paths = find_k_best_paths(graph, source, target, k=10, weight_attribute="combined_weight")

print("Best path(s) found optimizing for vibration cost only (ignoring crossings):\n")
for idx, (path, tot_weight) in enumerate(best_paths):
    vib, cross, comb = compute_path_costs(path, graph)
    print(f"Path {idx+1}: {path}")
    print(f"  Total Vibration Cost: {vib}")
    print(f"  Total Crossing Cost:  {cross}")
    print(f"  Total Combined Cost:  {comb}\n")

new_time = clock.time()
print("Time taken: ", new_time - old_time, "seconds")

plt.show()








