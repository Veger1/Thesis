import numpy as np
import casadi as ca
from config import *
from helper import *
from scipy.io import loadmat
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

def stitch_sections(all_equal_paths, start_idx, transition_costs, final_costs):
    num_sections = len(all_equal_paths)
    goal_node = 'goal'

    # Initial meta-graph connections
    meta_adj = {}
    found_first_paths = False

    # SECTION 0 — Only paths starting from start_idx
    for (s_idx, e_idx), path_list in all_equal_paths[0].items():
        if s_idx == start_idx:
            for cost, path in path_list:
                meta_adj[(f'section_0', e_idx)] = {'cost': cost, 'path': path}
                found_first_paths = True

    if not found_first_paths:
        print(f"[ERROR] No paths found starting from node {start_idx} in section 0.")
        return None

    # INTERMEDIATE SECTIONS
    for k in range(1, num_sections):
        current_paths = all_equal_paths[k]
        next_meta_adj = {}

        for (prev_sec, prev_end_idx), data in meta_adj.items():
            if not prev_sec.startswith(f'section_{k-1}'):
                continue
            for (s_idx, e_idx), path_list in current_paths.items():
                if s_idx != prev_end_idx:
                    continue
                for cost, path in path_list:
                    total_cost = data['cost'] + transition_costs[k-1][s_idx] + cost
                    new_path = data['path'] + path[1:]  # skip repeated node
                    next_meta_adj[(f'section_{k}', e_idx)] = {
                        'cost': total_cost,
                        'path': new_path
                    }

        if not next_meta_adj:
            print(f"[ERROR] No valid transitions from section {k-1} to {k}.")
            return None

        meta_adj = next_meta_adj

    # FINAL STEP: Connect to goal
    final_paths = {}
    for (sec_name, end_idx), data in meta_adj.items():
        total_cost = data['cost'] + final_costs[end_idx]
        final_paths[(sec_name, end_idx)] = {
            'cost': total_cost,
            'path': data['path'] + [goal_node]
        }

    if not final_paths:
        print("[ERROR] No valid final connections to goal.")
        return None

    # Choose shortest global path
    best = min(final_paths.values(), key=lambda x: x['cost'])
    return best

def visualize_paths(all_paths, num_solutions):
    G = nx.DiGraph()  # Directed graph
    position = {}  # To store positions of nodes for visualization
    edge_labels = {}  # To store edge labels (costs)
    added_nodes = set()  # Track added nodes

    # Nodes and edges for each solution
    solution_counter = 1
    solution_nodes = []  # To track the nodes of each solution
    for (start_idx, end_idx), paths in all_paths.items():

        for path in paths:
            cost, path_nodes = path
            solution_name = f"Solution {solution_counter}"

            # Add nodes for the solution
            start_node = f"{solution_name} start ({start_idx})"
            end_node = f"{solution_name} end ({end_idx})"
            if start_node not in added_nodes:
                G.add_node(start_node)
                added_nodes.add(start_node)
                position[start_node] = (0, start_idx)
            if end_node not in added_nodes:
                G.add_node(end_node)
                added_nodes.add(end_node)
                position[end_node] = (1, end_idx)


            # Add edges (start to end in the same solution)
            G.add_edge(start_node, end_node, weight=cost)
            edge_labels[(start_node, end_node)] = cost

            solution_nodes.append((start_node, end_node))

            # Transition edges between solutions
            if solution_counter < num_solutions:
                next_solution_name = f"Solution {solution_counter + 1}"
                next_start_node = f"{next_solution_name} start ({start_idx})"
                G.add_edge(end_node, next_start_node, weight=cost)
                edge_labels[(end_node, next_start_node)] = cost

                solution_counter += 1


    # Draw the graph
    plt.figure(figsize=(12, 6))
    nx.draw(G, pos=position, with_labels=True, node_size=2000, node_color='skyblue', font_size=10, font_weight='bold', arrows=True)
    nx.draw_networkx_edge_labels(G, pos=position, edge_labels=edge_labels, label_pos=0.15)

    plt.title("Shortest Path Problem with Multiple Solutions")
    # plt.show()

def compute_constraints(wheel_signs_before, wheel_signs_after, bands, position=0):
    N = bands.shape[1]
    min_constraints = np.full(N, -np.inf)
    max_constraints = np.full(N, np.inf)
    crossings_max = []
    crossings_min = []
    for i in range(len(wheel_signs_before)):
        band_idx = 2 * i  # Row index corresponding to the wheel's bands
        band_max = np.max(bands[band_idx:band_idx + 2], axis=0)
        band_min = np.min(bands[band_idx:band_idx + 2], axis=0)
        if wheel_signs_before[i] == wheel_signs_after[i]:  # Sign change detected
            if position < band_max[0]:
                max_constraints = np.minimum(max_constraints, band_min)

            if position > band_min[0]:
                min_constraints = np.maximum(min_constraints, band_max)
        else:
            crossings_max.append(band_max)
            crossings_min.append(band_min)


    return min_constraints, max_constraints, crossings_min, crossings_max

def find_crossings(signals):
    num_signals, N = signals.shape
    crossings = {}

    for i in range(num_signals):
        for j in range(i+1, num_signals):
            diff = signals[i] - signals[j]
            crossing_indices = np.where(np.diff(np.sign(diff)) != 0)[0]
            crossings[(i, j)] = crossing_indices

    return crossings

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

momentum4 = pseudo_sol(torque_data)
momentum3 = R @ momentum4
momentum0 = R_PSEUDO @ momentum3
sections = []
if len(rising) == len(falling):
    for i in range(len(rising)):
        sections.append(momentum4[:, falling[i]:rising[i]])
else:
    print("Error: rising and falling edges do not match")
    print(rising, falling)
    exit()

time = np.linspace(0, 800, 8005)
alpha = nullspace_alpha(momentum4)

segments = calc_segments(momentum4)
all_options = []
all_indices = []
all_signs = []
all_adj = []
all_equal_paths = []

for k in range(len(sections)):
    options = alpha_options(sections[k],alpha[0])
    full_indices = calc_indices(options)
    signs = alpha_to_sign(options, sections[k], full_indices)
    adj = build_adjacency_dict(signs)

    # all_paths = find_all_shortest_paths_from_adjacency(adj)
    equal_paths = find_all_equal_shortest_paths(adj)
    pprint(equal_paths)

    plot(scatter=True)
    visualize_paths(equal_paths, 1)
    plt.show()

    all_options.append(options)
    all_indices.append(full_indices)
    all_signs.append(signs)
    all_adj.append(adj)
    all_equal_paths.append(equal_paths)


# Example values
start_idx = 0
transition_costs = [
    [1, 10, 10, 10, 100],  # From section 0 to 1
    [1, 10, 10, 10, 100],  # From section 1 to 2
    [1, 10, 10, 10, 100],  # From section 2 to 3
]
final_costs = [2, 200, 200, 200, 200]  # From section 3 to goal

result = stitch_sections(all_equal_paths, start_idx, transition_costs, final_costs)

print("Total cost:", result['cost'])
print("Full stitched path:")
for node in result['path']:
    print(node)

new_time = clock.time()
print("Time taken: ", new_time - old_time)








