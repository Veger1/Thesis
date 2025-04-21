from rockit import Ocp, MultipleShooting
from casadi import sum1
from config import *
from helper import *
import networkx as nx
import matplotlib.pyplot as plt
from itertools import islice


def calc_segments(omega_input):
    length = omega_input.shape[1]
    segments_result = np.zeros((8, length))  # Constraints: 4 bands
    for j in range(length):
        omega = momentum4[:, j]
        for i in range(4):
            segments_result[2 * i, j] = (OMEGA_MIN - omega[i]) * (-1) ** i
            segments_result[1 + 2 * i, j] = (-OMEGA_MIN - omega[i]) * (-1) ** i
    return segments_result

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
    for j in range(len(change_indices) - 1):
        start = change_indices[j]
        end = change_indices[j + 1]
        # Only include midpoint if the second region has *more* options than the first
        if options_count[start] > options_count[end]:
            mid_indices.append((start + end) // 2)
    # mid_indices = [(change_indices[i] + change_indices[i+1]) // 2 for i in range(len(change_indices) - 1)]
    return mid_indices

def build_graph(sections):
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
            # node_id = 0
        if i == len(sections) - 1:
            for node in section.stationary_layer.nodes:
                node.id = f"{section.name}_{section.stationary_layer.time_index}_{node_id}"
                node.display_name = f"{node_id}"
                G.add_node(node.id, node=node)
                node_id += 1
    goal_node = NodeOption(alpha=0, base_speed=0, node_type="goal")
    goal_node.id, goal_node.display_name = "goal_8100_0", "goal"
    G.add_node(goal_node.id, node=goal_node)

    # Pass 2: Add all edges
    for i, section in enumerate(sections):
        all_layers = section.layers
        for l1, l2 in zip(all_layers[:-1], all_layers[1:]):
            for n1 in l1.nodes:
                for n2 in l2.nodes:
                    sign_diff = sum(s1 != s2 for s1, s2 in zip(n1.signs, n2.signs))
                    G.add_edge(n1.id, n2.id, sign_cost=sign_diff, vibration_cost=0)

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

def shortest_path(G, source_idx, target_idx, cost_type="sign_cost"):
    best_path = None
    best_cost = float("inf")
    best_sign_cost = None
    best_vibration_cost = None

    for source in source_ids:
        for target in target_ids:
            try:
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

def get_coord(node_id):
    parts = node_id.split('_')
    try:
        layer = int(parts[-2])  # second to last part = layer
        index = int(parts[-1])  # last part = index
        return layer, index
    except Exception as e:
        print(f"Error parsing node ID '{node_id}': {e}")
        return 0, 0

def k_shortest_paths(G, source_idx, target_idx, k, cost_type="sign_cost"):
    all_paths = []

    for source in source_ids:
        for target in target_ids:
            try:
                paths_gen = nx.shortest_simple_paths(G, source, target, weight=cost_type)
                for temp_path in islice(paths_gen, k):
                    temp_mixed_cost = nx.path_weight(G, temp_path, weight=cost_type)
                    temp_sign_cost = nx.path_weight(G, temp_path, weight="sign_cost")
                    temp_vibration_cost = nx.path_weight(G, temp_path, weight="vibration_cost")
                    all_paths.append((path, temp_mixed_cost, temp_sign_cost, temp_vibration_cost))
            except nx.NetworkXNoPath:
                continue

    # Sort all found paths by the mixed cost and return top k
    all_paths = sorted(all_paths, key=lambda x: x[1])[:k]
    return all_paths

def assign_mixed_costs(G, penalty=0):
    for u, v, data in G.edges(data=True):
        temp_sign_cost = data.get("sign_cost", 0)
        temp_vibration_cost = data.get("vibration_cost", 0)
        mixed = temp_vibration_cost + penalty * temp_sign_cost
        data["mixed_cost"] = mixed

def plot_shortest_path(graph, path, get_coordinates=None, title="Shortest Path Graph"):
    pos = {}
    xs = []
    for node in graph.nodes():
        if get_coordinates:
            x, y = get_coordinates(node)
        else:
            x, y = 0, 0
        pos[node] = (x, y)  # Flip y for nicer top-down plotting
        xs.append(x)

    plt.figure(figsize=(12, 6))
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

    plt.title(title)
    plt.axis('off')
    plt.xlabel("Time index or X coordinate")
    plt.tight_layout()
    plt.show()

def extract_info_from_path(graph, path):
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
    if last_data is None and len(path) >= 2:  # USeless code maybe
        print("Used")
        # If last is goal node, look one step back
        t, alpha = get_node_data(path[-2])
        if t is not None and alpha is not None:
            print("Appended")
            results.append((t, alpha))

    return results

def build_alpha_constraints(alpha_nodes, bands):
    N = bands.shape[1]
    min_constraints = np.full(N, -np.inf)
    max_constraints = np.full(N, np.inf)

    for j in range(len(alpha_nodes) - 1):
        t_start, alpha_start, signs_start = alpha_nodes[j]
        t_end, alpha_end, signs_end = alpha_nodes[j + 1]

        for i in range(4):  # For each wheel
            if signs_start[i] != signs_end[i]:
                continue  # Skip if the band is crossed (sign changes)

            band_idx = 2 * i  # Index for band (each wheel has 2 rows)
            band_max = np.max(bands[band_idx:band_idx + 2, t_start:t_end], axis=0)
            band_min = np.min(bands[band_idx:band_idx + 2, t_start:t_end], axis=0)

            alpha_val = alpha_start

            if alpha_val < np.mean([band_max[0], band_min[0]]):  # Band is above alpha → it's a max constraint
                max_constraints[t_start:t_end] = np.minimum(max_constraints[t_start:t_end], band_min)

            elif alpha_val > np.mean([band_max[0], band_min[0]]):  # Band is below alpha → it's a min constraint
                min_constraints[t_start:t_end] = np.maximum(min_constraints[t_start:t_end], band_max)

    return min_constraints, max_constraints

def plot(n0=0, n=8005, path=None, scatter=False, limits=None, null_path=None):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    time = np.linspace(0, 800, 8005)
    zeros = np.zeros(8005)
    # ax.plot(time, zeros, color='black', linestyle='--', label='Ideal path')
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0, 8, 2):
        color = colors[i // 2 % len(colors)]  # Cycle through colors
        avg = (segments[i] + segments[i + 1]) / 2
        plt.plot(time, avg, color=color)
        plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    for t, alpha, signs in alpha_points:
        if scatter:
            plt.scatter(time[t], alpha, color='red', marker='o')
            plt.scatter(0, alpha_nullspace, color='blue', marker='x')
    if limits is not None:
        min_limit, max_limit = limits
        plt.plot(time, max_limit, color='black', linestyle='--', label='Max Constraint')
        plt.plot(time, min_limit, color='black', label='Min Constraint')
    if null_path is not None:
        N = null_path.shape[0]
        plt.plot(time[0:N], null_path, color='blue', linestyle='--', label='Nullspace Path')

    plt.xlabel("Time (s)")
    plt.ylabel("Nullspace component")
    plt.title("Zero speed bands vs time")
    plt.legend()
    plt.show()

def solver(begin_alpha, limits=None, guess=None, N=200):
    if limits is None:
        return
    min_constraint, max_constraint = limits
    temp_time = np.linspace(0, 800, 8005)
    ocp = Ocp(t0=0, T=temp_time[100])
    w = ocp.state(4)
    alpha_control = ocp.control()
    T_sc = ocp.parameter(3, grid='control')

    ocp.set_value(T_sc, torque_data[:,0:N])

    T_rw = R_PSEUDO @ T_sc + NULL_R @ alpha_control
    der_state = I_INV @ T_rw
    ocp.set_der(w, der_state)

    alpha_null = ocp.variable(grid='control')
    ocp.subject_to(alpha_null == (-w[0] + w[1] - w[2] + w[3]) / 4)
    # alpha_null = (-w[0] + w[1] - w[2] + w[3]) / 4

    w_initial = OMEGA_START
    alpha_min_constraint = ocp.parameter(grid='control')
    alpha_max_constraint = ocp.parameter(grid='control')
    ocp.set_value(alpha_min_constraint, min_constraint[0:N])
    ocp.set_value(alpha_max_constraint, max_constraint[0:N])

    ocp.subject_to(-MAX_TORQUE <= (T_rw <= MAX_TORQUE))  # Add torque constraints
    ocp.subject_to(-OMEGA_MAX <= (w <= OMEGA_MAX))  # Add saturation constraints
    ocp.subject_to(alpha_min_constraint <= (alpha_null <= alpha_max_constraint))  # Add saturation constraints
    ocp.subject_to(ocp.at_t0(w) == w_initial)
    ocp.set_initial(w, w_initial)  # Set initial guess

    # ocp_t = ocp.t
    # w_sym, t_sym = symbols('w t')
    a = 0.1
    b = 1e-4
    # objective_expr = exp(-a * w ** 2)  # Gaussian function
    # objective_expr_casadi = lambdify((w_sym, t_sym), objective_expr, 'numpy')
    # objective_expr_casadi = objective_expr_casadi(w, ocp_t)
    objective_expr_casadi = np.exp(-a * w ** 2)
    objective_expr_casadi = b*w**2
    objective = ocp.integral(sum1(objective_expr_casadi))
    ocp.add_objective(objective)

    ocp.solver('ipopt', SOLVER_OPTS)  # Use IPOPT solver
    ocp.method(MultipleShooting(N=N, M=1, intg='rk'))
    sol = ocp.solve()  # Solve the problem

    # Post-processing: Sample solutions for this interval
    ts, w_sol = sol.sample(w, grid='control')
    _, alpha_sol = sol.sample(alpha, grid='control')
    _, T_rw_sol = sol.sample(T_rw, grid='control')
    _, alpha_null_sol = sol.sample(alpha_null, grid='control')

    return w_sol, alpha_sol, T_rw_sol, alpha_null_sol

def find_start_node(sections, null_alpha, base_speed):
    # desired_signs = np.sign((base_speed - (NULL_R @ null_alpha.reshape(1,1))).flatten())
    desired_signs = np.sign(base_speed).flatten()
    first_section = sections[0]
    first_begin_layer = first_section.begin_layer
    print("Desired signs:", desired_signs)
    for node in first_begin_layer.nodes:
        print("Node:", node)
        # print(node.signs)
        if np.array_equal(node.signs, desired_signs):
            print("Match found")
            return [f"{first_section.name}_{node.id}"]
    return None  # No matching node found

class NodeOption:
    def __init__(self, alpha, base_speed, node_type="mid", layer_idx=None, local_id=None):
        self.alpha = alpha
        self.type = node_type  # 'begin', 'end', 'stationary' or 'mid'
        self.time_index = layer_idx
        self.layer_type = node_type
        self.wheel_speeds = base_speed - (NULL_R * alpha).flatten()
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
    def __init__(self, time_index, layer_type="normal"):
        self.time_index = time_index
        self.layer_type = layer_type
        self.wheel_speeds = momentum4[:, self.time_index]
        self.nodes = []
        self.alphas = None
        self.populate_layer()

    def populate_layer(self):
        if options is None:
            return
        self.alphas = options[self.time_index]
        for k, alpha in enumerate(self.alphas):
            node = NodeOption(alpha=alpha, base_speed=self.wheel_speeds, layer_idx=self.time_index, local_id=k)
            self.add_node(node)

    def add_node(self, node: NodeOption):
        self.nodes.append(node)

    def __repr__(self):
        return f"Layer(t={self.time_index}, options={len(self.nodes)})"

class Section:
    def __init__(self, name, start_idx, end_idx, stationary_idx):
        self.name = name
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.stationary_idx = stationary_idx
        self.layers = []
        self.alpha_options = options[start_idx:stationary_idx]
        self.begin_layer = Layer(start_idx, "begin")
        self.end_layer = Layer(end_idx, "end")
        self.stationary_layer = Layer(stationary_idx, "stationary")  # Not in the layers list

    def populate_section(self, start_idx, end_idx, stationary_idx):
        mid_idx = calc_indices(self.alpha_options)
        self.add_layer(self.begin_layer)
        for j in range(len(mid_idx)):
            layer = Layer(mid_idx[j]+ start_idx, "normal")
            self.add_layer(layer)
        self.add_layer(self.end_layer)


    def add_layer(self, layer: Layer):
        self.layers.append(layer)


    def __repr__(self):
        return f"Section({self.name}, layers={len(self.layers)})"


torque_data = load_data('Data/Slew1.mat')
low_torque_flag = hysteresis_filter(torque_data, 0.000005, 0.000015)
low_torque_flag[0:2] = False
rising, falling = detect_transitions(low_torque_flag)
falling = np.insert(falling, 0, 0)
falling = np.append(falling, len(low_torque_flag)+1)

momentum4_with_nullspace = pseudo_sol(torque_data)
alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:,0:1])
momentum3 = R @ momentum4_with_nullspace
momentum4 = R_PSEUDO @ momentum3
segments = calc_segments(momentum4)

problem = []
options = alpha_options(momentum4)
for i, end_index in enumerate(rising):
    start_index = falling[i]
    stationary_index = falling[i+1]-1
    section_name = f"Section_{i+1}"
    section = Section(section_name, start_index, end_index, stationary_index)
    section.populate_section(start_index, end_index, stationary_index)
    problem.append(section)

# source_ids = ["Section_1_0_3"]
source_ids = find_start_node(problem, alpha_nullspace, OMEGA_START)
print("Source IDs:", source_ids)
target_ids = ["goal_8100_0"]
graph = build_graph(problem)
assign_mixed_costs(graph, penalty=50000)
path, mixed_cost, sign_cost, vib_cost = shortest_path(graph, source_ids, target_ids, cost_type="mixed_cost")
# plot_shortest_path(graph, path, get_coordinates=get_coord, title="Graph of the Shortest Path")

# k_paths = k_shortest_paths(graph, source_ids, target_ids, k=5, cost_type="mixed_cost")
# for i, (path, mixed, sign, vib) in enumerate(k_paths):
#     print(f"Path {i + 1}:")
#     print(f"  Nodes: {path}")
#     print(f"  Mixed Cost: {mixed}")
#     print(f"  Sign Cost: {sign}")
#     print(f"  Vibration Cost: {vib}")

alpha_points = extract_info_from_path(graph, path)
for t, alpha, signs in alpha_points:
    print(f"Time index: {t}, Alpha: {alpha}", f"Signs: {signs}")

print("Initial alpha:", alpha_nullspace)
print("omega start:", OMEGA_START)
alpha_min, alpha_max = build_alpha_constraints(alpha_points, segments)
plot(scatter=True, limits=(alpha_min,alpha_max), null_path=None)
try:
    omega_sol, _, _, null_solution =solver(alpha_points, limits=(alpha_min, alpha_max))
except Exception as e:
    print("Solver failed:", e)
    omega_sol = None
    null_solution = None
# print("intial omega:", OMEGA_START)
# print("initial omega:", omega_sol[0,:])
# print("nullspace:", null_solution[0])

plot(scatter=True, limits=(alpha_min,alpha_max), null_path=null_solution)
