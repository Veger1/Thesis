from config import *
from helper import *


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

class NodeOption:
    def __init__(self, alpha, base_speed, node_type="normal"):
        self.alpha = alpha
        self.type = node_type  # 'begin', 'end', 'dummy' or 'normal'
        self.config = None
        self.vibration_cost = None
        self.wheel_speeds = base_speed - NULL_R * alpha
        self.signs = np.sign(self.wheel_speeds)

    def populate_node(self):
        pass

    def __repr__(self):
        return f"NodeOption(alpha={self.alpha}, sign={self.signs}, type={self.type})"

class Layer:
    def __init__(self, time_index, layer_type="normal", alpha_options_list=None, global_start_idx=0):
        self.time_index = time_index
        self.layer_type = layer_type
        self.wheel_speeds = momentum4[:, self.time_index]
        self.options = []
        self.alpha_options_list = alpha_options_list  # List of lists
        self.global_start_idx = global_start_idx
        self.populate_layer()

    def populate_layer(self):
        print("populating layer")
        if self.alpha_options_list is None:
            return
        local_index = self.time_index - self.global_start_idx
        print("Length:", len(self.alpha_options_list), "index", local_index)
        if 0 <= local_index < len(self.alpha_options_list):
            print("yes")
            alphas = self.alpha_options_list[local_index]
            for alpha in alphas:
                option = NodeOption(alpha=alpha, base_speed =self.wheel_speeds)
                self.add_option(option)

    def add_option(self, option: NodeOption):
        self.options.append(option)

    def __repr__(self):
        return f"Layer(t={self.time_index}, options={len(self.options)})"


class Section:
    def __init__(self, name, start_idx, end_idx, stationary_idx):
        self.name = name
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.stationary_idx = stationary_idx
        self.alpha_options = alpha_options(momentum4[:, start_idx:end_idx])
        self.layers = []
        self.begin_layer = Layer(start_idx, "begin", self.alpha_options, start_idx)
        self.end_layer = Layer(end_idx, "end", self.alpha_options, start_idx)
        self.stationary_layer = Layer(stationary_idx, "stationary", self.alpha_options, start_idx)

    def populate_section(self, start_idx, end_idx, stationary_idx):
        mid_idx = calc_indices(self.alpha_options)
        self.add_layer(self.begin_layer)
        for j in range(len(mid_idx)):
            layer = Layer(mid_idx[j]+ start_idx, "normal", self.alpha_options, start_idx)
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
    # print(f"Start index: {start_index}, End index: {end_index}")
    stationary_index = falling[i+1]-1
    section_name = f"Section_{i+1}"
    section = Section(section_name, start_index, end_index, stationary_index)
    section.populate_section(start_index, end_index, stationary_index)
    problem.append(section)
