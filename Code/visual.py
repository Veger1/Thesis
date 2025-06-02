import os
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.io import loadmat
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from sympy import lambdify, symbols, sympify
from config import *
from helper import *
from realtime import overlap_constraint, optimal_alpha, line_constraint, constrained_alpha
from DAC import calc_segments, alpha_bounds
import matplotlib.patches as mpatches


def plot_cost_function(loaded_data):
    if 'cost_graph' in loaded_data:
        cost_graph = loaded_data['cost_graph'].flatten()
        omega_axis = loaded_data['omega_axis'].flatten()  # Could be changed with .flatten()
        plt.plot(omega_axis, cost_graph, 'r-', label="Cost Function")
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Cost')
        plt.title('Cost vs Angular Velocity')
        plt.show()


def plot_cost(loaded_data):
    if 'cost' in loaded_data:
        cost = loaded_data['cost']
        time = loaded_data['all_t'].flatten()
        plt.plot(time, cost)
        plt.xlabel('Time')
        plt.ylabel('Cost')
        plt.title('Cost vs Time')
        plt.show()


def plot_total_cost(loaded_data):
    if 'total_cost' in loaded_data:
        total_cost = loaded_data['total_cost'].flatten()
        time = loaded_data['all_t'].flatten()
        plt.plot(time, total_cost)
        plt.xlabel('Time')
        plt.ylabel('Total Cost')
        plt.title('Total Cost vs Time')
        plt.show()


def plot_rpm(loaded_data):
    if 'all_w_sol' in loaded_data:
        all_w_sol = loaded_data['all_w_sol']
        all_t = loaded_data['all_t'].flatten()
        plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
        plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
        plt.fill([all_t[0], all_t[0], all_t[-1], all_t[-1]],[-300, 300, 300, -300], 'r', alpha=0.1)
        plt.plot(all_t, rad_to_rpm(all_w_sol))
        plt.xlabel('Time (s)')
        plt.ylabel('RPM')
        plt.title('RPM vs Time')
        plt.show()


def plot_radians(loaded_data):
    if 'all_w_sol' in loaded_data:
        all_w_sol = loaded_data['all_w_sol'].T
        # all_t = loaded_data['all_t'].flatten()
        all_t  = np.linspace(0, all_w_sol.shape[1]/10, all_w_sol.shape[1])
        plt.axhline(y=600, color='r', linestyle='--', label=f'rad/s=600')
        plt.axhline(y=-600, color='r', linestyle='--', label=f'rad/s=-600')
        plt.fill([all_t[0], all_t[0], all_t[-1], all_t[-1]],[-30, 30, 30, -30], 'r', alpha=0.1)
        plt.plot(all_t, all_w_sol.T)
        # plt.ylim([-300, 300])
        plt.xlabel('Time (s)')
        plt.ylabel('Rad/s')
        plt.title('Rad/s vs Time')
        # plt.axvspan(68.9, 200.3, color='gray', alpha=0.2)
        # plt.axvspan(269, 400, color='gray', alpha=0.2)
        # plt.axvspan(510, 600, color='gray', alpha=0.2)
        # plt.axvspan(710, 800, color='gray', alpha=0.2)
        plt.show()


def plot_torque(loaded_data):
    if 'all_T_sol' in loaded_data:
        all_T_sol = loaded_data['all_T_sol']
        all_t = loaded_data['all_t'].flatten()
        plt.plot(all_t, all_T_sol)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Torque vs Time')
        plt.show()


def live_cost_plot_old(loaded_data, output_file=None):
    # Extract data from the dataset
    omega_axis = loaded_data['omega_axis'].flatten()
    cost_graph = loaded_data['cost_graph'].flatten()
    all_w_sol = loaded_data['all_w_sol']
    cost = loaded_data['cost']
    total_cost = loaded_data['total_cost'].flatten()
    cost_expr = loaded_data['cost_expr']
    time = np.linspace(0, 800, all_w_sol.shape[0])

    # Set up the figure and axes
    fig, (ax, ax_total_cost) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
    # ax.set_xlim(-300, 300)
    ax.set_ylim(min(cost_graph)-0.1, max(cost_graph)+0.1)
    ax.set_ylabel('Cost')
    ax.set_xlabel('Angular Velocity (rad/s)')
    ax.set_title(f'Cost Expr: {cost_expr}')
    ax.plot(omega_axis, cost_graph, 'r-', label="Cost Function")  # Cost function line

    # Initialize live points for each column
    colors = ['red', 'green', 'orange', 'blue']
    points = [ax.plot([], [], 'o', color=colors[i], label='Live point')[0] for i in range(all_w_sol.shape[1])]

    # Add a time display
    time_text = ax.text(0.05, 0.95, 'Time: 0.00 s', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Initialize total cost bar plot
    ax_total_cost.set_ylim(np.min(total_cost)-0.1, np.max(total_cost)+0.1)
    bar = ax_total_cost.bar(0, total_cost[0], color='blue')  # Initial bar
    ax_total_cost.set_xticks([])
    ax_total_cost.set_title("Total Cost")

    # Initialization function
    def init():
        for point in points:
            point.set_data([], [])
        time_text.set_text('Time: 0.00 s')
        bar[0].set_height(total_cost[0])
        return points + [time_text, bar[0]]

    # Update function for each frame
    def update(frame):
        for i, point in enumerate(points):
            point.set_data([all_w_sol[frame, i]], [cost[frame, i]])
        time_text.set_text(f'Time: {time[frame]:.2f} s')
        bar[0].set_height(total_cost[frame])
        return points + [time_text, bar[0]]

    # Animate
    ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=5)
    if output_file:
        Writer = writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(output_file, writer=writer)

    # Show the plot
    plt.tight_layout()
    plt.show()


def live_cost_plot(loaded_data, output_file=None):
    # Extract data from the dataset
    omega_axis = loaded_data['omega_axis'].flatten()
    all_w_sol = loaded_data['all_w_sol']
    cost = loaded_data['cost']
    total_cost = loaded_data['total_cost'].flatten()
    cost_expr = sympify(loaded_data['cost_expr'])
    #time = np.linspace(0, 800, all_w_sol.shape[0])
    time = loaded_data['all_t'].flatten()

    # Define symbolic variables
    w, t = symbols('w t')
    time_dependent = False
    # Check if cost_expr depends on time
    if t in cost_expr.free_symbols:
        time_dependent = True
        cost_func = lambdify((w, t), cost_expr, 'numpy')
        cost_graph_func = lambda t_val: cost_func(omega_axis, t_val)
        cost_graph = cost_graph_func(time[0]).flatten()  # Initialize for t = 0
    else:
        cost_func = lambdify(w, cost_expr, 'numpy')
        cost_graph = loaded_data['cost_graph'].flatten()  # Static cost graph

    # Set up the figure and axes
    fig, (ax, ax_total_cost) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
    ax.set_ylabel('Cost')
    ax.set_xlabel('Angular Velocity (rad/s)')
    ax.set_title(f'Cost Expr: {cost_expr}')

    # Plot initial cost function line
    cost_line, = ax.plot(omega_axis, cost_graph, 'r-', label="Cost Function")

    # Initialize live points for each column
    colors = ['red', 'green', 'orange', 'blue']
    points = [ax.plot([], [], 'o', color=colors[i])[0] for i in range(all_w_sol.shape[1])]

    # Add a time display
    time_text = ax.text(0.05, 0.95, 'Time: 0.00 s', transform=ax.transAxes, fontsize=12, verticalalignment='top')

    # Initialize total cost bar plot
    ax_total_cost.set_ylim(np.min(total_cost)-0.1, np.max(total_cost)+0.1)
    bar = ax_total_cost.bar(0, total_cost[0], color='blue')
    ax_total_cost.set_xticks([])
    ax_total_cost.set_title("Total Cost")

    # Initialization function
    def init():
        for point in points:
            point.set_data([], [])
        time_text.set_text('Time: 0.00 s')
        bar[0].set_height(total_cost[0])
        return points + [time_text, bar[0], cost_line]

    # Update function for each frame
    def update(frame):
        # Update live points
        for i, point in enumerate(points):
            point.set_data([all_w_sol[frame, i]], [cost[frame, i]])

        # Update cost function dynamically if it depends on time
        if time_dependent:
            cost_line.set_ydata(cost_graph_func(time[frame] % 200))

        # Update time and total cost display
        time_text.set_text(f'Time: {time[frame]:.2f} s')
        bar[0].set_height(total_cost[frame])

        return points + [time_text, bar[0], cost_line]

    # Animate
    ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=5)
    if output_file:
        Writer = writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(output_file, writer=writer)

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_cost_time(loaded_data):
    if 'cost' in loaded_data:
        cost = loaded_data['cost']
        time = loaded_data['all_t'].flatten()
        plt.plot(time, cost)
        plt.xlabel('Time')
        plt.ylabel('Cost')
        plt.title('Cost vs Time')
        plt.show()


def plot_MPI():
    mpi_data = loadmat('Data/50s/MPI.mat')
    w_sol = mpi_data['all_w_sol']
    mpi_data = loadmat('Data/50s/stic1.mat')
    plt.figure(figsize=(10, 5))
    plt.plot(mpi_data['all_t'].flatten(), w_sol[:8000, :])
    plt.axhline(y=OMEGA_MAX, color='black', linestyle='--')
    plt.axhline(y=-OMEGA_MAX, color='black', linestyle='--', label='Saturation')
    plt.fill([0, 0, 800, 800], [-OMEGA_MIN, OMEGA_MIN, OMEGA_MIN, -OMEGA_MIN], 'r', alpha=0.3, label="Stiction zone")
    # plt.ylim([-6000, 6000])
    plt.xlabel('Time (s)')
    plt.ylabel('Angular velocity (rad/s)')
    plt.grid()
    plt.legend([r'$\omega$1', r'$\omega$2', r'$\omega$3', r'$\omega$4'])
    plt.legend()
    plt.show()


def plot_input(slew=1):
    if slew == 1:
        slew_data = loadmat('Data/Slew1.mat')
    else:
        slew_data = loadmat('Data/Slew2.mat')
    test_data = slew_data['Test']
    time = np.linspace(0, 800, 8004)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(time, test_data)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_formatter(formatter)
    plt.xlabel('Time (s)')
    plt.ylabel('Body Frame Torque (Nm)')
    # plt.title('Body Frame Torque vs Time')
    plt.legend(["Tx", "Ty", "Tz"])
    # plt.fill([100, 100, 200, 200], [-1, 1, 1, -1], 'r', alpha=0.1)
    # plt.fill([300, 300, 400, 400], [-1, 1, 1, -1], 'r', alpha=0.1)
    # plt.fill([500, 500, 600, 600], [-1, 1, 1, -1], 'r', alpha=0.1)
    # plt.fill([700, 700, 800, 800], [-1, 1, 1, -1], 'r', alpha=0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.grid()
    # plt.show()


def plot_a(data):
    all_a_sol = data['all_a_sol'].transpose()
    all_t = data['all_t'].flatten()
    plt.plot(all_t, all_a_sol)
    plt.xlabel('Time (s)')
    plt.ylabel('a')
    plt.title('a vs Time')
    plt.show()


def plot_difference(data1,data2):
    all_w_sol1 = data1['all_w_sol']
    all_w_sol2 = data2['all_w_sol']
    all_t = data1['all_t'].flatten()
    plt.plot(all_t, all_w_sol1-all_w_sol2)
    plt.xlabel('Time (s)')
    plt.ylabel('Difference')
    plt.title('Difference vs Time')
    plt.show()


def plot_input_separate(slew=1):
    if slew == 1:
        slew_data = loadmat('Data/Slew1.mat')
    else:
        slew_data = loadmat('Data/Slew2.mat')
    test_data = slew_data['Test']
    time = np.linspace(0, 800, 8004)

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))

    for i in range(3):
        axs[i].plot(time, test_data[:, i])
        axs[i].yaxis.set_major_formatter(formatter)
        axs[i].set_ylabel(f'Row {i+1} Torque (Nm)')
        axs[i].grid()

    axs[-1].set_xlabel('Time (s)')
    plt.suptitle('Body Frame Torque vs Time')
    plt.show()


def repeat_function(func, directory):
    for filename in os.listdir(directory):
        if filename.endswith('.mat'):
            filepath = os.path.join(directory, filename)
            data = loadmat(filepath)
            func(data)

def plot_omega_squared():
    w_range = np.linspace(-300, 300, 100)  # Range for w values
    w1, w2 = np.meshgrid(w_range, w_range)  # Grid for first plot
    w3, w4 = np.meshgrid(w_range, w_range)  # Grid for second plot

    cost1 = w1 ** 2 + w2 ** 2  # Cost when varying w1, w2
    cost2 = w3 ** 2 + w4 ** 2  # Cost when varying w3, w4

    random_omega = np.random.uniform(-200, 200, (4,))
    w1_rand, w2_rand, w3_rand, w4_rand = random_omega

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot (w1 vs w2)
    contour1 = axes[0].contourf(w1, w2, cost1, levels=30, cmap='viridis')
    axes[0].scatter(w1_rand, w2_rand, color='red', marker='o', s=100, label="Random ω")
    axes[0].set_xlabel("w1")
    axes[0].set_ylabel("w2")
    axes[0].set_title("Cost Visualization (w1 vs w2)")
    fig.colorbar(contour1, ax=axes[0])  # Add colorbar

    # Second subplot (w3 vs w4)
    contour2 = axes[1].contourf(w3, w4, cost2, levels=30, cmap='viridis')
    axes[1].scatter(w3_rand, w4_rand, color='red', marker='o', s=100, label="Random ω")
    axes[1].set_xlabel("w3")
    axes[1].set_ylabel("w4")
    axes[1].set_title("Cost Visualization (w3 vs w4)")
    fig.colorbar(contour2, ax=axes[1])  # Add colorbar

    omega_min = 30

    for ax in axes:
        ax.axhline(omega_min, color='white', linestyle='dotted', linewidth=1)
        ax.axhline(-omega_min, color='white', linestyle='dotted', linewidth=1)
        ax.axvline(omega_min, color='white', linestyle='dotted', linewidth=1)
        ax.axvline(-omega_min, color='white', linestyle='dotted', linewidth=1)

    for ax, (w_x, w_y) in zip(axes, [(w1_rand, w2_rand), (w3_rand, w4_rand)]):
        x_vals = np.array(ax.get_xlim())  # Get current x-axis limits
        y_vals = w_y - (x_vals - w_x)  # y = (x - w_x) + w_y (45-degree line)
        ax.plot(x_vals, y_vals, color='white', linestyle='dashed', linewidth=1)

    axes[0].set_xlim(-300, 300)
    axes[0].set_ylim(-300, 300)
    axes[1].set_xlim(-300, 300)
    axes[1].set_ylim(-300, 300)

    plt.tight_layout()
    plt.show()


def live_plot_omega_squared(data):
    omega_values = data['all_w_sol'].T
    if omega_values.shape[0] != 4:
        raise ValueError("omega_values should have shape (4, N)")
    N = omega_values.shape[1]  # Number of frames
    w_range = np.linspace(-300, 300, 100)  # Range for w values
    w1, w2 = np.meshgrid(w_range, w_range)  # Grid for first plot
    w3, w4 = np.meshgrid(w_range, w_range)  # Grid for second plot

    cost1 = w1**2 + w2**2  # Cost when varying w1, w2
    cost2 = w3**2 + w4**2  # Cost when varying w3, w4

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # First subplot (w1 vs w2)
    contour1 = axes[0].contourf(w1, w2, cost1, levels=30, cmap='viridis')
    scatter1, = axes[0].plot([], [], 'ro', markersize=8)
    axes[0].set_xlabel("w1")
    axes[0].set_ylabel("w2")
    axes[0].set_title("Cost Visualization (w1 vs w2)")
    fig.colorbar(contour1, ax=axes[0])

    # Second subplot (w3 vs w4)
    contour2 = axes[1].contourf(w3, w4, cost2, levels=30, cmap='viridis')
    scatter2, = axes[1].plot([], [], 'ro', markersize=8)
    axes[1].set_xlabel("w3")
    axes[1].set_ylabel("w4")
    axes[1].set_title("Cost Visualization (w3 vs w4)")
    fig.colorbar(contour2, ax=axes[1])

    omega_min = 50

    for ax in axes:
        ax.axhline(omega_min, color='white', linestyle='dotted', linewidth=1)
        ax.axhline(-omega_min, color='white', linestyle='dotted', linewidth=1)
        ax.axvline(omega_min, color='white', linestyle='dotted', linewidth=1)
        ax.axvline(-omega_min, color='white', linestyle='dotted', linewidth=1)

    line1, = axes[0].plot([], [], 'w--', linewidth=1)
    line2, = axes[1].plot([], [], 'w--', linewidth=1)

    def update(frame):
        w1_rand, w2_rand, w3_rand, w4_rand = omega_values[:, frame]

        # Update scatter points
        scatter1.set_data([w1_rand], [w2_rand])  # Wrapping values in a list to make them sequences
        scatter2.set_data([w3_rand], [w4_rand])

        # Update diagonal lines
        x_vals = np.array(axes[0].get_xlim())
        line1.set_data(x_vals, w2_rand - (x_vals - w1_rand))
        line2.set_data(x_vals, w4_rand - (x_vals - w3_rand))

        return scatter1, scatter2, line1, line2

    ani = FuncAnimation(fig, update, frames=N, interval=100, blit=True)
    plt.show()


def check_momentum(data):
    omega = data['all_w_sol'].T
    momentum = R @ omega
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(momentum.T)

def check_momentum2(data):
    omega = data['all_w_sol']
    momentum = R @ omega
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(momentum.T)

def check_omega_squared(data):
    omega = data['all_w_sol'].T
    omega_squared = np.sum(omega**2, axis=0)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(omega_squared)

def check_omega_squared2(data):
    omega = data['all_w_sol'].T
    omega_squared = np.sum(omega**2, axis=1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(omega_squared)

def check_diff(data1, data2):
    omega1 = data1['all_w_sol'].T
    sq1 = np.sum(omega1**2, axis=0)
    omega2 = data2['all_w_sol'].T
    sq1 = np.sum(omega2 ** 2, axis=0)
    diff = sq1 - sq1
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(diff.T)
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.plot(sq1)
    ax.plot(sq1)


def plot_torque_flag(data):
    data = data['Test'].T
    low_torque_flag = hysteresis_filter(data, 0.000005, 0.000015)
    binary_flags = np.array(low_torque_flag, dtype=int)
    time = np.arange(len(binary_flags))

    plt.figure(figsize=(8, 4))
    plt.step(time, binary_flags, where="mid", label="Low Torque Flag", linewidth=2)
    plt.scatter(time, binary_flags, color="red", zorder=3)  # Highlight transitions

    # Formatting
    plt.yticks([0, 1], ["Normal", "Low Torque"])
    plt.xlabel("Time (Index)")
    plt.ylabel("Torque State")
    plt.title("Low Torque Detection Over Time")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()


def plot_radians_with_torque_flag(loaded_data):
    """Plots rad/s vs time and shades regions where low_torque_flag is True."""
    if 'all_w_sol' in loaded_data:
        # Extract wheel speeds
        all_w_sol = loaded_data['all_w_sol'].T
        all_t = np.linspace(0, all_w_sol.shape[0] / 10, all_w_sol.shape[0])

        data = load_data('Data/Slew1.mat')
        # Extract and process torque flag
        data = data['Test'].T
        low_torque_flag = hysteresis_filter(data, 0.000005, 0.000015)  # Compute flag
        binary_flags = np.array(low_torque_flag, dtype=int)

        # Plot wheel speed
        plt.figure(figsize=(10, 5))
        plt.plot(all_t, all_w_sol, label='Wheel Speed (rad/s)', color='b')

        # Highlight threshold lines
        plt.axhline(y=600, color='r', linestyle='--', label='Rad/s = 600')
        plt.axhline(y=-600, color='r', linestyle='--', label='Rad/s = -600')

        # Shade regions where low torque flag is True
        for i in range(1, len(binary_flags)):
            if binary_flags[i - 1] == 0 and binary_flags[i] == 1:  # Rising edge
                start_time = all_t[i]
            elif binary_flags[i - 1] == 1 and binary_flags[i] == 0:  # Falling edge
                end_time = all_t[i]
                plt.axvspan(start_time, end_time, color='red', alpha=0.3)

        # Formatting
        plt.xlabel('Time (s)')
        plt.ylabel('Rad/s')
        plt.title('Rad/s vs Time with Low Torque Regions')
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

def plot_method():
    # Generate an example omega state
    omega = np.random.uniform(-600, 600, (4, 1))
    omega = np.array([[-100], [-60], [150], [400]])

    # Compute values
    opt_alpha_unconstrained = optimal_alpha(omega)
    segments = line_constraint(omega)
    overlapped_constraints = overlap_constraint(segments)
    opt_alpha_constrained = constrained_alpha(omega)
    optimal = np.array([opt_alpha_constrained])

    # Generate alpha values for plotting the objective function
    alpha_values = np.linspace(-OMEGA_MAX, OMEGA_MAX, 500).reshape(1,500)
    objective_values = [(omega + np.array([1, -1, 1, -1]).reshape(4, 1) * alpha) ** 2 for alpha in alpha_values]
    objective_values = np.sum(objective_values[0], axis=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(alpha_values.flatten(), objective_values, label="Objective Function (sum omega²)", color="blue")

    # Mark the unconstrained optimal alpha
    ax.axvline(opt_alpha_unconstrained, color="green", linestyle="--", label="Unconstrained Optimal α")

    # Fill in constraint regions
    for start, end in overlapped_constraints:
        ax.axvspan(start, end, color='gray', alpha=0.3)
    for start, end in segments:
        ax.axvline(start, color="gray", linestyle="-.")
        ax.axvline(end, color="gray", linestyle="-.")

    # Mark the constrained optimal alpha
    ax.axvline(optimal, color="red", linestyle="--", label="Constrained Optimal α")
    # Labels and legend
    ax.set_xlabel("Alpha (α)")
    ax.set_ylabel("Objective Function Value")
    ax.set_title("Objective Function vs. Alpha with Constraints")
    ax.legend()
    ax.grid()

    plt.show()


def plot_path(data):
    # momentum3 = total_momentum(data)
    # momentum4 = R_PSEUDO @ momentum3
    momentum4 = pseudo_sol(data, OMEGA_START)
    momentum3 = R @ momentum4
    momentum5 = R_PSEUDO @ momentum3
    solution = loadmat('Data/Realtime/squared_omega.mat')
    solution = loadmat('Data/output.mat')

    ideal = solution['all_w_sol'].T

    time = np.linspace(0, 800, 8005)
    alpha = nullspace_alpha(momentum4)

    other_time = np.linspace(0, 800, 8000)
    other_alpha = -nullspace_alpha(ideal) +alpha[0]

    length = momentum4.shape[1]
    segments = np.zeros((8, length))

    for j in range(length):
        omega = momentum4[:, j]
        for i in range(4):
            segments[2*i, j] = (OMEGA_MIN - omega[i]) * (-1) ** i
            segments[1+2*i, j] = (-OMEGA_MIN - omega[i]) * (-1) ** i

    segments = segments - alpha[0]
    alpha = alpha - alpha[0]
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time, segments.T, color='gray')

    ax.plot(time, alpha, color='black', linestyle='--', label='Ideal path')
    ax.plot(other_time, other_alpha, color='red', label='Alpha')


    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i in range(0, 8, 2):
        color = colors[i // 2 % len(colors)]  # Cycle through colors
        plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    plt.xlabel("Time (s)")
    plt.ylabel("Nullspace component")
    plt.title("Zero speed bands vs time")
    plt.legend()

def plot_pockets(torque_data):
   # Simulate an 8x8004 array: 4 pairs of (top, bottom) stiction bands over 8004 time steps
    n_points = 8005
    n_bands = 4
    momentum4_with_nullspace = pseudo_sol(torque_data, OMEGA_START)
    momentum3 = R @ momentum4_with_nullspace
    momentum4 = R_PSEUDO @ momentum3
    stiction_array = calc_segments(momentum4)

    # Simulate each band pair as (bottom, top) with some sinusoidal structure
    time = np.linspace(0, 800, n_points)
    # stiction_array = np.zeros((2 * n_bands, n_points))

    # Store pocket bounds as (top_limit, bottom_limit, index_band1, index_band2)
    pockets = []

    # Iterate over all unique band pairs (i < j)
    for i in range(n_bands):
        bottom_i = stiction_array[2 * i]
        top_i = stiction_array[2 * i + 1]
        for j in range(i + 1, n_bands):
            bottom_j = stiction_array[2 * j]
            top_j = stiction_array[2 * j + 1]

            # Find regions where one band's top is below another band's bottom (non-overlapping)
            mask_ij = top_i < bottom_j
            mask_ji = top_j < bottom_i

            if np.any(mask_ij):
                pockets.append({
                    'top': bottom_j,
                    'bottom': top_i,
                    'mask': mask_ij,
                    'band_pair': (i, j)
                })
            if np.any(mask_ji):
                pockets.append({
                    'top': bottom_i,
                    'bottom': top_j,
                    'mask': mask_ji,
                    'band_pair': (j, i)
                })

    # Plotting the pockets over time
    plt.figure(figsize=(12, 6))
    for i in range(n_bands):
        plt.fill_between(time, stiction_array[2 * i], stiction_array[2 * i + 1], alpha=0.3, label=f'Band {i}')

    # Plot pockets
    for pocket in pockets:
        top = pocket['top']
        bottom = pocket['bottom']
        mask = pocket['mask']
        plt.fill_between(time[mask], bottom[mask], top[mask], color='black', alpha=0.2)

    plt.title("Stiction Bands and Pockets")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def mask_signal_diff(base, signal, tol=1e-6):
    diff_mask = np.abs(signal - base) > tol
    signal_masked = np.where(diff_mask, signal, np.nan)
    return signal_masked

def plot_nullspace(torque_data, n0=0, n=8005, limits=None, null_path=None, bounds=True,
         show_bands=True, scatter_points=None, start_point=None, momentum=None, legend=False):
    solution = loadmat('Data/DAC/full_objective/slew1/10000/9.mat')
    solution = loadmat('Data/OPT/02_zero/slew1/42.mat')
    # solution = loadmat('Data/Realtime/target/110.mat')
    omega_start = solution['omega_start']
    # omega_start = OMEGA_START
    opt0 = solution['null_sol']
    # opt0 = nullspace_alpha(solution['all_w_sol'].T)
    # solution = loadmat('Data/Realtime/target/-110.mat')
    # opt1 = nullspace_alpha(solution['all_w_sol'].T)


    momentum4_with_nullspace = pseudo_sol(torque_data, omega_start)
    alpha_nullspace = nullspace_alpha(momentum4_with_nullspace[:, 0:1])
    alpha_nullspace = nullspace_alpha(momentum4_with_nullspace)
    momentum3 = R @ momentum4_with_nullspace
    momentum4 = R_PSEUDO @ momentum3
    segments = calc_segments(momentum4)

    plt.figure(figsize=(10, 6))
    time = np.linspace(0, 800, 8005)
    zeros = np.zeros(8005)
    time_optimal = time[0:opt0.shape[1]]
    # plt.plot(time, zeros, color='black', linestyle='--', label='Ideal path')
    plt.plot(time_optimal, opt0.T, color='black', linestyle='--', label=r'Nullspace path')
    # plt.plot(time_optimal, opt1.T, color='black', linestyle='-.', label=r'Target = -110')
    # plt.plot(time_optimal, opt2.T, color='orange', linestyle='-.', label=r'Nullspace path')
    # plt.plot(time_optimal, opt3.T, color='green', linestyle='-.', label=r'Nullspace path')
    # plt.plot(time_optimal, opt2.T, color='black', linestyle='--', label=r'Pseudo Torque')
    # plt.plot(time_optimal, opt2.T, color='blue', linestyle='--', label='Min-Max Omega')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    if show_bands:
        for i in range(0, 8, 2):
            color = colors[i // 2 % len(colors)]  # Cycle through colors
            avg = (segments[i] + segments[i + 1]) / 2
            plt.plot(time, avg, color=color)
            plt.fill_between(time, segments[i], segments[i + 1], color=color, alpha=0.5, label=f'Band {i // 2 + 1}')
    if scatter_points is not None:
        for t, alpha, signs in scatter_points:
            plt.scatter(time[t], alpha, color='red', marker='o')
            # plt.scatter(time[2002:2003], [0], color='green', marker='x')
    if start_point is not None:
        plt.scatter(0, start_point, color='blue', marker='x')
    if limits is not None:
        min_limit, max_limit = limits
        plt.plot(time, max_limit, color='black', linestyle='--', label='Max Constraint')
        plt.plot(time, min_limit, color='black', label='Min Constraint')
    if null_path is not None:
        N = null_path.shape[0]
        plt.plot(time[0:N], null_path, color='blue', linestyle='--', label='Nullspace Path')
    if bounds:
        min_bounds, max_bounds = alpha_bounds(momentum4)
        plt.plot(time, min_bounds, color='gray', linestyle='--', label='Saturation limits')
        plt.plot(time, max_bounds, color='gray', linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel(r'Nullspace component $\left[\mathrm{rad/s}\right]$')
    # plt.title("Zero speed bands vs time")
    if legend:
        plt.legend()

        # handles, labels = plt.gca().get_legend_handles_labels()
        #
        # # Add grey patch
        # grey_patch = mpatches.Patch(color='lightgray', label='Pocket')
        # handles.append(grey_patch)
        # labels.append('Pocket')
        #
        # # Set updated legend
        # plt.legend(handles=handles, labels=labels, loc='upper right')
        plt.grid()
    plt.show()

if __name__ == "__main__":
    full_data = load_data('Data/Slew1.mat')
    # plot_path(full_data)
    # plt.rcParams.update({
    #     "font.size": 14,  # Global font size
    #     "axes.titlesize": 16,  # Title font size
    #     "axes.labelsize": 12,  # Axis label size
    #     "xtick.labelsize": 12,  # X tick label size
    #     "ytick.labelsize": 12,  # Y tick label size
    #     "legend.fontsize": 12,  # Legend font size
    # })
    plot_nullspace(full_data, legend=True, bounds=False)
    # plot_input(2)
    # momentum = pseudo_sol(full_data)
    # plt.plot(momentum.T)
    # plt.ylim([-600,600])
    # plot_MPI()
    plt.show()
