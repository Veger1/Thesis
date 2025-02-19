from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from scipy.io import loadmat, whosmat
import numpy as np
from matplotlib.animation import FuncAnimation, writers
from sympy import lambdify, symbols, sympify

from helper import Helper

helper = Helper()


def load_data(dataset):  # Improve by doing the flattening/transposing here, this function
    # essentially does nothing except catching exceptions
    try:
        loaded_data = loadmat(dataset)
    except Exception as e:
        print(f"Error: {e}")
        return None
    return loaded_data


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
        plt.plot(all_t, helper.rad_to_rpm(all_w_sol))
        plt.xlabel('Time (s)')
        plt.ylabel('RPM')
        plt.title('RPM vs Time')
        plt.show()


def plot_radians(loaded_data):
    if 'all_w_sol' in loaded_data:
        all_w_sol = loaded_data['all_w_sol']
        all_t = loaded_data['all_t'].flatten()
        plt.axhline(y=600, color='r', linestyle='--', label=f'rad/s=600')
        plt.axhline(y=-600, color='r', linestyle='--', label=f'rad/s=-600')
        plt.fill([all_t[0], all_t[0], all_t[-1], all_t[-1]],[-30, 30, 30, -30], 'r', alpha=0.1)
        plt.plot(all_t, all_w_sol)
        # plt.ylim([-300, 300])
        plt.xlabel('Time (s)')
        plt.ylabel('Rad/s')
        plt.title('Rad/s vs Time')
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
    plt.plot(mpi_data['all_t'].flatten(), helper.rad_to_rpm(w_sol[:8000, :]))
    plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
    plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
    plt.fill([0, 0, 800, 800], [-300, 300, 300, -300], 'r', alpha=0.1)
    # plt.ylim([-6000, 6000])
    plt.xlabel('Time (s)')
    plt.ylabel('RPM')
    plt.title('RPM vs Time')
    plt.legend([r'$\omega$1', r'$\omega$2', r'$\omega$3', r'$\omega$4'])
    plt.show()


def plot_input(slew=1):
    if slew == 1:
        slew_data = loadmat('Data/Slew1.mat')
    else:
        slew_data = loadmat('Data/Slew2.mat')
    test_data = slew_data['Test']
    time = np.linspace(0, 800, 8004)
    fig, ax = plt.subplots()
    ax.plot(time, test_data)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-3, -3))
    ax.yaxis.set_major_formatter(formatter)
    plt.xlabel('Time (s)')
    plt.ylabel('Body Frame Torque (Nm)')
    plt.title('Body Frame Torque vs Time')
    plt.legend(["Tx", "Ty", "Tz"])
    # plt.fill([100, 100, 200, 200], [-1, 1, 1, -1], 'r', alpha=0.1)
    # plt.fill([300, 300, 400, 400], [-1, 1, 1, -1], 'r', alpha=0.1)
    # plt.fill([500, 500, 600, 600], [-1, 1, 1, -1], 'r', alpha=0.1)
    # plt.fill([700, 700, 800, 800], [-1, 1, 1, -1], 'r', alpha=0.1)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # plt.grid()
    plt.show()


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


data = loadmat('Data/slew1_2periods.mat')

# plot_cost_function(data)
# plot_cost_time(data)
# live_cost_plot(data, 'Data/Slew1_cost.mp4')
plot_radians(data)
# plot_torque(data)
# plot_a(data)
plot_input(1)

# plot_difference(loadmat('Data/slow1.mat'), loadmat('Data/fast1.mat'))


