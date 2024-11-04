import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.io import loadmat
from sympy import symbols, exp, Abs

# Load the data
loaded_data = loadmat('Data/50s/reference.mat')
all_w_sol = loaded_data['all_w_sol']  # Transpose to make it 8000x4

# Define constants and symbolic variables
s1, s2 = -1, 0.001
w_ref = 104

time = np.linspace(0, 800, 8000)
omega_axis = np.linspace(-300, 300, 500)

# Define the symbolic expression for the cost function
omega = symbols('omega')
cost_expr = s1 * exp(-s2 * ((Abs(omega) - w_ref) ** 2))

# Evaluate the cost for each element in all_w_sol
# Convert cost_expr to a NumPy-friendly function
cost_func = np.vectorize(lambda omega_val: cost_expr.subs(omega, omega_val).evalf())
cost = cost_func(all_w_sol)
cost_graph = cost_func(omega_axis)
total_cost = np.sum(cost, axis=1)
total_cost = np.array(total_cost, dtype=float)

# Create the plot
fig, (ax, ax_total_cost) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [4, 1]})
ax.set_xlim(-300, 300)
ax.set_ylim(-1, 1)
ax.plot(omega_axis, cost_graph, 'r-', label="Cost Function")  # Red line for cost function
points = [ax.plot([], [], 'o', label='Live point')[0] for _ in range(4)]  # Create 4 points for each column

# Add a text box for the time display
time_text = ax.text(0.05, 0.95, 'Time: 0.00 s', transform=ax.transAxes, fontsize=12, verticalalignment='top')

# Plot total cost in the second subplot
ax_total_cost.set_ylim(np.min(total_cost) * 1.1, np.max(total_cost) * 1.1)  # Set y limit with a little padding
bar = ax_total_cost.bar(0, total_cost[0], color='blue')  # Initial bar
ax_total_cost.set_xticks([])  # Remove x-ticks to keep it clean
ax_total_cost.set_title("Total Cost")

# Initialization function
def init():
    for point in points:
        point.set_data([], [])
    time_text.set_text('Time: 0.00 s')  # Reset time text
    bar[0].set_height(total_cost[0])  # Reset total cost bar
    return points + [time_text]

# Update function for each frame
def update(frame):
    for i, point in enumerate(points):
        point.set_data([all_w_sol[frame, i]], [cost[frame, i]])
    time_text.set_text(f'Time: {time[frame]:.2f} s')
    bar[0].set_height(total_cost[frame])
    return points + [time_text, bar[0]]

# Animate
ani = FuncAnimation(fig, update, frames=len(time), init_func=init, blit=True, interval=5)

# Show the plot
plt.tight_layout()
plt.show()
