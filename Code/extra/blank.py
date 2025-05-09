import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axvline
from matplotlib.widgets import Slider

# Define the range of x values
x = np.linspace(-50, 50, 4000)
X = -10
Y = 10

# Define multiple functions that depend on k
def func1(x, k):
    return 1

def func2(x, k):
    return (np.tanh(k * (x - X))) * 0.5 + (np.tanh(-k * (x - Y))) * 0.5

def func3(x, k):
    return (x/k)**2

# Initialize k values for each function
initial_k1 = 1
initial_k2 = 1
initial_k3 = 1

# Initial y value for the sum of functions
y_sum = func2(x, initial_k2) + func3(x, initial_k3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

# Plot the sum of functions
line_sum, = ax.plot(x, y_sum, label="Sum of Functions")

# Set up plot limits and labels
ax.set_ylim(-0.3, 10.5)  # Adjusted the y-limits for better visualization of the sum
ax.set_xlabel("x")
ax.set_ylabel("y")
axvline(X, color='r', linestyle='--')
axvline(Y, color='r', linestyle='--')
ax.legend()

# Create sliders for each k
ax_slider1 = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
ax_slider2 = plt.axes([0.25, 0.15, 0.65, 0.03])
ax_slider3 = plt.axes([0.25, 0.2, 0.65, 0.03])

slider1 = Slider(ax_slider1, "k1 (not used)", 0.01, 50.0, valinit=initial_k1)
slider2 = Slider(ax_slider2, "k2 (Hyperbolic Tangent)", 0.01, 50.0, valinit=initial_k2)
slider3 = Slider(ax_slider3, "k3 (Omega squared)", 0.01, 50.0, valinit=initial_k3)

# Update function to change k values dynamically and plot the sum
def update(val):
    k1 = slider1.val
    k2 = slider2.val
    k3 = slider3.val
    # Update the sum of functions
    y_sum = func1(x, k1) + func2(x, k2) + func3(x, k3)
    line_sum.set_ydata(y_sum)
    fig.canvas.draw_idle()  # Redraw the plot

# Link the sliders to the update function
slider1.on_changed(update)
slider2.on_changed(update)
slider3.on_changed(update)

plt.show()
