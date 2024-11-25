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
    return (1 / np.pi) * np.atan(k * (x - X)) + (1 / np.pi) * np.atan(-k * (x - Y))

def func2(x, k):
    return (np.tanh(k * (x - X))) * 0.5 + (np.tanh(-k * (x - Y))) * 0.5

def func3(x, k):
    return 1 / (1 + np.exp(k * (x - Y))) + 1 / (1 + np.exp(-k * (x - X))) - 1

# Initialize k value and create initial plots
initial_k = 1
y1 = func1(x, initial_k)
y2 = func2(x, initial_k)
y3 = func3(x, initial_k)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)

# Plot initial functions
line1, = ax.plot(x, y1, label="Arctangent")
line2, = ax.plot(x, y2, label="Hyperbolic Tangent")
line3, = ax.plot(x, y3, label="Logistic")

# Set up plot limits and labels
ax.set_ylim(-0.1, 1.1)
ax.set_xlabel("Angular Velocity (rad/s)")
ax.set_ylabel("Cost")
axvline(X, color='r', linestyle='--')
axvline(Y, color='r', linestyle='--')

ax.legend()

# Create slider axis and the slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
slider = Slider(ax_slider, "k", 0.01, 50.0, valinit=initial_k)

# Update function to change k value dynamically
def update(val):
    k = slider.val
    line1.set_ydata(func1(x, k))
    line2.set_ydata(func2(x, k))
    line3.set_ydata(func3(x, k))
    fig.canvas.draw_idle()  # Redraw the plot

# Link the slider to the update function
slider.on_changed(update)

plt.show()
