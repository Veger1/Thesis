import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the range for x, y, and a (parameter)
x = np.linspace(-10, 10, 100)  # Range of x
a_bis = np.linspace(0.1, 2, 10)   # Range of a (parameter varying along z)
a = 200/(a_bis+1) + 1
# Create a meshgrid for x and a
X, A = np.meshgrid(x, a)

# Calculate Y based on the parabola equation y = ax^2
Y = A * X**2

# Use the parameter 'a' as the z-axis
Z = A  # Z corresponds to the varying 'a'

# Plot the 3D surface
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

# Add labels and title
ax.set_title("3D Surface Plot of Parabola with Varying Parameter 'a'", fontsize=14)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis (Parabola)")
ax.set_zlabel("Z-axis (Parameter 'a')")

# Add color bar to indicate the scale of 'a'
cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
cbar.set_label("Parameter 'a'")

# Show the plot
plt.tight_layout()
plt.show()
