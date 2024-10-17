import matplotlib.pyplot as plt
from casadi import sum1, fabs
import numpy as np
from matplotlib.pyplot import figure

# Sample data
x = np.linspace(-6000, 6000, 1201)          # Data for the second y-axis

c1, c2 = 10, 0.0001  # Define the cost function terms
stiction = c1 * np.exp(-c2 * (x ** 2))  # Compute the exponential cost terms
s1, s2 = -5, 0.0005
x_ref = 1000
reference = s1 * np.exp(-s2 * ((abs(x) - x_ref) ** 2))

# # Create the figure and the primary axis
# fig, ax1 = plt.subplots()
#
# # Plot on the primary axis (y1)
# ax1.plot(x, stiction, 'g-', label="Stiction")
# ax1.set_xlabel('Rotation Speed (RPM)')
# ax1.set_ylabel('Cost', color='g')
# ax1.tick_params(axis='y', labelcolor='g')
# # Create a second y-axis (twin the x-axis)
# ax2 = ax1.twinx()
# # Plot on the secondary axis (y2) and flip the y-axis
# ax2.plot(x, reference, 'b-', label="Reference")
# ax2.set_ylabel('Cost', color='b')
# ax2.tick_params(axis='y', labelcolor='b')
# ax2.invert_yaxis()  # Flip the y-axis
#
# # Add a title and show the plot
# plt.title("Two Y-axes with One Flipped")


figure()
plt.plot(x, stiction, 'g-', label="Stiction")
plt.plot(x, reference, 'b-', label="Reference")
plt.xlabel('Rotation Speed (RPM)')
plt.ylabel('Cost')
plt.show()
