from matplotlib import pyplot as plt
from scipy.io import loadmat
from helper import Helper

loaded_data = loadmat('Data/50s/stic2.mat')
# loaded_data = loadmat('output.mat')

helper = Helper()

def plot_cost_function():
    if 'cost_graph' in loaded_data:
        cost_graph = loaded_data['cost_graph'].transpose()
        omega_axis = loaded_data['omega_axis'].transpose()
        plt.plot(omega_axis, cost_graph, 'r-', label="Cost Function")
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.show()

def plot_cost():
    if 'cost' in loaded_data:
        cost = loaded_data['cost']
        time = loaded_data['all_t'].transpose()
        plt.plot(time, cost)
        plt.xlabel('Time')
        plt.ylabel('Cost')
        plt.title('Cost vs Time')
        plt.show()

def plot_total_cost():
    if 'total_cost' in loaded_data:
        total_cost = loaded_data['total_cost'].transpose()
        time = loaded_data['all_t'].transpose()
        plt.plot(time, total_cost)
        plt.xlabel('Time')
        plt.ylabel('Total Cost')
        plt.title('Total Cost vs Time')
        plt.show()

def plot_rpm():
    if 'all_w_sol' in loaded_data:
        all_w_sol = loaded_data['all_w_sol']
        all_t = loaded_data['all_t'].transpose()
        plt.axhline(y=6000, color='r', linestyle='--', label=f'rpm=6000')
        plt.axhline(y=-6000, color='r', linestyle='--', label=f'rpm=-6000')
        plt.fill([all_t[0], all_t[0], all_t[-1], all_t[-1]],[-300, 300, 300, -300], 'r', alpha=0.1)
        plt.plot(all_t, helper.rad_to_rpm(all_w_sol))
        plt.xlabel('Time')
        plt.ylabel('RPM')
        plt.title('RPM vs Time')
        plt.show()


plot_cost_function()
plot_rpm()
# print(loaded_data.keys())
print(loaded_data['cost_expr'])
