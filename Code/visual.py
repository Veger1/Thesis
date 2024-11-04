from matplotlib import pyplot as plt
from scipy.io import loadmat

loaded_data = loadmat('Data/50s/reference.mat')

def plot_cost_function():
    if 'cost_graph' in loaded_data:
        cost_graph = loaded_data['cost_graph']
        omega_axis = loaded_data['omega_axis']
        plt.plot(omega_axis, cost_graph, 'r-', label="Cost Function")
        plt.xlabel('Angular Velocity (rad/s)')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.show()

def plot_cost():
    if 'cost' in loaded_data:
        cost = loaded_data['cost']
        time = loaded_data['all_t']
        plt.plot(time, cost)
        plt.xlabel('Time')
        plt.ylabel('Cost')
        plt.title('Cost vs Time')
        plt.show()

def plot_total_cost():
    if 'total_cost' in loaded_data:
        total_cost = loaded_data['total_cost']
        time = loaded_data['all_t']
        plt.plot(time, total_cost)
        plt.xlabel('Time')
        plt.ylabel('Total Cost')
        plt.title('Total Cost vs Time')
        plt.show()