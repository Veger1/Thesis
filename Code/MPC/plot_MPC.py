import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def plot_mpc_evolution(actual_w, all_w, total_points=8000, horizon=1, Omega_max=600, save_path=None):
    """
    Plots and animates the MPC Prediction vs Actual Evolution.

    Parameters:
    - actual_w: (4, N) array -> Actual angular velocity over time
    - all_w: (4, horizon_length + 1, N) array -> MPC predicted values
    - total_points: int -> Total time steps in the simulation
    - horizon_length: int -> MPC prediction horizon
    - Omega_max: float -> Y-axis limit (max angular velocity)
    - save_path: str or None -> If provided, saves animation to this path
    """

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(10, 6))
    actual_lines = [ax.plot([], [], label=f"State {j + 1}", linestyle="-")[0] for j in range(4)]
    pred_lines = [ax.plot([], [], linestyle="dotted")[0] for j in range(4)]

    ax.set_xlim(0, total_points + horizon)
    ax.set_ylim(-Omega_max, Omega_max)
    ax.set_title("MPC Prediction vs Actual Evolution")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Angular Velocity w")
    ax.legend()
    ax.grid()

    def update_plot(frame):
        """Update live plot with new predictions and actual trajectory."""
        if frame >= total_points:
            return actual_lines + pred_lines  # Stop updating when simulation ends

        for j in range(4):
            # Actual values up to time `frame`
            actual_lines[j].set_data(range(frame), actual_w[j, :frame])

            # MPC prediction for `frame`
            pred_lines[j].set_data(range(frame, frame + horizon + 1), all_w[j, :, frame])

        return actual_lines + pred_lines

    ani = animation.FuncAnimation(fig, update_plot, frames=total_points, interval=1, blit=True)

    if save_path:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(save_path, writer=writer)

    plt.show()
