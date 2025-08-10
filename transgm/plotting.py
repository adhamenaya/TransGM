import numpy as np
import matplotlib.pyplot as plt


class Plotting:
    def plot_3d_wireframe(self, grid1, values, color='r'):
        # Generate meshgrid for plotting
        X, Y = np.meshgrid(np.arange(grid1.shape[0]), np.arange(grid1.shape[1]))

        # Plotting
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(121, projection='3d')
        ax.plot_wireframe(X, Y, values, color=color, linewidth=0.6)

        # Setting the labels
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Density")

        plt.tight_layout()
        plt.show()