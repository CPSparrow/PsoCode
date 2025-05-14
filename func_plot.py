import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np

from pso import f

if __name__ == "__main__":
    mplstyle.use("fast")
    x_values = np.linspace(5.95, 6.2, 900)
    y_values = np.linspace(7.65, 7.9, 900)

    x, y = np.meshgrid(x_values, y_values)

    z = f(x, y)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none")

    ax.set_title("3D Plot of f(x, y)")
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

    print(np.max(z))
    # plt.show()
    # plt.savefig('./pics/plot2.svg')
