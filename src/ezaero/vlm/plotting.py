import numpy as np

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3


def plot_panels(x, elev=25, azim=-160, edge_color='k',
                fill_color=1, transp=0.2, ax=None):

    m, n = x.shape[0], x.shape[1]
    X, Y, Z = [x[:, :, :, i] for i in range(3)]
    bp = Y.max() - Y.min()
    new_ax = not ax
    if new_ax:
        ax = a3.Axes3D(plt.figure())
    for i in range(m):
        for j in range(n):
            vtx = np.array([X[i, j], Y[i, j], Z[i, j]]).T
            panel = a3.art3d.Poly3DCollection([vtx])
            panel.set_facecolor((0, 0, fill_color, transp))
            panel.set_edgecolor(edge_color)
            ax.add_collection3d(panel)
    if new_ax:
        limits = (-bp / 1.8, bp / 1.8)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_zlim(limits)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(elev=elev, azim=azim)
    return ax


def plot_control_points(cpoints, ax):
    ax.scatter(
        xs=cpoints[:, :, 0].ravel(),
        ys=cpoints[:, :, 1].ravel(),
        zs=cpoints[:, :, 2].ravel()
    )
    return ax
