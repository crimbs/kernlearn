import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx

from kernlearn.utils import pdist, nearest_neighbours

plt.style.use("classic")
plt.rcParams.update(
    {
        "figure.figsize": [5.6, 3],  # Width, height in inches
        "figure.constrained_layout.use": True,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 10,
        "font.size": 10,
        "legend.fontsize": 9,
        "savefig.dpi": 300,  # The resolution in dots per inch
        "savefig.bbox": "tight",  # Bounding box in inches
        "animation.writer": "pillow",
        "image.cmap": "plasma",
    }
)


def kde_plot(ax, train_r, test_r):
    sns.kdeplot(
        train_r, color="black", shade=True, ax=ax, linestyle="solid", label="Train"
    )
    sns.kdeplot(
        test_r, color="black", shade=True, ax=ax, linestyle="dashed", label="Test"
    )


def loss_plot(
    ax,
    train_loss,
    test_loss=None,
    train_label="Train",
    test_label="Test",
    color="black",
):
    epochs = list(range(len(train_loss)))
    ax.semilogy(epochs, train_loss, color=color, label=train_label)
    if test_loss is not None:
        ax.semilogy(epochs, test_loss, color=color, label=test_label, alpha=0.5)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)


def trajectory_plot(ax, x, v, color="black", alpha=1, linewidth=0.75, arrows=True):
    x = np.asarray(x)
    v = np.asarray(v)
    vn = v / np.linalg.norm(v, axis=-1)[..., None]  # Normalize
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(x[..., 0], x[..., 1], color=color, linewidth=linewidth, alpha=alpha)
    if arrows:
        ax.quiver(
            x[-1, :, 0],
            x[-1, :, 1],
            vn[-1, :, 0],
            vn[-1, :, 1],
            angles="xy",
            scale_units="xy",
            scale=30,
            headaxislength=5,
            pivot="mid",
            color=color,
        )


def predator_prey_plot(ax, x_prey, v_prey, x_pred, v_pred, color="black", alpha=1.0):
    x_prey = np.asarray(x_prey)
    v_prey = np.asarray(v_prey)
    x_pred = np.asarray(x_pred)
    v_pred = np.asarray(v_pred)
    vpreyhat = v_prey / np.linalg.norm(v_prey, axis=-1, keepdims=True)
    vpredhat = v_pred / np.linalg.norm(v_pred, axis=-1, keepdims=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(x_prey[..., 0], x_prey[..., 1], color=color, linewidth=0.75, alpha=alpha)
    ax.quiver(
        x_prey[-1, :, 0],
        x_prey[-1, :, 1],
        vpreyhat[-1, :, 0],
        vpreyhat[-1, :, 1],
        angles="xy",
        scale_units="xy",
        scale=30,
        headaxislength=5,
        pivot="mid",
        color=color,
    )
    ax.plot(x_pred[..., 0], x_pred[..., 1], color="red", linewidth=0.75, alpha=1.0)
    ax.quiver(
        x_pred[-1, :, 0],
        x_pred[-1, :, 1],
        vpredhat[-1, :, 0],
        vpredhat[-1, :, 1],
        angles="xy",
        scale_units="xy",
        scale=30,
        headaxislength=5,
        pivot="mid",
        color="red",
    )


def phi_plot(ax, r, phi, error=0, color="black", label=None):
    ax.plot(r, phi, color=color, label=label)
    ax.fill_between(
        r,
        phi - error,
        phi + error,
        color=color,
        alpha=0.2,
        linewidth=0,
        label=r"$1\sigma$",
    )
    ax.set_xlabel(r"Pairwise distance $r$")
    ax.set_ylabel(r"Interaction kernel $\phi(r)$")
    ax.legend(frameon=False)


def FG_plot(ax, r, F, G, color="black"):
    ax[0].plot(r, F, color=color)
    ax[1].plot(r, G, color=color)
    ax[0].set_xticklabels([])
    ax[1].set_xlabel(r"Pairwise distance $r$")
    ax[0].set_ylabel(r"$F(r)$")
    ax[1].set_ylabel(r"$G(r)$")


def phi_comparison_plot(ax, r, phi_dict, color_list=["r", "g", "b"]):
    for i, key in enumerate(phi_dict):
        phi, error = phi_dict[key]
        if error is not None:
            ax.plot(r, phi, color=color_list[i], label=key)
            ax.fill_between(
                r,
                phi - error,
                phi + error,
                color=color_list[i],
                linewidth=0,
                alpha=0.2,
            )
        else:
            ax.plot(r, phi, color=color_list[i], label=key)
    ax.set_xlabel(r"Pairwise distance $r$")
    ax.set_ylabel(r"Interaction kernel $\phi(r)$")
    # ax.legend(frameon=False)


def elbow_plot(ax, k, loss):
    ax.plot(k, loss, "ok")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Loss")


def tau_elbow_plot(ax, tau_values, final_loss):
    ax.plot(tau_values, final_loss, "o-k")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel("Loss")


def animate_trajectory(path, x, v, t):
    x = np.asarray(x)
    v = np.asarray(v)
    v /= np.linalg.norm(v, axis=-1)[..., None]  # Normalize
    # lims = np.stack((np.min(x, axis=(0, 1)), np.max(x, axis=(0, 1))))
    fig, ax = plt.subplots()
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.axis("equal")
    ax.set_axis_off()
    (lax,) = ax.plot(x[0, :, 0], x[0, :, 1], ".k", markersize=1)
    qax = ax.quiver(
        x[0, :, 0],
        x[0, :, 1],
        v[0, :, 0],
        v[0, :, 1],
        angles="xy",
        scale_units="xy",
        scale=30,
        headaxislength=5,
        pivot="mid",
    )

    def animate(i):
        lax.set_data(x[:i].T)
        qax.set_offsets(x[i])
        qax.set_UVC(v[i, :, 0], v[i, :, 1])

    anim = FuncAnimation(fig, animate, frames=np.arange(1, len(t)), interval=100)
    anim.save(os.path.join(path, "animation.gif"))


def knn_animate(path, data, k):
    x = np.asarray(data["x"])
    lims = np.stack((np.min(x, axis=(0, 1)), np.max(x, axis=(0, 1))))
    fig, ax = plt.subplots()
    ax.set(xlim=lims[:, 0], ylim=lims[:, 1])
    ax.axis("equal")

    def animate(i):
        ax.clear()
        distance = pdist(x, x)
        _, ind = nearest_neighbours(distance, k)
        # Adjacency representation of graph as a dictionary of lists
        N = x[i].shape[0]
        keys = list(range(N))
        adj = dict(zip(keys, ind[:, 1:].tolist()))
        G = nx.DiGraph(adj)
        nx.set_node_attributes(G, dict(zip(keys, x[i])), "pos")
        nx.draw(
            G,
            ax=ax,
            pos=nx.get_node_attributes(G, "pos"),
            node_size=100,
            node_color="k",
            node_shape=".",
            width=0.5,
            arrowstyle="->",
            arrowsize=5,
        )

    anim = FuncAnimation(fig, animate, frames=len(data["t"]) - 1, interval=100)
    anim.save(os.path.join(path, "knn_animation.gif"), fps=24)


def plot_3d(path, x, t):
    x = np.asarray(x)
    t = np.asarray(t)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    N = x.shape[1]
    for i in range(N):
        ax.plot(x[:, i, 0], x[:, i, 1], t, "k", linewidth=1, alpha=0.6)
    ax.set_axis_off()
    fig.savefig(os.path.join(path, "3d_plot.pdf"), fps=24)


def anim_plot_3d_rotating(path, x, t):
    x = np.asarray(x)
    t = np.asarray(t)
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    N = x.shape[1]
    for i in range(N):
        ax.plot(x[:, i, 0], x[:, i, 1], t, "k", linewidth=1, alpha=0.6)
    ax.set_axis_off()
    animate = lambda degree: ax.view_init(azim=degree)
    anim = FuncAnimation(fig, animate, frames=np.arange(360), interval=100)
    anim.save(os.path.join(path, "3d_plot.gif"), fps=24)


def anim_plot_3d_time(path, x, t):
    x = np.asarray(x)
    t = np.asarray(t)
    nsteps, N, _ = x.shape
    lims = np.stack((np.min(x, axis=(0, 1)), np.max(x, axis=(0, 1))))
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set(xlim=lims[:, 0], ylim=lims[:, 1], zlim=[t[0], t[-1]])
    ax.set_axis_off()

    def animate(n):
        for i in range(N):
            ax.plot(
                x[:n, i, 0],
                x[:n, i, 1],
                t[:n],
                "k",
                linewidth=1,
                alpha=0.6,
            )

    anim = FuncAnimation(fig, animate, frames=np.arange(5, nsteps), interval=100)
    anim.save(os.path.join(path, "3d_plot_time.gif"), fps=24)
