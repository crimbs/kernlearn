import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import networkx as nx
from scipy.spatial import KDTree

plt.style.use("classic")
plt.rcParams.update(
    {
        "figure.figsize": [5.6, 4.2],  # Width, height in inches
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "savefig.dpi": 300,  # The resolution in dots per inch
        "savefig.bbox": "tight",  # Bounding box in inches
        "animation.writer": "pillow",
    }
)


def loss_plot(path, train_losses, test_losses=None):
    fig, ax = plt.subplots()
    epochs = list(range(len(train_losses)))
    ax.semilogy(epochs, train_losses, "-k", label="Train")
    if test_losses is not None:
        assert len(train_losses) == len(test_losses)
        ax.semilogy(epochs, test_losses, "--k", label="Test")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    fig.savefig(os.path.join(path, "loss.pdf"))


def trajectory_plot(path, x, v, alpha=1):
    x = np.asarray(x)
    v = np.asarray(v)
    vn = v / np.linalg.norm(v, axis=-1)[..., None]  # Normalize
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.axis("equal")
    ax.set(xlim=[0, 1], ylim=[0, 1])
    ax.plot(x[..., 0], x[..., 1], "-k", linewidth=0.75, alpha=alpha)
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
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    fig.savefig(os.path.join(path, "trajectory.pdf"))


def trajectory_comparison_plot(path, x0, v0, x1, v1, alpha=1):
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    v0 = np.asarray(v0)
    v1 = np.asarray(v1)
    vn0 = v0 / np.linalg.norm(v0, axis=-1)[..., None]  # Normalize
    vn1 = v1 / np.linalg.norm(v1, axis=-1)[..., None]  # Normalize
    fig, ax = plt.subplots(ncols=2)
    ax[0].set_aspect("equal", adjustable="box")
    ax[0].set(xlim=[0, 1], ylim=[0, 1])
    ax[0].plot(x0[..., 0], x0[..., 1], "-k", linewidth=1, alpha=alpha)
    ax[0].quiver(
        x0[-1, :, 0],
        x0[-1, :, 1],
        vn0[-1, :, 0],
        vn0[-1, :, 1],
        angles="xy",
        scale_units="xy",
        scale=30,
        headaxislength=5,
        pivot="mid",
    )
    # ax[0].set_xlabel(r"$x$")
    # ax[0].set_ylabel(r"$y$")
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_aspect("equal", adjustable="box")
    ax[1].set(xlim=[0, 1], ylim=[0, 1])
    ax[1].plot(x1[..., 0], x1[..., 1], "-k", linewidth=1, alpha=alpha)
    ax[1].quiver(
        x1[-1, :, 0],
        x1[-1, :, 1],
        vn1[-1, :, 0],
        vn1[-1, :, 1],
        angles="xy",
        scale_units="xy",
        scale=30,
        headaxislength=5,
        pivot="mid",
    )
    # ax[1].set_xlabel(r"$x$")
    # ax[1].set_ylabel(r"$y$")
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(path, "trajectory_comparison.pdf"))


def predator_prey_plot(path, x_pred, x_prey):
    fig, ax = plt.subplots()
    ax.plot(x_pred[..., 0], x_pred[..., 1], "-k")
    ax.plot(x_prey[..., 0], x_prey[..., 1], "or")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.axis("equal")
    fig.savefig(os.path.join(path, "predator_prey.pdf"))


def phi_plot(path, r, ik):
    fig, ax = plt.subplots()
    ax.plot(r, ik, "-k")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$\phi(r)$")
    fig.savefig(os.path.join(path, "phi.pdf"))


def elbow_plot(path, k_values, final_loss):
    fig, ax = plt.subplots()
    ax.plot(k_values, final_loss, "o-k")
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.set_xlabel(r"$k$")
    ax.set_ylabel("Loss")
    fig.savefig(os.path.join(path, "elbow.pdf"))


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
        _, ind = KDTree(x[i]).query(x[i], k)
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
