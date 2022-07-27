import numpy as np
import matplotlib.pyplot as plt
from kernlearn.utils import pdist
from kernlearn.plot_utils import *


def get_r(x, b):
    nsteps, N, _ = x.shape
    if b.shape[1] > 1:
        r = np.zeros((nsteps, N * (N - 1) // 2))
        for i in range(nsteps):
            r[i] = pdist(x[i, :, :], b[i, :, :])[np.triu_indices(N, 1)]
    else:
        r = np.zeros((nsteps, N))
        for i in range(nsteps):
            r[i] = pdist(x[i, :, :], b[i, :, :]).flatten()
    return r.flatten()


def plot_fish():
    data = np.load("data/fish/npy/data.npy", allow_pickle=True).item()
    r = get_r(data["x"], data["x"])
    BINS = 70
    plt.figure(figsize=(5.5, 3))
    plt.tight_layout()
    plt.hist(r, bins=BINS, color="#9D9D9D")
    # plt.xlim([0, 100])
    plt.xlabel("Pairwise distance $r$")
    plt.yticks([])
    plt.ylabel("")
    ax = plt.gca()
    ax.set_aspect(0.03)
    plt.savefig("figures/dist_hist_fish.pdf")
    plt.show()


def plot_sheep():
    x1 = np.load("data/sheep/npy/E1.npy")
    x2 = np.load("data/sheep/npy/E2.npy")
    x3 = np.load("data/sheep/npy/E3.npy")

    u1 = np.load("data/sheep/npy/Dog1.npy")
    u2 = np.load("data/sheep/npy/Dog2.npy")
    u3 = np.load("data/sheep/npy/Dog3.npy")
    r0 = get_r(x1, x1)
    r1 = get_r(x2, x2)
    r2 = get_r(x3, x3)
    r = np.concatenate((r0, r1, r2))
    np.save("figures/dist_hist", r)
    BINS = 100
    plt.hist(r, bins=BINS, color="black")
    plt.xlim([0, 100])
    plt.xlabel("Pairwise distance $r$")
    plt.yticks([])
    plt.ylabel("")
    plt.savefig("figures/dist_hist_u.pdf")
    plt.show()

    plt.tight_layout()
    fig, axs = plt.subplots(3)
    axs[0].hist(r0, bins=BINS, color="black")
    axs[0].set_xlim([0, 100])
    axs[0].set_ylabel("")
    axs[0].set_yticks([])
    axs[0].set_xticks([])
    axs[1].hist(r1, bins=BINS, color="black")
    axs[1].set_xlim([0, 100])
    axs[1].set_ylabel("")
    axs[1].set_yticks([])
    axs[1].set_xticks([])
    axs[2].hist(r2, bins=BINS, color="black")
    axs[2].set_xlim([0, 100])
    axs[2].set_ylabel("")
    axs[2].set_yticks([])
    axs[2].set_xlabel("Pairwise distance $r$")
    plt.savefig("figures/dist_hist_triple_u.pdf")
    plt.show()


if __name__ == "__main__":
    plot_sheep()
