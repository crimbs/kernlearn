import os
import numpy as np
import matplotlib.pyplot as plt

from kernlearn.utils import pdist, data_loader, train_test_split
from kernlearn.plot_utils import kde_plot


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


data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, ind=28)
train_r = get_r(train_data["x"], train_data["x"])
test_r = get_r(test_data["x"], test_data["x"])
train_v = get_r(train_data["v"], train_data["v"])
test_v = get_r(test_data["v"], test_data["v"])
fig, ax = plt.subplots(nrows=2, figsize=(5.6, 4))
kde_plot(ax[0], train_r, test_r)
kde_plot(ax[1], train_v, test_v)
ax[0].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.2),
    ncol=2,
    frameon=False,
)
ax[0].set_xlim([0, 1])
ax[1].set_xlim([0, 0.005])
ax[0].set_xlabel(r"Pairwise distance $r$")
ax[1].set_xlabel(r"Pairwise velocity $v$")
fig.savefig(os.path.join("figures", "fish", "dist_hist", "dist_hist.pdf"))
