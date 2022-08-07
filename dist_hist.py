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
fig, ax = plt.subplots()
kde_plot(ax, train_r, test_r)
fig.savefig(os.path.join("figures", "fish", "dist_hist", "dist_hist.pdf"))
