import os

import matplotlib.pyplot as plt
import jax.numpy as jnp

from kernlearn.learn import Optimisation
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import loss_plot
from kernlearn.models import *

data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, 28)
mle = Optimisation(optimiser="adam", learning_rate=0.001, n_epochs=50)
model = CuckerSmaleNN(hidden_layer_sizes=[64, 64], activation="tanh")
dropout_values = jnp.linspace(0, 0.9, 10)
fig, ax = plt.subplots(2, figsize=(5.6, 4))
cmap = plt.get_cmap("viridis")
for dropout_rate in dropout_values:
    model.dropout_rate = dropout_rate
    mle.fit(model, train_data, test_data)
    loss_plot(
        ax[0],
        mle.train_loss,
        train_label="%.1f" % dropout_rate,
        color=cmap(dropout_rate),
    )
    loss_plot(ax[1], mle.test_loss, train_label="", color=cmap(dropout_rate))
ax[0].set_xlabel("")
ax[0].set_xticklabels([])
ax[0].set_ylabel("Train Loss")
ax[1].set_ylabel("Test Loss")
ax[0].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, 1.3),
    ncol=len(dropout_values) // 2,
    frameon=False,
)
colorbar = False
if colorbar:
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    cb = fig.colorbar(sm, ax=ax[0], location="top")
    cb.outline.set_visible(False)
    cb.set_ticks([])
fig.savefig(os.path.join("figures", "fish", "dropout_effect", "dropout.pdf"))
