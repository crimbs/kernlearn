import os
import json
import numpy as np
import matplotlib.pyplot as plt
from kernlearn.kern_learn import KernLearn
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import (
    loss_comparison_plot,
    phi_comparison_plot,
    trajectory_plot,
)
from kernlearn.models import *

fish_data = data_loader("data/fish/json/processed_data.json")
fish_train, fish_test = train_test_split(fish_data, ind=28)

KL = KernLearn(optimiser="adam", learning_rate=1e-3, n_epochs=150)

# Initialize
r = np.linspace(0, 1, num=200)
loss_dict = {}
phi_dict = {}

# Vanilla model
V = CuckerSmale()
path = os.path.join("figures", "fish", V.id)
KL.fit(V, fish_train, fish_test)
x_V_train, v_V_train = KL.predict(fish_train, V.params)
x_V_test, v_V_test = KL.predict(fish_test, V.params)
phi_dict[V.id] = (V.phi(r, V.params).tolist(), None)
loss_dict[V.id] = (KL.test_loss, None)

# NN model
n = 2
phi_matrix = np.zeros((n, len(r)))
loss_matrix = np.zeros((n, KL.n_epochs + 1))
start = 1
for i, seed in enumerate(range(start, start + n)):
    print(seed)
    NN = CuckerSmaleNN(
        hidden_layer_sizes=[10],
        activation="tanh",
        dropout_rate=0.0,
        seed=seed,
    )
    path = os.path.join("figures", "fish", NN.id)
    KL.fit(NN, fish_train, fish_test)
    phi_matrix[i, :] = np.asarray(NN.phi(r, NN.params))
    loss_matrix[i, :] = np.asarray(KL.test_loss)
x_NN_test, v_NN_test = KL.predict(fish_test, NN.params)
x_NN_train, v_NN_train = KL.predict(fish_train, NN.params)
phi_dict[NN.id] = (np.mean(phi_matrix, axis=0), np.std(phi_matrix, axis=0))
loss_dict[NN.id] = (np.mean(loss_matrix, axis=0), np.std(loss_matrix, axis=0))

# Plot
fig = plt.figure(figsize=(5.6, 9.7))
G = fig.add_gridspec(5, 3)
true_train_ax = fig.add_subplot(G[0, 0])
true_train_ax.set_ylabel("Training Data")
true_train_ax.set_title("Ground truth")
true_test_ax = fig.add_subplot(G[1, 0])
true_test_ax.set_ylabel("Test Data")
V_train_ax = fig.add_subplot(G[0, 1])
V_train_ax.set_title(r"$\phi(r)=K(\sigma^2+r^2)^{-\beta}$")
V_test_ax = fig.add_subplot(G[1, 1])
NN_train_ax = fig.add_subplot(G[0, 2])
NN_train_ax.set_title(r"$\phi(r)=$ Neural network")
NN_test_ax = fig.add_subplot(G[1, 2])
phi_ax = fig.add_subplot(G[2, :])
loss_ax = fig.add_subplot(G[3, :])
trajectory_plot(true_train_ax, fish_train["x"], fish_train["v"], alpha=1)
trajectory_plot(true_test_ax, fish_test["x"], fish_train["v"], alpha=1)
trajectory_plot(V_train_ax, x_V_train, v_V_train, colour="g", alpha=1)
trajectory_plot(V_test_ax, x_V_test, v_V_test, colour="g", alpha=1)
trajectory_plot(NN_train_ax, x_NN_train, v_NN_train, colour="b", alpha=1)
trajectory_plot(NN_test_ax, x_NN_test, v_NN_test, colour="b", alpha=1)
phi_comparison_plot(phi_ax, r, phi_dict)
loss_comparison_plot(loss_ax, loss_dict)
# axs[0, 0].set_title("hello")
fig.savefig(os.path.join("figures", "fish", "phi_loss_comparison.pdf"))
plot_info = {"opt_hparams": KL.hparams, NN.id: NN.hparams, "n": n}
json.dump(plot_info, open(os.path.join(path, "phi_loss_comparison.txt"), "w"))
