import os
import json

import numpy as np
import matplotlib.pyplot as plt

from kernlearn.learn import MLE
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import *
from kernlearn.models import *

# Initialize data, optimiser, and various dictionaries
data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, ind=28)
mle = MLE(optimiser="adam", learning_rate=1e-3, n_epochs=50)
r = np.linspace(0, 1, num=300)
test_loss_dict = {}
train_loss_dict = {}
phi_dict = {}


def opt_loop(model, n=1, start=11):
    phi_matrix = np.zeros((n, len(r)))
    train_loss_matrix = np.zeros((n, mle.n_epochs + 1))
    test_loss_matrix = np.zeros((n, mle.n_epochs + 1))
    for i, seed in enumerate(range(start, start + n)):
        model.seed = seed
        print(model.seed)
        mle.fit(model, train_data, test_data)
        phi_matrix[i, :] = np.asarray(model.phi(r, model.params))
        train_loss_matrix[i, :] = np.asarray(mle.train_loss)
        test_loss_matrix[i, :] = np.asarray(mle.test_loss)
    train_loss_dict[model.id] = (
        np.mean(train_loss_matrix, axis=0),
        np.std(train_loss_matrix, axis=0),
    )
    test_loss_dict[model.id] = (
        np.mean(test_loss_matrix, axis=0),
        np.std(test_loss_matrix, axis=0),
    )
    # Phi estimate
    phi_dict[model.id] = (
        np.mean(phi_matrix, axis=0),
        np.std(phi_matrix, axis=0),
    )
    # Trajectory prediction
    x_train, v_train = mle.predict(train_data, model)
    x_test, v_test = mle.predict(test_data, model)
    return x_train, v_train, x_test, v_test


# Initialise models
k = 7
V = CuckerSmale(k=k)
NN = CuckerSmaleNN(k=k, hidden_layer_sizes=[8], activation="tanh", dropout_rate=0.3)
CH = CuckerSmaleCheb(k=k, n=40)

x_V_train, v_V_train, x_V_test, v_V_test = opt_loop(V)
x_NN_train, v_NN_train, x_NN_test, v_NN_test = opt_loop(NN)
x_CH_train, v_CH_train, x_CH_test, v_CH_test = opt_loop(CH)


# Plot
fig = plt.figure(figsize=(5.6, 7.8))
G = fig.add_gridspec(5, 4)
true_train_ax = fig.add_subplot(G[0, 0])
true_test_ax = fig.add_subplot(G[1, 0])
V_train_ax = fig.add_subplot(G[0, 1])
V_test_ax = fig.add_subplot(G[1, 1])
NN_train_ax = fig.add_subplot(G[0, 2])
NN_test_ax = fig.add_subplot(G[1, 2])
CH_train_ax = fig.add_subplot(G[0, 3])
CH_test_ax = fig.add_subplot(G[1, 3])
phi_ax = fig.add_subplot(G[2, :])
loss_ax = fig.add_subplot(G[3, :])
true_train_ax.set_ylabel("Training Data")
true_test_ax.set_ylabel("Test Data")
true_train_ax.set_title("Ground Truth")
V_train_ax.set_title(r"$K(\sigma^2+r^2)^{-\beta}$")
NN_train_ax.set_title("Neural Network")
CH_train_ax.set_title("Chebyshev")
loss_ax.set_ylim([10e-2, 10e1])
color_list = ["r", "g", "b"]
trajectory_plot(true_train_ax, train_data["x"], train_data["v"])
trajectory_plot(true_test_ax, test_data["x"], train_data["v"])
trajectory_plot(V_train_ax, x_V_train, v_V_train, color=color_list[0])
trajectory_plot(V_test_ax, x_V_test, v_V_test, color=color_list[0])
trajectory_plot(NN_train_ax, x_NN_train, v_NN_train, color=color_list[1])
trajectory_plot(NN_test_ax, x_NN_test, v_NN_test, color=color_list[1])
trajectory_plot(CH_train_ax, x_CH_train, v_CH_train, color=color_list[2])
trajectory_plot(CH_test_ax, x_CH_test, v_CH_test, color=color_list[2])
phi_comparison_plot(phi_ax, r, phi_dict, color_list)
loss_comparison_plot(loss_ax, train_loss_dict, color_list, linestyle="solid")
loss_comparison_plot(loss_ax, test_loss_dict, color_list, linestyle="dashed")
path = os.path.join("figures", "fish", "fish_fitting")
fig.savefig(os.path.join(path, "phi_loss_comparison.pdf"))
plot_info = {"opt_hparams": mle.hparams, NN.id: NN.hparams, CH.id: CH.hparams}
json.dump(plot_info, open(os.path.join(path, "phi_loss_comparison.txt"), "w"))
