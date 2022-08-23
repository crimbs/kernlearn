import os

import numpy as np
import matplotlib.pyplot as plt

from kernlearn.learn import Optimisation
from kernlearn.models import CuckerSmale
from kernlearn.utils import data_loader, save_hyperparameters, train_test_split
from kernlearn.plot_utils import *

SEED = 0
PATH = os.path.join("figures", "fish", "local_interactions")


def main():
    data = data_loader("data/fish/json/processed_data.json")
    train_data, test_data = train_test_split(data, 28)
    opt = Optimisation(optimiser="adam", learning_rate=0.001, n_epochs=50)
    save_hyperparameters(opt, PATH)
    opt.init_mle_loss_fn()
    model = CuckerSmale(seed=SEED, rayleigh=True)
    fig, ax = plt.subplots(2, figsize=(5.6, 4), sharex=True)
    r0_values = np.linspace(0, 1, 6)[1:]
    # train = np.zeros_like(r0_values)
    # test = np.zeros_like(r0_values)
    epochs = np.arange(opt.n_epochs + 1)
    color = plt.cm.plasma(np.linspace(0, 1, len(r0_values) + 1))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color)
    for i, r0 in enumerate(r0_values):
        model.r0 = r0
        opt.fit(model, train_data, test_data)
        # train[i] = opt.train_loss[-1]
        # test[i] = opt.test_loss[-1]
        ax[0].semilogy(epochs, opt.train_loss, label=r"$r_0$ = " + f"{r0}")
        ax[1].semilogy(epochs, opt.test_loss)
    # ax[0].plot(r0_values, train, "-k")
    # ax[1].plot(r0_values, test, "-k")
    # ax[1].set_xlabel(r"$r_0$")
    ax[1].set_xlabel("Epochs")
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Test Loss")
    # ax[0].set_ylabel("Train Loss")
    # ax[1].set_ylabel("Test Loss")
    ax[0].legend(frameon=False)
    ax[0].set_ylim([1e-2, 1])
    ax[1].set_ylim([1e-2, 1e2])
    fig.savefig(os.path.join(PATH, "loss.pdf"))


if __name__ == "__main__":
    main()
