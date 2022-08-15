import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from kernlearn.learn import Optimisation
from kernlearn.models import CuckerSmaleAnticipation
from kernlearn.utils import data_loader, save_hyperparameters, train_test_split
from kernlearn.plot_utils import *

SEED = 0
PATH = os.path.join("figures", "fish", "anticipation_effect")


def main():
    data = data_loader("data/fish/json/processed_data.json")
    train_data, test_data = train_test_split(data, 28)
    opt = Optimisation(optimiser="adam", learning_rate=0.001, n_epochs=25)
    save_hyperparameters(opt, PATH)
    model = CuckerSmaleAnticipation(seed=SEED)
    fig, ax = plt.subplots(2, figsize=(5.6, 4), sharex=True)
    for tau in jnp.linspace(0, 1, 21):
        model.tau = tau
        opt.fit(model, train_data, test_data)
        ax[0].plot(model.tau, opt.train_loss[-1], ".-k")
        ax[1].plot(model.tau, opt.test_loss[-1], ".-k")
    ax[1].set_xlabel(r"$\tau$")
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Test Loss")
    fig.savefig(os.path.join(PATH, "elbow.pdf"))


if __name__ == "__main__":
    main()
