import os

import matplotlib.pyplot as plt

from kernlearn.learn import Optimisation
from kernlearn.models import CuckerSmaleKNN
from kernlearn.utils import data_loader, save_hyperparameters, train_test_split
from kernlearn.plot_utils import *

SEED = 0
PATH = os.path.join("plots")


def main():
    data = data_loader("data/fish/json/processed_data.json")
    train_data, test_data = train_test_split(data, 28)
    opt = Optimisation(optimiser="adam", learning_rate=0.001, n_epochs=25)
    save_hyperparameters(opt, PATH)
    opt.init_mle_loss_fn()
    model = CuckerSmaleKNN(seed=SEED)
    fig, ax = plt.subplots(2, figsize=(5.6, 4), sharex=True)
    for k in range(1, 51):
        model.k = k
        opt.fit(model, train_data, test_data)
        ax[0].plot(model.k, opt.train_loss[-1], ".-k")
        ax[1].plot(model.k, opt.test_loss[-1], ".-k")
    ax[0].xaxis.get_major_locator().set_params(integer=True)
    ax[1].set_xlabel(r"$k$")
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Test Loss")
    fig.savefig(os.path.join(PATH, "elbow.pdf"))


if __name__ == "__main__":
    main()
