import os
from time import perf_counter

import matplotlib.pyplot as plt

from kernlearn.utils import data_loader, train_test_split, save_hyperparameters
from kernlearn.plot_utils import *
from kernlearn.models import *
from kernlearn.learn import Optimisation


SEED = 0
PATH = os.path.join("figures", "fish", "optimiser_choice")


def main():
    data = data_loader("data/fish/json/processed_data.json")
    train_data, test_data = train_test_split(data, ind=28)
    opt = Optimisation(
        optimiser="adam",
        learning_rate=1e-3,
        n_epochs=50,
        batch_size=28,
        seed=SEED,
        reg_coeff=0.0,
    )
    save_hyperparameters(opt, PATH)
    epochs = list(range(opt.n_epochs + 1))
    model = CuckerSmaleNN(
        seed=SEED,
        rayleigh=True,
        hidden_layer_sizes=[16],
        activation="tanh",
        dropout_rate=0.0,
    )
    optimisers = ["adabelief", "adam", "sgd", "rmsprop"]
    n = len(optimisers)
    color = plt.cm.plasma(np.linspace(0, 1, n))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color)
    fig, ax = plt.subplots(nrows=2, figsize=(5.6, 4), sharex=True)
    for optimiser in optimisers:
        opt.optimiser = optimiser
        print(opt.optimiser)
        tic = perf_counter()
        opt.fit(model, train_data, test_data)
        toc = perf_counter()
        ax[0].semilogy(
            epochs, opt.train_loss, label=opt.optimiser + " (%.2f s)" % (toc - tic)
        )
        ax[1].semilogy(epochs, opt.test_loss)
    ax[0].xaxis.get_major_locator().set_params(integer=True)
    ax[0].set_ylabel("Train Loss")
    ax[1].set_ylabel("Test Loss")
    ax[1].set_xlabel("Epochs")
    ax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.475, 1.3),
        ncol=4,
        frameon=False,
    )
    fig.savefig(os.path.join(PATH, "loss.pdf"))


if __name__ == "__main__":
    main()
