import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from kernlearn.utils import data_loader, train_test_split, save_hyperparameters
from kernlearn.plot_utils import *
from kernlearn.models import *
from kernlearn.learn import Optimisation


SEED = 0
PATH = os.path.join("figures", "fish", "model_selection")


def main():
    data = data_loader("data/fish/json/processed_data.json")
    train_data, test_data = train_test_split(data, ind=28)
    opt = Optimisation(
        optimiser="adam",
        learning_rate=1e-3,
        n_epochs=100,
        batch_size=7,
        seed=SEED,
        reg_coeff=0.0,
    )
    save_hyperparameters(opt, PATH)
    epochs = list(range(opt.n_epochs + 1))
    models = {
        r"CS: $K(1+r^2)^{-\beta}$": CuckerSmale(seed=SEED),
        "CS + Rayleigh": CuckerSmale(seed=SEED, rayleigh=True),
        "Polynomial (3)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=3),
        "Polynomial (6)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=6),
        "Chebyshev (3)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=3, chebyshev=True),
        "Chebyshev (6)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=6, chebyshev=True),
        "Neural Network (8)": CuckerSmaleNN(
            seed=SEED,
            rayleigh=True,
            hidden_layer_sizes=[8],
            activation="tanh",
            dropout_rate=0.0,
        ),
        "Neural Network (16)": CuckerSmaleNN(
            seed=SEED,
            rayleigh=True,
            hidden_layer_sizes=[16],
            activation="tanh",
            dropout_rate=0.0,
        ),
    }
    n = len(models)
    color = plt.cm.plasma(np.linspace(0, 1, n))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color)

    barfig, barax = plt.subplots(nrows=3, figsize=(5.6, 6), sharex=True)
    lossfig, lossax = plt.subplots(nrows=2, figsize=(5.6, 4), sharex=True)
    phifig, phiax = plt.subplots(figsize=(5.6, 3))
    traintrajfig, traintrajax = plt.subplots(nrows=3, ncols=3, figsize=(5.6, 5.6))
    testtrajfig, testtrajax = plt.subplots(nrows=3, ncols=3, figsize=(5.6, 5.6))
    trajectory_plot(traintrajax[0, 0], train_data["x"], train_data["v"], color="black")
    trajectory_plot(testtrajax[0, 0], test_data["x"], test_data["v"], color="black")
    traintrajax[0, 0].set_title("Ground Truth")
    testtrajax[0, 0].set_title("Ground Truth")
    r = jnp.linspace(0, 1, num=300)

    for i, key in enumerate(models):
        model = models[key]
        opt.fit(model, train_data, test_data)
        # Bar charts
        barax[0].bar(i + 1, opt.train_loss[-1], linewidth=0, label=key)
        barax[1].bar(i + 1, opt.test_loss[-1], linewidth=0)
        barax[2].bar(i + 1, opt.time, linewidth=0)
        # Loss plots
        lossax[0].semilogy(epochs, opt.train_loss, label=key)
        lossax[1].semilogy(epochs, opt.test_loss)
        # Phi
        (line,) = phiax.plot(r, model.phi(r, model.params), label=key)
        # Trajectory
        trainax = traintrajax[(i + 1) // 3, (i + 1) % 3]
        testax = testtrajax[(i + 1) // 3, (i + 1) % 3]
        trainax.set_title(key)
        testax.set_title(key)
        trajectory_plot(
            trainax, *opt.predict(model.params, train_data), color=line.get_color()
        )
        trajectory_plot(
            testax, *opt.predict(model.params, test_data), color=line.get_color()
        )
    # Bar charts
    barax[0].grid(axis="y", linestyle="-", alpha=0.5)
    barax[1].grid(axis="y", linestyle="-", alpha=0.5)
    barax[2].grid(axis="y", linestyle="-", alpha=0.5)
    barax[0].set_ylabel("Train Loss")
    barax[1].set_ylabel("Test Loss")
    barax[2].set_ylabel("Walltime (s)")
    barax[0].set_xticklabels([])
    barax[0].set_xlabel("")
    barax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.475, 1.3),
        ncol=4,
        frameon=False,
    )
    barfig.savefig(os.path.join(PATH, "bars.pdf"))

    # Loss plots
    lossax[0].xaxis.get_major_locator().set_params(integer=True)
    lossax[0].set_ylabel("Train Loss")
    lossax[1].set_ylabel("Test Loss")
    lossax[1].set_xlabel("Epochs")
    lossax[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.475, 1.3),
        ncol=4,
        frameon=False,
    )
    lossfig.savefig(os.path.join(PATH, "loss.pdf"))

    # Phi plot
    phiax.set_ylabel(r"Interaction kernel $\phi(r)$")
    phiax.set_xlabel(r"Pairwise distance $r$")
    phiax.legend(
        loc="upper center",
        bbox_to_anchor=(0.475, 1.2),
        ncol=4,
        frameon=False,
    )
    phifig.savefig(os.path.join(PATH, "phi.pdf"))

    # Trajectory plots
    traintrajfig.savefig(os.path.join(PATH, "train_trajectory.pdf"))
    testtrajfig.savefig(os.path.join(PATH, "test_trajectory.pdf"))


if __name__ == "__main__":
    main()
