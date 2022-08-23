import os

import jax.numpy as jnp
import matplotlib.pyplot as plt

from kernlearn.utils import data_loader, train_test_split, save_hyperparameters
from kernlearn.plot_utils import *
from kernlearn.models import *
from kernlearn.learn import Optimisation
from kernlearn.generate_data import generate_data

SEED = 13
PATH = os.path.join("figures", "fish", "model_selection")


def main():
    # data = generate_data(CuckerSmale(seed=SEED, rayleigh=True), SEED+1)
    data = data_loader("data/fish/json/processed_data.json")
    train_data, test_data = train_test_split(data, ind=28)
    opt = Optimisation(
        optimiser="adam",
        learning_rate=1e-3,
        n_epochs=50,
        batch_size=28,
    )
    save_hyperparameters(opt, PATH)
    opt.init_mle_loss_fn()
    epochs = list(range(opt.n_epochs + 1))
    models = {
        r"CS: $K(1+r^2)^{-\beta}$": CuckerSmale(seed=SEED),
        "CS + Rayleigh": CuckerSmale(seed=SEED, rayleigh=True),
        "Mononomial (3)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=3),
        "Mononomial (6)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=6),
        "Chebyshev (3)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=3, chebyshev=True),
        "Chebyshev (6)": CuckerSmalePoly(seed=SEED, rayleigh=True, n=6, chebyshev=True),
        "Neural Network (8)": CuckerSmaleNN(
            seed=SEED,
            rayleigh=True,
            hidden_layer_sizes=[8],
            activation="elu",
            dropout_rate=0.0,
        ),
        "Neural Network (16)": CuckerSmaleNN(
            seed=SEED,
            rayleigh=True,
            hidden_layer_sizes=[16],
            activation="elu",
            dropout_rate=0.0,
        ),
    }
    n = len(models)
    color = plt.cm.plasma(np.linspace(0, 1, n + 1))
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", color)
    barfig = plt.figure(figsize=(5.6, 8))
    spec = plt.GridSpec(ncols=1, nrows=5, figure=barfig)
    phi_ax = barfig.add_subplot(spec[:2, 0])
    train_ax = barfig.add_subplot(spec[2, 0])
    test_ax = barfig.add_subplot(spec[3, 0])
    time_ax = barfig.add_subplot(spec[4, 0])
    test_ax.sharex(train_ax)
    time_ax.sharex(train_ax)
    lossfig, lossax = plt.subplots(nrows=2, figsize=(5.6, 4), sharex=True)
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
        phi_ax.plot(r, model.phi(r, model.params), label=key, alpha=0.8)
        train_ax.bar(i + 1, opt.train_loss[-1], linewidth=0, alpha=0.8)
        test_ax.bar(i + 1, opt.test_loss[-1], linewidth=0, alpha=0.8)
        time_ax.bar(i + 1, opt.time, linewidth=0, alpha=0.8)
        # Loss plots
        lossax[0].semilogy(epochs, opt.train_loss, label=key, alpha=0.8)
        lossax[1].semilogy(epochs, opt.test_loss, alpha=0.8)
        # Trajectory
        _traintrajax = traintrajax[(i + 1) // 3, (i + 1) % 3]
        _testtrajax = testtrajax[(i + 1) // 3, (i + 1) % 3]
        _traintrajax.set_title(key)
        _testtrajax.set_title(key)
        trajectory_plot(_traintrajax, *opt.predict(model.params, train_data), color="k")
        trajectory_plot(_testtrajax, *opt.predict(model.params, test_data), color="k")

    # Bar charts
    train_ax.grid(axis="y", linestyle="-", alpha=0.5)
    test_ax.grid(axis="y", linestyle="-", alpha=0.5)
    time_ax.grid(axis="y", linestyle="-", alpha=0.5)
    train_ax.set_ylabel("Train Loss")
    test_ax.set_ylabel("Test Loss")
    time_ax.set_ylabel("Wall Time (s)")
    phi_ax.set_ylabel(r"Interaction kernel $\phi(r)$")
    phi_ax.set_xlabel(r"Pairwise distance $r$")
    phi_ax.set_ylim([-0.2, 0.2])
    train_ax.set_xticklabels([])
    train_ax.set_xticks([])
    train_ax.set_xlabel("")
    phi_ax.legend(
        loc="lower left",
        markerscale=2,
        handlelength=1.5,
        columnspacing=1.0,
        ncol=2,
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

    # Trajectory plots
    traintrajfig.savefig(os.path.join(PATH, "train_trajectory.pdf"))
    testtrajfig.savefig(os.path.join(PATH, "test_trajectory.pdf"))


if __name__ == "__main__":
    main()
