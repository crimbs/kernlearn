import os

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax.tree_util import tree_map
from jax.scipy.stats.multivariate_normal import pdf

from kernlearn.utils import data_loader, train_test_split, save_hyperparameters
from kernlearn.plot_utils import *
from kernlearn.models import *
from kernlearn.learn import Optimisation


SEED = 0
PATH = os.path.join("figures", "fish", "bayesian")


def mvn(model):
    mean = jnp.array(
        [model.params["mean"]["beta"].item(), model.params["mean"]["K"].item()]
    )
    cov = jnp.array(
        [
            [jnp.exp(model.params["log_std"]["beta"]).item(), 0],
            [0, jnp.exp(model.params["log_std"]["K"]).item()],
        ]
    )
    w, h = 201, 101
    x = np.linspace(1, 3, w)
    y = np.linspace(-0.5, 0.5, h)
    x, y = np.meshgrid(x, y)
    pos = np.dstack((x, y))
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.contour(x, y, pdf(pos, mean, cov), levels=20, cmap="gray_r")
    ax.set_ylim([-0.5, 0.5])
    ax.set_aspect("equal")
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel(r"$K$")
    # ax.annotate(r"$\phi(r)=\frac{K}{(1+r^2)^{\beta}}$", (2.55, 0.35))
    fig.savefig(os.path.join(PATH, "mvn.pdf"))


def main():
    data = data_loader("data/fish/json/processed_data.json")
    M, N, d = data["x"].shape
    train_data, test_data = train_test_split(data, ind=28)
    opt = Optimisation(
        optimiser="adam",
        learning_rate=1e-3,
        n_epochs=200,
        batch_size=28,
    )
    save_hyperparameters(opt, PATH)
    opt.init_elbo_loss_fn(num_samples=16, beta=0.001)
    model = CuckerSmale(seed=SEED)
    model.params = {
        "mean": model.params,
        "log_std": tree_map(lambda x: -4 * jnp.ones_like(x), model.params),
    }
    opt.fit(model, train_data, test_data)

    # Plot loss
    lossfig, lossax = plt.subplots()
    loss_plot(lossax, opt.train_loss, opt.test_loss)
    lossfig.savefig(os.path.join(PATH, "loss.pdf"))

    # Trajectory
    r = jnp.linspace(0, 1, num=250)
    fig = plt.figure(figsize=(5.6, 5.6))
    spec = plt.GridSpec(ncols=2, nrows=2, figure=fig)
    phi_ax = fig.add_subplot(spec[1, :])
    train_ax = fig.add_subplot(spec[0, 0])
    test_ax = fig.add_subplot(spec[0, 1])
    n_plot_samples = 16
    train_x_samples = np.zeros((n_plot_samples, 28, N, d))
    train_v_samples = np.zeros((n_plot_samples, 28, N, d))
    test_x_samples = np.zeros((n_plot_samples, M - 28, N, d))
    test_v_samples = np.zeros((n_plot_samples, M - 28, N, d))
    phi_samples = np.zeros((n_plot_samples, len(r)))
    key = jax.random.PRNGKey(SEED)
    key, *subkeys = jax.random.split(key, n_plot_samples + 1)
    for l, key in enumerate(subkeys):
        params = opt.sample(key, model.params)
        # Phi
        phi_samples[l] = model.phi(r, params)
        phi_ax.plot(r, phi_samples[l], color="darkred", linewidth=0.667, alpha=0.4)
        # Trajectory
        train_x_samples[l], train_v_samples[l] = opt.predict(params, train_data)
        test_x_samples[l], test_v_samples[l] = opt.predict(params, test_data)
        trajectory_plot(
            train_ax,
            train_x_samples[l],
            train_v_samples[l],
            color="darkred",
            alpha=0.4,
            linewidth=0.5,
            arrows=False,
        )
        trajectory_plot(
            test_ax,
            test_x_samples[l],
            test_v_samples[l],
            color="darkred",
            alpha=0.4,
            linewidth=0.5,
            arrows=False,
        )
    # phi
    phi = jnp.mean(phi_samples, axis=0)
    err = jnp.std(phi_samples, axis=0)
    phi_plot(phi_ax, r, phi, err, color="k", label="Mean")
    phi_ax.set_ylim([-0.2, 0.2])
    # traj
    train_x = jnp.mean(train_x_samples, axis=0)
    test_x = jnp.mean(test_x_samples, axis=0)
    train_v = jnp.nanmean(train_v_samples, axis=0)
    test_v = jnp.nanmean(test_v_samples, axis=0)
    trajectory_plot(
        train_ax, train_x, train_v, color="black", alpha=1, linewidth=0.75, arrows=True
    )
    trajectory_plot(
        test_ax, test_x, test_v, color="black", alpha=1, linewidth=0.75, arrows=True
    )
    train_ax.set_title("Train Data")
    test_ax.set_title("Test Data")
    fig.savefig(os.path.join(PATH, "bayesian.pdf"))

    # Standard deviation time horizon
    test_x_std = jnp.std(test_x_samples, axis=0)
    t_test = jnp.arange(M - 28) / 30
    fig, ax = plt.subplots(figsize=(5.6, 3))
    av = test_x_std.mean(axis=(1, 2))
    ax.plot(t_test, av, color="k", label="Mean")
    err = test_x_std.std(axis=(1, 2))
    ax.fill_between(
        t_test,
        av - err,
        av + err,
        color="k",
        linewidth=0,
        alpha=0.2,
        label=r"1$\sigma$",
    )
    ax.set_xlim([t_test[0], t_test[-1]])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Standard Deviation of Future Trajectories")
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(os.path.join(PATH, "std_horizon.pdf"))

    mvn(model)


if __name__ == "__main__":
    main()
