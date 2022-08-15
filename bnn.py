import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kernlearn.utils import data_loader, train_test_split, save_hyperparameters
from kernlearn.plot_utils import *
from kernlearn.models import *
from kernlearn.learn import Optimisation, sample


SEED = 0
PATH = os.path.join("figures", "fish", "bnn")


def main():
    data = data_loader("data/fish/json/processed_data.json")
    M, N, d = data["x"].shape
    train_data, test_data = train_test_split(data, ind=28)
    opt = Optimisation(
        n_epochs=100, batch_size=7, seed=SEED, bayesian=True, L=8, beta=0.001
    )
    save_hyperparameters(opt, PATH)
    model = CuckerSmalePoly(seed=SEED, rayleigh=True, n=6, chebyshev=True)
    opt.fit(model, train_data, test_data)

    # Loss
    lossfig, lossax = plt.subplots()
    loss_plot(lossax, opt.train_loss, opt.test_loss)
    lossfig.savefig(os.path.join(PATH, "loss.pdf"))

    # Trajectory
    r = jnp.linspace(0, 1, num=250)
    trajfig, trajax = plt.subplots(1, 2, figsize=(5.6, 2.8))
    phifig, phiax = plt.subplots()
    train_x_samples = np.zeros((opt.L, 28, N, d))
    train_v_samples = np.zeros((opt.L, 28, N, d))
    test_x_samples = np.zeros((opt.L, M - 28, N, d))
    test_v_samples = np.zeros((opt.L, M - 28, N, d))
    phi_samples = np.zeros((opt.L, len(r)))
    for seed in range(opt.L):
        key = jax.random.PRNGKey(seed)
        params = sample(key, model.params)
        # Phi
        phi_samples[seed] = model.phi(r, params)
        phiax.plot(r, phi_samples[seed], color="darkred", linewidth=0.5, alpha=0.2)
        # Trajectory
        train_x_samples[seed], train_v_samples[seed] = opt.predict(params, train_data)
        test_x_samples[seed], test_v_samples[seed] = opt.predict(params, test_data)
        trajectory_plot(
            trajax[0],
            train_x_samples[seed],
            train_v_samples[seed],
            color="darkred",
            alpha=0.2,
            linewidth=0.5,
            arrows=False,
        )
        trajectory_plot(
            trajax[1],
            test_x_samples[seed],
            test_v_samples[seed],
            color="darkred",
            alpha=0.2,
            linewidth=0.5,
            arrows=False,
        )
    # phi
    phi = jnp.mean(phi_samples, axis=0)
    err = jnp.std(phi_samples, axis=0)
    phi_plot(phiax, r, phi, err, color="k")
    # traj
    train_x = jnp.mean(train_x_samples, axis=0)
    test_x = jnp.mean(test_x_samples, axis=0)
    train_v = jnp.nanmean(train_v_samples, axis=0)
    test_v = jnp.nanmean(test_v_samples, axis=0)
    trajectory_plot(
        trajax[0], train_x, train_v, color="black", alpha=1, linewidth=0.75, arrows=True
    )
    trajectory_plot(
        trajax[1], test_x, test_v, color="black", alpha=1, linewidth=0.75, arrows=True
    )
    trajfig.savefig(os.path.join(PATH, "trajectory.pdf"))
    phifig.savefig(os.path.join(PATH, "phi.pdf"))


if __name__ == "__main__":
    main()
