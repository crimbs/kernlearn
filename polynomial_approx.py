import os

import matplotlib.pyplot as plt
import jax.numpy as jnp

from kernlearn.learn import Optimisation
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import loss_plot, phi_plot, trajectory_plot
from kernlearn.models import CuckerSmalePoly

path = os.path.join("figures", "fish", "polynomial_approx")
data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, 28)
mle = Optimisation(
    optimiser="adam", learning_rate=0.001, n_epochs=100, reg_coeff=0.5, batch_size=2
)
model = CuckerSmalePoly(n=50)
mle.fit(model, train_data, test_data)
fig, ax = plt.subplots()
loss_plot(ax, mle.train_loss, mle.test_loss)
fig.savefig(os.path.join(path, "loss.pdf"))
fig, ax = plt.subplots()
r = jnp.linspace(0, 1, num=300)
phi_plot(ax, r, model.phi(r, model.params))
fig.savefig(os.path.join(path, "phi.pdf"))
fig, (ax1, ax2) = plt.subplots(2)
x1, v1 = mle.predict(model.params, train_data)
x2, v2 = mle.predict(model.params, test_data)
trajectory_plot(ax1, x1, v1)
trajectory_plot(ax2, x2, v2)
fig.savefig(os.path.join(path, "trajectory.pdf"))
