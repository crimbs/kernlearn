import os

import matplotlib.pyplot as plt
import jax.numpy as jnp

from kernlearn.learn import Optimisation
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import tau_elbow_plot
from kernlearn.models import CuckerSmaleAnticipation

data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, 28)
mle = Optimisation(
    optimiser="adam", learning_rate=0.001, n_epochs=20, reg_coeff=0, batch_size=28
)
model = CuckerSmaleAnticipation()
tau_values = jnp.linspace(-2, 2, num=21)
final_loss = []
for k in tau_values:
    model.k = k
    mle.fit(model, train_data, test_data)
    final_loss.append(mle.test_loss[-1])
fig, ax = plt.subplots()
tau_elbow_plot(ax, tau_values, final_loss)
fig.savefig(os.path.join("figures", "fish", "anticipation_effect", "elbow.pdf"))
