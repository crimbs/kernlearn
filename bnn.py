import os
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import *
from kernlearn.models import *
from kernlearn.learn import Optimisation, sample

seed = 0
path = os.path.join("figures", "fish", "bnn")
data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, ind=28)
opt = Optimisation(n_epochs=100, batch_size=14, seed=seed, L=5, beta=0.001)
json.dump(opt.__dict__, open(os.path.join(path, type(opt).__name__ + ".json"), "w"))
model = CuckerSmaleNN(seed=seed)
opt.fit(model, train_data, test_data, bayesian=True)
fig, ax = plt.subplots()
loss_plot(ax, opt.train_loss, opt.test_loss)
fig.savefig(os.path.join(path, "loss.pdf"))
fig, ax = plt.subplots()
key = jax.random.PRNGKey(seed)
params = sample(key, model.params)
x, v = opt.predict(params, train_data)
trajectory_plot(ax, x, v)
fig.savefig(os.path.join(path, "trajectory.pdf"))
