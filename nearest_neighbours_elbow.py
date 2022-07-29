import os
import json
from kernlearn.kern_learn import second_order_system
from kernlearn.plot_utils import elbow_plot

hparams = {
    "learning_rate": 1e-3,
    "n_epochs": 10,
    "nn_width": 16,
    "nn_depth": 1,
    "activation": "relu",
    "optimizer": "adam",
}
data = json.load(open("data/fish/json/processed_data.json", "r"))
N = len(data["x"][0])
k_values = list(range(1, N))
fin_loss = list()
for k in k_values:
    hparams["k"] = k
    fin_loss.append(second_order_system(data, hparams, plot=False))
dir = os.path.join("figures", "fish-csfric-nn")
elbow_plot(dir, k_values, fin_loss)
