import os

import matplotlib.pyplot as plt

from kernlearn.learn import MLE
from kernlearn.models import CuckerSmale
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import elbow_plot

data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, 28)
mle = MLE(optimiser="adam", learning_rate=0.001, n_epochs=25)
model = CuckerSmale()
k_values = list(range(2, 51, 2))
final_loss = []
for k in k_values:
    model.k = k
    mle.fit(model, train_data, test_data)
    final_loss.append(mle.test_loss[-1])
fig, ax = plt.subplots()
elbow_plot(ax, k_values, final_loss)
fig.savefig(os.path.join("figures", "fish", "nearest_neighbours", "elbow.pdf"))
# TODO compare difference between train and test losses
