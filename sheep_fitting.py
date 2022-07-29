import os
from kernlearn.kern_learn import KernLearn
from kernlearn.utils import data_loader, train_test_split, save_individual
from kernlearn.plot_utils import phi_plot, phi_comparison_plot
from kernlearn.models import *


# Load data
sheep_data1 = data_loader("data/sheep/json/processed_data1.json")
sheep_data3 = data_loader("data/sheep/json/processed_data3.json")
sheep_data2 = data_loader("data/sheep/json/processed_data2.json")
sheep_train1, sheep_test1 = train_test_split(sheep_data1, ind=28)
sheep_train2, sheep_test2 = train_test_split(sheep_data2, ind=28)
sheep_train3, sheep_test3 = train_test_split(sheep_data3, ind=28)

# Sheep
KL = KernLearn(optimiser="adam", learning_rate=1e-3, n_epochs=50)
r = jnp.linspace(0, 1, num=200)

PP1 = FirstOrderPredatorPrey(K=0.0, beta=0.5, p=1.0, kappa=1.0)
path = os.path.join("figures", "fish", PP1.id)
KL.fit(PP1, sheep_train1, sheep_test1)
save_individual(path, PP1, KL, sheep_train1, sheep_test1)
phi_plot(path, r, PP1.phi(r, PP1.params))
