import os
from kernlearn.kern_learn import KernLearn
from kernlearn.utils import data_loader, train_test_split, save_individual
from kernlearn.plot_utils import phi_plot
from kernlearn.models import *

seed = 0

# Load data
fish_data = data_loader("data/fish/json/processed_data.json")
fish_train, fish_test = train_test_split(fish_data, ind=28)

# Initialise learning algorithm
KL = KernLearn(optimiser="adam", learning_rate=1e-3, n_epochs=10)
r = jnp.linspace(0, 1, num=200)  # horizontal axis for phi plots


CS = CuckerSmale()
path = os.path.join("figures", "fish", CS.id)
KL.fit(CS, fish_train, fish_test)
# save_individual(path, CS, KL, fish_train, fish_test)
phi_plot(path, r, CS.phi(r, CS.params))

CSR = CuckerSmaleRayleigh(K=0.0, beta=0.5, p=0.8, kappa=0.027)
path = os.path.join("figures", "fish", CSR.id)
KL.fit(CSR, fish_train, fish_test)
save_individual(path, CSR, KL, fish_train, fish_test)
phi_plot(path, r, CSR.phi(r, CSR.params))

CSRNN = CuckerSmaleRayleighNN(
    hidden_layer_sizes=[8],
    activation="tanh",
    dropout_rate=0.0,
    seed=seed,
    p=0.8,
    kappa=0.027,
)
path = os.path.join("figures", "fish", CSRNN.id)
KL.fit(CSRNN, fish_train, fish_test)
save_individual(path, CSRNN, KL, fish_train, fish_test)
phi_plot(path, r, CSRNN.phi(r, CSRNN.params))
