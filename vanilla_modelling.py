from kernlearn.learning import main
from kernlearn.utils import data_loader, train_test_split
from kernlearn.vanilla_models import *

# Data
data = data_loader("data/fish/json/processed_data.json")
train_data, test_data = train_test_split(data, ind=28)

# Hyperparameters
hparams = {
    "learning_rate": 1e-3,
    "n_epochs": 50,
    "optimizer": "adam",
}

# Vanilla models
CS = CuckerSmale(K=-0.03, beta=2.55)
CSR = CuckerSmaleRayleigh(K=0.0, beta=2.55, p=8e-1, kappa=2.7e-2)

# Perform learning algorithm
loss, params = main(CS, hparams, train_data, test_data, save=True, data_string="fish")
# loss, params = main(CSR, hparams, train_data, test_data, save=True, data_string="fish")
