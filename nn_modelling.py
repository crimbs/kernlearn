from kernlearn.learning import main
from kernlearn.utils import data_loader, train_test_split
from kernlearn.nn_models import *

# Data
data = data_loader("data/fish/json/processed_data.json")
nsteps, N, d = data["x"].shape
train_data, test_data = train_test_split(data, ind=28)

hparams = {
    "learning_rate": 1e-3,
    "n_epochs": 200,
    "optimizer": "adam",
    "layer_sizes": [128],
    "activation": "tanh",
}

# Neural network models
CSNN = CuckerSmaleNN(hparams["layer_sizes"], hparams["activation"], seed=0)
CSRNN = CuckerSmaleRayleighNN(
    hparams["layer_sizes"], hparams["activation"], seed=0, kappa=0.0, p=0.0
)
NODE1 = FirstOrderNeuralODE(N, d, hparams["layer_sizes"], hparams["activation"], seed=0)
NODE2 = SecondOrderNeuralODE(
    N, d, hparams["layer_sizes"], hparams["activation"], seed=0
)

# Perform learning algorithm
# loss, params = main(CSNN, hparams, train_data, test_data, save=True, data_string="fish")
loss, params = main(
    CSRNN, hparams, train_data, test_data, save=True, data_string="fish"
)
