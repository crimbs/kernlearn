from kernlearn.generate_data import generate_data
from kernlearn.learning import main
from kernlearn.utils import train_test_split
from kernlearn.vanilla_models import *
from kernlearn.nn_models import *

SEED = 0
# Generate data with known parameters
CS = CuckerSmale(K=0.25, beta=0.25, sigma=1.0)
data = generate_data(CS, seed=SEED, nsteps=50, N=10, d=2, ss=0)
train_data, test_data = train_test_split(data, ind=28)

# Now define a function apporximation version of the model
hparams = {
    "learning_rate": 1e-2,
    "n_epochs": 10,
    "optimizer": "adam",
    "layer_sizes": [32],
    "activation": "relu",
}
CSNN = CuckerSmaleNN(hparams["layer_sizes"], hparams["activation"], seed=SEED)
loss, params = main(
    CSNN, hparams, train_data, test_data, save=True, data_string="synthetic"
)
