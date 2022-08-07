import matplotlib.pyplot as plt

from kernlearn.learn import MLE
from kernlearn.utils import data_loader, train_test_split
from kernlearn.plot_utils import *
from kernlearn.models import *


data = data_loader("data/sheep/json/processed_data2.json")
train_data, test_data = train_test_split(data, ind=28)
model = SecondOrderSheep(
    N=data["x"].shape[1],
    hidden_layer_sizes=[16],
    activation="tanh",
    dropout_rate=0.3,
)
path = os.path.join("figures", "sheep", "E2", type(model).__name__)
mle = MLE(optimiser="adam", learning_rate=1e-3, n_epochs=150)
mle.fit(model, train_data, test_data)

# Trajectory comparison
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5.6, 5.6))
predator_prey_plot(
    ax[0, 0],
    train_data["x"],
    train_data["v"],
    train_data["u"],
    train_data["uv"],
    color="black",
    alpha=0.5,
)
x_prey_train, v_prey_train = mle.predict(train_data, model)
predator_prey_plot(
    ax[0, 1],
    x_prey_train,
    v_prey_train,
    train_data["u"],
    train_data["uv"],
    color="black",
    alpha=0.5,
)
predator_prey_plot(
    ax[1, 0],
    test_data["x"],
    test_data["v"],
    test_data["u"],
    test_data["uv"],
    color="black",
    alpha=0.5,
)
x_prey_test, v_prey_test = mle.predict(test_data, model)
predator_prey_plot(
    ax[1, 1],
    x_prey_test,
    v_prey_test,
    test_data["u"],
    test_data["uv"],
    color="black",
    alpha=0.5,
)
ax[0, 0].set_ylabel("Training Data")
ax[0, 0].set_title("Ground Truth")
ax[0, 1].set_title("Neural Network")
ax[1, 0].set_ylabel("Test Data")
fig.savefig(os.path.join(path, "trajectory_comparison.pdf"))

# Force plot
r = jnp.linspace(0, 1, num=200)
fig, ax = plt.subplots(2, figsize=(5.6, 3.5))
FG_plot(ax, r, model.F(r, model.params), model.G(r, model.params))
fig.savefig(os.path.join(path, "force.pdf"))

# Loss plot
fig, ax = plt.subplots()
loss_plot(ax, mle.train_loss, mle.test_loss)
fig.savefig(os.path.join(path, "loss.pdf"))
