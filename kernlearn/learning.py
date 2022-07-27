import os
import json
import optax
from tqdm import tqdm

import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from jax import config

config.update("jax_enable_x64", True)

from kernlearn.utils import jax2list
from kernlearn.plot_utils import loss_plot
from kernlearn.plot_utils import trajectory_comparison_plot


def main(
    model,
    hparams,
    train_data,
    test_data,
    save=False,
    verbose=False,
    data_string="experimental",
):

    # Initialize optimizer and parameters
    optimization_method = getattr(optax, hparams["optimizer"])
    optimizer = optimization_method(hparams["learning_rate"])
    params = model.params
    opt_state = optimizer.init(params)
    dynamics = jit(model.f)

    def forward(params, data):
        inputs = (data["x"][0], data["v"][0])
        return odeint(dynamics, inputs, data["t"], params)

    def loss_fn(params, data):
        """Mean distance squared error + mean velocity squared error."""
        x_pred, v_pred = forward(params, data)
        x_loss = jnp.mean(jnp.sum((x_pred - data["x"]) ** 2, axis=-1))
        v_loss = jnp.mean(jnp.sum((v_pred - data["v"]) ** 2, axis=-1))
        return x_loss + v_loss

    loss = {
        "train": [float(loss_fn(params, train_data))],
        "test": [float(loss_fn(params, test_data))],
    }
    print(model.string)
    for epoch in tqdm(range(hparams["n_epochs"])):
        # Compute gradients and update parameters
        grads = grad(loss_fn)(params, train_data)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Compute train and test loss
        loss["train"].append(float(loss_fn(params, train_data)))
        loss["test"].append(float(loss_fn(params, test_data)))
        if verbose:
            print(f"""Train Loss: {loss['train'][-1]}\tTest Loss: {loss['test'][-1]}""")

    if save:
        path = os.path.join("figures", data_string, model.string)
        try:
            os.mkdir(path)
        except OSError:
            pass
        json.dump(loss, open(os.path.join(path, "loss.txt"), "w"))
        json.dump(jax2list(params), open(os.path.join(path, "params.txt"), "w"))
        json.dump(hparams, open(os.path.join(path, "hyperparams.txt"), "w"))
        x, v = forward(params, test_data)
        trajectory_comparison_plot(path, test_data["x"], test_data["v"], x, v)
        loss_plot(path, loss["train"], loss["test"])

    return loss, params
