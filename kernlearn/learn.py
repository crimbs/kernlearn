from functools import partial
from typing import Callable

import optax
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from tqdm import trange


class MLE:
    def __init__(
        self, optimiser: str = "adam", learning_rate: float = 1e-3, n_epochs: int = 50
    ):
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    @staticmethod
    def _forward(params: dict, data: dict, dynamics: Callable):
        """dynamics is ODE function with signature dynamics(state, time, params)"""
        inputs = (data["x"][0], data["v"][0])
        if "u" in data:
            return odeint(dynamics, inputs, data["t"], params, data["u"])
        else:
            return odeint(dynamics, inputs, data["t"], params)

    def fit(self, model, train_data: dict, test_data: dict, verbose: bool = False):
        self.predict = partial(self._forward, dynamics=jit(model.f))
        # Some neural network models use dropout for training
        if hasattr(model, "f_training"):
            forward = partial(self._forward, dynamics=jit(model.f_training))
        else:
            forward = self.predict

        def loss_fn(params, data, ord=2):
            x_pred, v_pred = forward(params, data)
            x_sq_error = jnp.linalg.norm((x_pred - data["x"]), ord, axis=-1) ** 2
            v_sq_error = jnp.linalg.norm((v_pred - data["v"]), ord, axis=-1) ** 2
            return jnp.sum(x_sq_error) + jnp.sum(v_sq_error)

        # Initialise optax optimiser
        optimiser = getattr(optax, self.optimiser)(self.learning_rate)
        opt_state = optimiser.init(model.params)
        # Initialise train and test loss history in list
        self.train_loss = [float(loss_fn(model.params, train_data))]
        self.test_loss = [float(loss_fn(model.params, test_data))]
        # Main loop
        for _ in trange(self.n_epochs):
            # Compute gradients and update parameters
            grads = grad(loss_fn)(model.params, train_data)
            updates, opt_state = optimiser.update(grads, opt_state)
            model.params = optax.apply_updates(model.params, updates)
            # Record updated train and test loss values
            self.train_loss.append(float(loss_fn(model.params, train_data)))
            self.test_loss.append(float(loss_fn(model.params, test_data)))
            if verbose:
                print(
                    f"""Train Loss: {self.train_loss[-1]}\tTest Loss: {self.test_loss[-1]}"""
                )
        return self
