import optax
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from tqdm import trange


class MLE:
    def __init__(self, optimiser="adam", learning_rate=1e-3, n_epochs=50):
        self.n_epochs = n_epochs
        self.optimiser = getattr(optax, optimiser)(learning_rate)
        self.hparams = {
            "optimiser": optimiser,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
        }

    def fit(self, model, train_data, test_data, verbose=False):
        # Some neural network models use dropout for training
        if hasattr(model, "f_training"):
            dynamics = jit(
                lambda state, time, params: model.f_training(state, time, params)
            )
        else:
            dynamics = jit(lambda state, time, params: model.f(state, time, params))

        def forward(params, data):
            inputs = (data["x"][0], data["v"][0])
            return odeint(dynamics, inputs, data["t"], params)

        def loss_fn(params, data, ord=2):
            x_pred, v_pred = forward(params, data)
            x_sq_error = jnp.linalg.norm((x_pred - data["x"]), ord, axis=-1) ** 2
            v_sq_error = jnp.linalg.norm((v_pred - data["v"]), ord, axis=-1) ** 2
            return jnp.sum(x_sq_error) + jnp.sum(v_sq_error)

        # Initialise train and test loss history in list
        opt_state = self.optimiser.init(model.params)
        self.train_loss = [float(loss_fn(model.params, train_data))]
        self.test_loss = [float(loss_fn(model.params, test_data))]
        # Main loop
        for _ in trange(self.n_epochs):
            # Compute gradients and update parameters
            grads = grad(loss_fn)(model.params, train_data)
            updates, opt_state = self.optimiser.update(grads, opt_state)
            model.params = optax.apply_updates(model.params, updates)
            # Record updated train and test loss values
            self.train_loss.append(float(loss_fn(model.params, train_data)))
            self.test_loss.append(float(loss_fn(model.params, test_data)))
            if verbose:
                print(
                    f"""Train Loss: {self.train_loss[-1]}\tTest Loss: {self.test_loss[-1]}"""
                )
        return self

    def predict(self, data, model):
        dynamics = jit(lambda state, time, params: model.f(state, time, params))
        inputs = (data["x"][0], data["v"][0])
        return odeint(dynamics, inputs, data["t"], model.params)
