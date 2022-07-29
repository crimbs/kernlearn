import optax
from tqdm import tqdm
import jax.numpy as jnp
from jax import grad, jit
from jax.experimental.ode import odeint
from jax import config

config.update("jax_enable_x64", True)


class KernLearn:
    def __init__(self, optimiser="adam", learning_rate=1e-3, n_epochs=50):
        self.n_epochs = n_epochs
        self.optimiser = getattr(optax, optimiser)(learning_rate)
        self.hparams = {
            "optimiser": optimiser,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
        }

    def fit(self, model, train_data, test_data, verbose=False):
        self.dynamics = jit(lambda state, time, params: model.f(state, time, params))

        def forward(params, data):
            inputs = (data["x"][0], data["v"][0])
            return odeint(self.dynamics, inputs, data["t"], params)

        def loss_fn(params, data, ord=2):
            """Mean distance squared error + mean velocity squared error."""
            x_pred, v_pred = forward(params, data)
            x_sq_error = jnp.linalg.norm((x_pred - data["x"]), ord, axis=-1) ** 2
            v_sq_error = jnp.linalg.norm((v_pred - data["v"]), ord, axis=-1) ** 2
            return jnp.sum(x_sq_error) + jnp.sum(v_sq_error)

        opt_state = self.optimiser.init(model.params)
        # Initialise train and test loss history in list
        self.train_loss = [float(loss_fn(model.params, train_data))]
        self.test_loss = [float(loss_fn(model.params, test_data))]
        # Main loop
        for _ in tqdm(range(self.n_epochs)):
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

    def predict(self, data, params):
        inputs = (data["x"][0], data["v"][0])
        return odeint(self.dynamics, inputs, data["t"], params)
