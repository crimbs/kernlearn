import random
from functools import partial
from time import perf_counter
from typing import Callable, Dict

import optax
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_map, tree_leaves, tree_reduce
from jax.experimental.ode import odeint
from tqdm import tqdm


def get_batches(data: Dict, n_batches: int, seed: int):
    """Batches the data dictionary into `n_batches` minibatches and shuffles them.
    Each minibatch can be retrieved by `tree_map(next, shuffled_minibatches)`.

    Returns
    -------
    shuffled_minibatches : a dict of list_iterator objects
    """

    def batch_fn(data):
        split_data = jnp.array_split(data, n_batches)
        random.Random(seed).shuffle(split_data)  # in-place operation
        return iter(split_data)

    shuffled_minibatches = tree_map(batch_fn, data)
    return shuffled_minibatches


def integrate(params: Dict, data: Dict, dynamics: Callable):
    """Computes an integration forward pass.

    Parameters
    ----------
    dynamics : ODE function with signature (state, time, params, control)

    Returns
    -------
    predictions : tuple (x, v) of position and velocity jnp.ndarrays
    """
    inputs = (data["x"][0], data["v"][0])
    times = data["t"]
    control = data["u"] if "u" in data else None
    predictions = odeint(dynamics, inputs, times, params, control)
    return predictions


def l2_norm(params: Dict):
    """Evaluate the l2 norm of the parameters dictionary."""
    return 0.5 * sum(jnp.sum(p**2) for p in tree_leaves(params))


def sample(key, q: Dict):
    """Draw a random Gaussian sample from variational posterior q."""

    def _sample(mu, logvar):
        eps = jax.random.normal(key, mu.shape)
        return mu + eps * jnp.exp(0.5 * logvar)

    return tree_map(_sample, q["mu"], q["logvar"])


def kl_divergence(q: Dict):
    """Analytical solution to the Kullback-Leibler divergence when
    the prior p and variational posterior q are both Gaussian.

    Notes
    -----
    See Appendix B of https://arxiv.org/pdf/1312.6114.pdf
    """

    def _kl_divergence(mu, logvar):
        return -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar))

    out = tree_map(_kl_divergence, q["mu"], q["logvar"])
    return tree_reduce(lambda a, b: a + b, out)


def log_likelihood(params: Dict, data: Dict, forward: Callable):
    x_pred, v_pred = forward(params, data)
    x_sq_error = jnp.sum((x_pred - data["x"]) ** 2, axis=-1)
    v_sq_error = jnp.sum((v_pred - data["v"]) ** 2, axis=-1)
    return -0.5 * (jnp.sum(x_sq_error) + jnp.sum(v_sq_error))


class Optimisation:
    def __init__(
        self,
        optimiser: str = "adam",
        learning_rate: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 4,
        seed: int = 0,
        bayesian: bool = False,
        L: int = 5,  # No. of Monte Carlo samples
        beta: float = 0.001,  # KL divergence weighting
        reg_coeff: float = 0.0,
    ):
        assert optimiser in optax.__all__, "Must be an optax optimiser."
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.n_epochs = int(n_epochs)
        self.seed = seed
        self.reg_coeff = reg_coeff
        self.batch_size = int(batch_size)
        self.bayesian = bayesian
        self.L = int(L)
        self.beta = beta

    def print_losses(self):
        print(f"Train Loss: {self.train_loss[-1]}\tTest Loss: {self.test_loss[-1]}")

    def fit(
        self,
        model,
        train_data: Dict,
        test_data: Dict,
        verbose: bool = False,
    ):
        self.predict = partial(integrate, dynamics=jit(model.f))
        # Some neural network models use dropout for training
        if hasattr(model, "f_training"):
            forward = partial(integrate, dynamics=jit(model.f_training))
            log_lik = jit(partial(log_likelihood, forward=forward))
        else:
            log_lik = jit(partial(log_likelihood, forward=self.predict))

        if self.bayesian:
            # Equip parameters with log(variance) for mean-field variational inference
            # Set the initial variance to exp(-7) ≈ 0.001
            model.params = {
                "mu": model.params,
                "logvar": tree_map(lambda x: -7 * jnp.ones_like(x), model.params),
            }

            # TODO Investigate using pmap to parallelise this
            @partial(vmap, in_axes=(None, None, 0))
            def sample_log_lik(q: Dict, data: Dict, key):
                params = sample(key, q)
                return log_lik(params, data)

            def elbo(q: Dict, data: Dict, key):
                """Evidence lower bound."""
                key, subkey = jax.random.split(key)
                sample_keys = jax.random.split(subkey, self.L)
                expected_log_lik = jnp.mean(sample_log_lik(q, data, sample_keys))
                return expected_log_lik - self.beta * kl_divergence(q)

            @jit
            def loss_fn(params, data, key):
                return -elbo(params, data, key)

        else:
            # Maximum likelihood estimation
            @jit
            def loss_fn(params, data, _):
                return -log_lik(params, data) + self.reg_coeff * l2_norm(params)

        optimiser = getattr(optax, self.optimiser)(self.learning_rate)
        opt_state = optimiser.init(model.params)
        key = jax.random.PRNGKey(self.seed)
        self.train_loss = [float(loss_fn(model.params, train_data, key))]
        self.test_loss = [float(loss_fn(model.params, test_data, key))]
        n_batches = len(train_data["t"]) // self.batch_size
        tic = perf_counter()
        for epoch in tqdm(
            range(self.n_epochs), desc=f"Fitting\t{type(model).__name__}"
        ):
            shuffled_minibatches = get_batches(train_data, n_batches, seed=epoch)
            for _ in range(n_batches):
                minibatch = tree_map(next, shuffled_minibatches)
                gradients = grad(loss_fn)(model.params, minibatch, key)
                updates, opt_state = optimiser.update(gradients, opt_state)
                model.params = optax.apply_updates(model.params, updates)
            self.train_loss.append(float(loss_fn(model.params, train_data, key)))
            self.test_loss.append(float(loss_fn(model.params, test_data, key)))
            if verbose:
                self.print_losses()
        toc = perf_counter()
        self.time = toc - tic
        return self
