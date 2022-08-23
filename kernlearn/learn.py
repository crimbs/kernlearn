from functools import partial
from time import perf_counter

import optax
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.tree_util import tree_map, tree_leaves, tree_reduce
from tqdm import tqdm

from kernlearn.utils import get_batches, integrate


class Optimisation:
    def __init__(
        self,
        optimiser: str = "adam",
        learning_rate: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 16,
    ):
        assert optimiser in optax.__all__, "Must be an optax optimiser."
        self.optimiser = optimiser
        self.learning_rate = learning_rate
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)

    def print_losses(self):
        print(f"Train Loss: {self.train_loss[-1]}\tTest Loss: {self.test_loss[-1]}")

    def log_lik(self, params, data):
        x_pred, v_pred = self.predict(params, data)
        x_sq_error = jnp.sum((x_pred - data["x"]) ** 2, axis=-1)
        v_sq_error = jnp.sum((v_pred - data["v"]) ** 2, axis=-1)
        return -0.5 * (jnp.sum(x_sq_error) + jnp.sum(v_sq_error))

    def init_mle_loss_fn(self):
        """Maximum likelihood estimate"""
        self.loss_fn = lambda params, data, _: -self.log_lik(params, data)

    def init_reg_mle_loss_fn(self, reg: float = 0.1):
        """Maximum likelihood estimate with regularisation"""

        def l2_norm(params):
            return 0.5 * sum(jnp.sum(p**2) for p in tree_leaves(params))

        self.loss_fn = lambda params, data, _: -self.log_lik(
            params, data
        ) + reg * l2_norm(params)

    def init_elbo_loss_fn(self, num_samples: int = 5, beta: float = 0.001):
        """Negative evidence lower bound"""

        def kl(params):
            # Analytical KL divergence between two Gaussians.
            # See Appendix B of https://arxiv.org/pdf/1312.6114.pdf
            def _kl(mean, log_std):
                var = jnp.exp(log_std) ** 2
                return -0.5 * jnp.sum(1 + jnp.log(var) - mean**2 - var)

            out = tree_map(_kl, params["mean"], params["log_std"])
            return tree_reduce(lambda a, b: a + b, out)

        def diag_gaussian_sample(rng, params):
            # Draw a random sample from diagonal multivariate Gaussian.

            def _diag_gaussian_sample(mean, log_std):
                return mean + jax.random.normal(rng, mean.shape) * jnp.exp(log_std)

            return tree_map(_diag_gaussian_sample, params["mean"], params["log_std"])

        def sample_log_lik(rng, data, params):
            # Single-sample Monte Carlo estimate of the log likelihood.
            sample = diag_gaussian_sample(rng, params)
            return self.log_lik(sample, data)

        def expected_log_lik(rng, data, params):
            # Average over a batch of random samples.
            rngs = jax.random.split(rng, num_samples)
            vectorized_log_lik = vmap(sample_log_lik, in_axes=(0, None, None))
            return jnp.mean(vectorized_log_lik(rngs, data, params))

        def elbo(params, data, epoch):
            rng = jax.random.PRNGKey(epoch)
            return expected_log_lik(rng, data, params) - beta * kl(params)

        self.loss_fn = lambda params, data, epoch: -elbo(params, data, epoch)
        self.sample = diag_gaussian_sample  # For use in plotting

    def fit(self, model, train_data, test_data, verbose=False):
        # Define the forward pass
        self.predict = partial(integrate, dynamics=jit(model.f))
        # Intialise optimiser
        optimiser = getattr(optax, self.optimiser)(self.learning_rate)
        opt_state = optimiser.init(model.params)
        # Record initial loss value pre training
        loss_fn = jit(self.loss_fn)
        self.train_loss = [float(loss_fn(model.params, train_data, 0))]
        self.test_loss = [float(loss_fn(model.params, test_data, 0))]
        n_batches = len(train_data["t"]) // self.batch_size
        # Main loop
        tic = perf_counter()
        for epoch in tqdm(range(self.n_epochs), desc=type(model).__name__):
            shuffled_minibatches = get_batches(train_data, n_batches, seed=epoch)
            for _ in range(n_batches):
                minibatch = tree_map(next, shuffled_minibatches)
                gradients = grad(loss_fn)(model.params, minibatch, epoch)
                updates, opt_state = optimiser.update(gradients, opt_state)
                model.params = optax.apply_updates(model.params, updates)
            self.train_loss.append(float(loss_fn(model.params, train_data, epoch)))
            self.test_loss.append(float(loss_fn(model.params, test_data, epoch)))
            if verbose:
                self.print_losses()
        toc = perf_counter()
        self.time = toc - tic
        return self
