import haiku as hk
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from kernlearn.utils import pdist, pdisp


class CuckerSmaleNN:
    """Cucker-Smale dynamics with neural network approximation of phi."""

    def __init__(self, layer_sizes=[8], activation="relu", seed=0):
        self.string = "cucker-smale-nn"
        self.output_sizes = layer_sizes + [1]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp = hk.transform(_mlp)
        self.params = self.mlp.init(self.rng, jnp.array([0.0]))

    def phi(self, r, params):
        return self.mlp.apply(params, self.rng, r.reshape(-1, 1)).reshape(r.shape)

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        dvdt = jnp.mean(self.phi(r, params)[..., None] * pdisp(v, v), axis=1)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleRayleighNN:
    """Cucker-Smale dynamics with Rayleigh-type friction force and neural network
    approximation of phi."""

    def __init__(self, layer_sizes=[8], activation="relu", seed=0, kappa=0.0, p=0.0):
        self.string = "cucker-smale-rayleigh-nn"
        self.output_sizes = layer_sizes + [1]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.kappa = kappa
        self.p = p
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp = hk.transform(_mlp)
        self.params = {
            "nn": self.mlp.init(self.rng, jnp.array([0.0])),
            "kappa": self.kappa,
            "p": self.p,
        }

    def phi(self, r, params):
        return self.mlp.apply(params["nn"], self.rng, r.reshape(-1, 1)).reshape(r.shape)

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        F = params["kappa"] * v * (1 - jnp.linalg.norm(v) ** params["p"])
        dvdt = F + jnp.mean(self.phi(r, params)[..., None] * pdisp(v, v), axis=1)
        dxdt = v
        return dxdt, dvdt


class FirstOrderNeuralODE:
    """Entire right-hand side of first order ODE approximated as a neural network.

    Input and output size should be N x d."""

    def __init__(self, N, d, layer_sizes=[64], activation="relu", seed=0):
        self.string = "firs-order-neural-ode"
        self.output_sizes = layer_sizes + [N * d]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp = hk.transform(_mlp)
        dummy_input = jnp.zeros(self.output_sizes[-1])
        self.params = self.mlp.init(self.rng, dummy_input)

    def f(self, state, time, params):
        forward = self.mlp.apply(params, self.rng, state.ravel())
        return jnp.vsplit(forward.reshape(state.shape), 2)


class SecondOrderNeuralODE:
    """Entire right-hand side of second order ODE approximated as a neural network.

    Input and output size should be 2 x N x d."""

    def __init__(self, N, d, layer_sizes=[64], activation="relu", seed=0):
        self.string = "second-order-neural-ode"
        self.output_sizes = layer_sizes + [2 * N * d]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp = hk.transform(_mlp)
        dummy_input = jnp.zeros(self.output_sizes[-1])
        self.params = self.mlp.init(self.rng, dummy_input)

    def f(self, state, time, params):
        x_and_v = jnp.concatenate(state)
        forward = self.mlp.apply(params, self.rng, x_and_v.ravel())
        return jnp.vsplit(forward.reshape(x_and_v.shape), 2)
