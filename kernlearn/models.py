from functools import partialmethod

import jax
import jax.numpy as jnp
from jax import random
import haiku as hk

from kernlearn.utils import pdist, pdisp, nearest_neighbours
from kernlearn.chebyshev import chebval


class CuckerSmale:
    """Cucker-Smale dynamics with k nearest neighbours."""

    def __init__(self, k=None):
        self.id = "cucker-smale"
        self.k = k
        self.params = {
            "K": 0.046,
            "beta": 2.835,
        }

    def phi(self, r, params):
        return params["K"] * (1.0 + r**2) ** (-params["beta"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        if self.k is not None:
            rknn, idx = nearest_neighbours(r, self.k)
            dvdt = jnp.mean(
                self.phi(rknn, params)[..., None] * -pdisp(v, v, idx), axis=1
            )
        else:
            dvdt = jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleRayleigh:
    """Cucker-Smale dynamics with Rayleigh-type friction force."""

    def __init__(self, seed=0, K=0.0, beta=0.5, p=8e-1, kappa=2.7e-2):
        self.id = "cucker-smale-rayleigh"
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        self.K = K
        self.beta = beta
        self.p = p
        self.kappa = kappa
        self.params = {
            "K": self.K,
            "beta": self.beta,
            "p": self.p,
            "kappa": self.kappa,
        }

    def phi(self, r, params):
        return params["K"] * (1 + r**2) ** (-params["beta"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        F = (
            params["kappa"]
            * v
            * (1 - jnp.linalg.norm(v, axis=-1, keepdims=True) ** params["p"])
        )
        dvdt = F + jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleCheb:
    """Cucker-Smale dynamics with Chebychev approximation of phi."""

    def __init__(self, seed=0, n=5, k=None):
        self.id = "cucker-smale-rayleigh-cheb"
        self.k = k
        self.n = n  # polynomial order
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        self.params = {
            "c": 0.1
            * random.normal(self.rng, (n + 1,))
            * (1e-8 + jnp.exp(-jnp.arange(1, n + 2)))
        }
        self.hparams = {"n": self.n}

    def phi(self, r, params):
        return chebval(r, params["c"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        if self.k is not None:
            rknn, idx = nearest_neighbours(r, self.k)
            dvdt = jnp.mean(
                self.phi(rknn, params)[..., None] * -pdisp(v, v, idx), axis=1
            )
        else:
            dvdt = jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleRayleighCheb:
    """Cucker-Smale dynamics with Rayleigh-type friction force and
    Chebychev approximation of phi."""

    def __init__(self, seed=0, n=5, p=8e-1, kappa=2.7e-2):
        self.id = "cucker-smale-rayleigh-cheb"
        self.n = n  # polynomial order
        self.p = p
        self.kappa = kappa
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        self.params = {
            "c": random.normal(self.rng, (n + 1,)) / jnp.sqrt(n + 1),
            "p": self.p,
            "kappa": self.kappa,
        }
        self.hparams = {
            "n": self.n,
        }

    def phi(self, r, params):
        return chebval(r, params["c"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        F = (
            params["kappa"]
            * v
            * (1 - jnp.linalg.norm(v, axis=-1, keepdims=True) ** params["p"])
        )
        dvdt = F + jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class FirstOrderPredatorPrey:
    """First order predator prey model from Chen & Kolokolnikov (2014)."""

    def __init__(self, a=0.5, b=1.0, c=1.0, p=2.0):
        self.id = "first-order-predator-prey"
        self.a = a
        self.b = b
        self.c = c
        self.p = p
        self.params = {"a": self.a, "b": self.b, "c": self.c, "p": self.p}

    def phi_a(self, r, params):
        r_valid = jnp.where(r != 0, r, 1)
        return jnp.where(r != 0, 1 / r_valid**2 - params["a"], 0)

    def phi_b(self, r, params):
        r_valid = jnp.where(r != 0, r, 1)
        return jnp.where(r != 0, params["b"] / r_valid**2, 0)

    def phi_c(self, r, params):
        r_valid = jnp.where(r != 0, r, 1)
        return jnp.where(r != 0, params["c"] / r_valid ** params["p"], 0)

    def f(self, state, time, params):
        x, u = state[:-1], state[-1]
        xx = pdisp(x, x)
        xu = pdisp(x, u)
        rxx = pdist(xx)
        rxu = pdist(xu)
        Fa = jnp.mean(self.phi_a(rxx, params)[..., None] * -xx, axis=0)
        Fb = self.phi_b(rxu, params) * xu.squeeze()
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))


class CuckerSmaleNN:
    """Cucker-Smale dynamics with neural network approximation of phi."""

    def __init__(
        self,
        k=None,
        hidden_layer_sizes=[8],
        activation="tanh",
        dropout_rate=0.0,
        seed=0,
    ):
        self.id = "cucker-smale-nn"
        self.k = k
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = getattr(jax.nn, activation)
        self.dropout_rate = dropout_rate
        self.hparams = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "dropout_rate": dropout_rate,
        }
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        _train_mlp = lambda r: hk.nets.MLP(
            hidden_layer_sizes + [1], activation=self.activation
        )(r, self.dropout_rate, self.rng)
        _mlp = lambda r: hk.nets.MLP(
            hidden_layer_sizes + [1], activation=self.activation
        )(
            r
        )  # no dropout
        self.train_mlp = hk.transform(_train_mlp)
        self.mlp = hk.transform(_mlp)
        self.params = self.mlp.init(self.rng, jnp.array([0.0]))

    def phi(self, r, params, training=False):
        if training:
            return self.train_mlp.apply(params, self.rng, r.reshape(-1, 1)).reshape(
                r.shape
            )
        else:
            return self.mlp.apply(params, self.rng, r.reshape(-1, 1)).reshape(r.shape)

    def _f(self, state, time, params, training):
        x, v = state
        r = pdist(x, x)
        if self.k is not None:
            rknn, idx = nearest_neighbours(r, self.k)
            dvdt = jnp.mean(
                self.phi(rknn, params, training)[..., None] * -pdisp(v, v, idx), axis=1
            )
        else:
            dvdt = jnp.mean(
                self.phi(r, params, training)[..., None] * -pdisp(v, v), axis=0
            )
        dxdt = v
        return dxdt, dvdt

    f = partialmethod(_f, training=False)
    f_training = partialmethod(_f, training=True)


class CuckerSmaleFNN:
    """Cucker-Smale dynamics with neural network approximation of phi and F."""

    def __init__(
        self, hidden_layer_sizes=[8], activation="tanh", dropout_rate=0.0, seed=0
    ):
        self.id = "cucker-smale-f-nn"
        self.hparams = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "dropout_rate": dropout_rate,
        }
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        self.dropout_rate = dropout_rate
        _mlp1 = lambda r: hk.nets.MLP(
            hidden_layer_sizes + [1], activation=self.activation
        )(r, self.dropout_rate, self.rng)
        _mlp2 = lambda r: hk.nets.MLP(
            hidden_layer_sizes + [2], activation=self.activation
        )(r, self.dropout_rate, self.rng)
        self.mlp1 = hk.transform(_mlp1)
        self.mlp2 = hk.transform(_mlp2)
        self.params = {
            "phi": self.mlp1.init(self.rng, jnp.array([0.0])),
            "F": self.mlp2.init(self.rng, jnp.array([0.0, 0.0])),
        }

    def phi(self, r, params):
        return self.mlp1.apply(params["phi"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def F(self, v, params):
        return self.mlp2.apply(params["F"], self.rng, v)

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        dvdt = self.F(v, params) + jnp.mean(
            self.phi(r, params)[..., None] * -pdisp(v, v), axis=0
        )
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleRayleighNN:
    """Cucker-Smale dynamics with Rayleigh-type friction force and neural network
    approximation of phi."""

    def __init__(
        self,
        hidden_layer_sizes=[8],
        activation="tanh",
        dropout_rate=0.0,
        seed=0,
        p=8e-1,
        kappa=2.7e-2,
    ):
        self.id = "cucker-smale-rayleigh-nn"
        self.hparams = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "dropout_rate": dropout_rate,
        }
        self.output_sizes = hidden_layer_sizes + [1]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.kappa = kappa
        self.dropout_rate = dropout_rate
        self.p = p
        self.rng = random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(
            r, self.dropout_rate, self.rng
        )
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
        F = (
            params["kappa"]
            * v
            * (1 - jnp.linalg.norm(v, axis=-1, keepdims=True) ** params["p"])
        )
        dvdt = F + jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class FirstOrderNeuralODE:
    """Entire right-hand side of first order ODE approximated as a neural network.

    Input and output size should be N x d."""

    def __init__(self, N, d, hidden_layer_sizes=[64], activation="tanh", seed=0):
        self.id = "firs-order-neural-ode"
        self.output_sizes = hidden_layer_sizes + [N * d]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp = hk.transform(_mlp)
        dummy_input = jnp.zeros(self.output_sizes[-1])
        self.params = self.mlp.init(self.rng, dummy_input)

    def f(self, state, time, params):
        state_and_time = jnp.concatenate((state.ravel(), jnp.array(time)))
        return self.mlp.apply(params, self.rng, state_and_time).reshape(state.shape)


class SecondOrderNeuralODE:
    """Entire right-hand side of second order ODE approximated as a neural network.

    Input and output size should be 2 x N x d."""

    def __init__(
        self, N, d, hidden_layer_sizes=[64], activation="tanh", dropout_rate=0.0, seed=0
    ):
        self.id = "second-order-neural-ode"
        self.output_sizes = hidden_layer_sizes + [2 * N * d]
        self.activation = getattr(jax.nn, activation)
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(
            r, self.dropout_rate, self.rng
        )
        self.mlp = hk.transform(_mlp)
        dummy_input = jnp.zeros(self.output_sizes[-1])
        self.params = self.mlp.init(self.rng, dummy_input)

    def f(self, state, time, params):
        x_and_v = jnp.concatenate(state)
        state_and_time = jnp.hstack([state, jnp.array(time)])
        state_and_time = jnp.concatenate((x_and_v.ravel(), time))
        forward = self.mlp.apply(params, self.rng, state_and_time)
        return jnp.vsplit(forward.reshape(x_and_v.shape), 2)


class FirstOrderPredatorPreyNN3:
    """First order predator prey model from Chen & Kolokolnikov (2014) with
    neural network approximation of all three kernels."""

    def __init__(self, hidden_layer_sizes=[8], activation="tanh", seed=0):
        self.id = "first-order-predator-prey-nn"
        self.output_sizes = hidden_layer_sizes + [1]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp_a = hk.transform(_mlp)
        self.mlp_b = hk.transform(_mlp)
        self.mlp_c = hk.transform(_mlp)
        self.params = {
            "a": self.mlp_a.init(self.rng, jnp.array([0.0])),
            "b": self.mlp_b.init(self.rng, jnp.array([0.0])),
            "c": self.mlp_c.init(self.rng, jnp.array([0.0])),
        }

    def phi_a(self, r, params):
        return self.mlp_a.apply(params["a"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def phi_b(self, r, params):
        return self.mlp_b.apply(params["b"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def phi_c(self, r, params):
        return self.mlp_c.apply(params["c"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def f(self, state, time, params):
        x, u = state[:-1], state[-1]
        xx = pdisp(x, x)
        xu = pdisp(x, u)
        rxx = pdist(xx)
        rxu = pdist(xu)
        Fa = jnp.mean(self.phi_a(rxx, params)[..., None] * -xx, axis=0)
        Fb = self.phi_b(rxu, params) * xu.squeeze()
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))


class SecondOrderPredatorPreyNN3:
    """Second order predator prey model from Chen & Kolokolnikov (2014) with
    neural network approximation of all three forces."""

    def __init__(self, hidden_layer_sizes=[8], activation="tanh", seed=0):
        self.id = "second-order-predator-prey-nn"
        self.output_sizes = hidden_layer_sizes + [1]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.rng = random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(r)
        self.mlp_a = hk.transform(_mlp)
        self.mlp_b = hk.transform(_mlp)
        self.mlp_c = hk.transform(_mlp)
        self.params = {
            "a": self.mlp_a.init(self.rng, jnp.array([0.0])),
            "b": self.mlp_b.init(self.rng, jnp.array([0.0])),
            "c": self.mlp_c.init(self.rng, jnp.array([0.0])),
        }

    def phi_a(self, r, params):
        return self.mlp_a.apply(params["a"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def phi_b(self, r, params):
        return self.mlp_b.apply(params["b"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def phi_c(self, r, params):
        return self.mlp_c.apply(params["c"], self.rng, r.reshape(-1, 1)).reshape(
            r.shape
        )

    def f(self, state, time, params):
        x, u = state[:-1], state[-1]
        xx = pdisp(x, x)
        xu = pdisp(x, u)
        rxx = pdist(xx)
        rxu = pdist(xu)
        Fa = jnp.mean(self.phi_a(rxx, params)[..., None] * -xx, axis=0)
        Fb = self.phi_b(rxu, params) * xu.squeeze()
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))
