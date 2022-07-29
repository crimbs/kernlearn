import haiku as hk
import jax
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from kernlearn.utils import pdist, pdisp


class CuckerSmale:
    """Cucker-Smale dynamics."""

    def __init__(self, K=-0.05, beta=1.0, sigma=1.0):
        self.id = "cucker-smale"
        self.K = K
        self.beta = beta
        self.sigma = sigma
        self.params = {"K": self.K, "beta": self.beta, "sigma": self.sigma}

    def phi(self, r, params):
        return params["K"] * (params["sigma"] ** 2 + r**2) ** (-params["beta"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(pdisp(x, x))
        dvdt = jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleRayleigh:
    """Cucker-Smale dynamics with Rayleigh-type friction force."""

    def __init__(self, K=0.0, beta=0.5, sigma=1.0, p=8e-1, kappa=2.7e-2):
        self.id = "cucker-smale-rayleigh"
        self.K = K
        self.beta = beta
        self.sigma = sigma
        self.p = p
        self.kappa = kappa
        self.params = {
            "K": self.K,
            "beta": self.beta,
            "sigma": self.sigma,
            "p": self.p,
            "kappa": self.kappa,
        }

    def phi(self, r, params):
        return params["K"] * (params["sigma"] ** 2 + r**2) ** (-params["beta"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(pdisp(x, x))
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
        Fb = self.phi_b(rxu, params) * jnp.squeeze(xu)
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))


class CuckerSmaleNN:
    """Cucker-Smale dynamics with neural network approximation of phi."""

    def __init__(
        self, hidden_layer_sizes=[8], activation="tanh", dropout_rate=0.0, seed=0
    ):
        self.id = "cucker-smale-nn"
        self.hparams = {
            "hidden_layer_sizes": hidden_layer_sizes,
            "activation": activation,
            "dropout_rate": dropout_rate,
        }
        self.output_sizes = hidden_layer_sizes + [1]
        self.activation = getattr(jax.nn, activation)
        self.seed = seed
        self.dropout_rate = dropout_rate
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(self.output_sizes, activation=self.activation)(
            r, self.dropout_rate, self.rng
        )
        self.mlp = hk.transform(_mlp)
        self.params = self.mlp.init(self.rng, jnp.array([0.0]))

    def phi(self, r, params):
        return self.mlp.apply(params, self.rng, r.reshape(-1, 1)).reshape(r.shape)

    def f(self, state, time, params):
        x, v = state
        r = pdist(pdisp(x, x))
        dvdt = jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=0)
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
        self.rng = jax.random.PRNGKey(self.seed)
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
        r = pdist(pdisp(x, x))
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
        self.rng = jax.random.PRNGKey(self.seed)
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
        self.rng = jax.random.PRNGKey(self.seed)
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
        self.rng = jax.random.PRNGKey(self.seed)
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
        Fb = self.phi_b(rxu, params) * jnp.squeeze(xu)
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
        self.rng = jax.random.PRNGKey(self.seed)
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
        Fb = self.phi_b(rxu, params) * jnp.squeeze(xu)
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))
