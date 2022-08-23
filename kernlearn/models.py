from functools import partialmethod

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import haiku as hk

from kernlearn.utils import pdist, pdisp, nearest_neighbours, normalise
from kernlearn.chebyshev import chebval


def rayleigh_friction(v, params):
    v_norm = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return params["kappa"] * v * (1 - v_norm ** params["p"])


def get_rayleigh_params(params, rng):
    rng, kappa_rng, p_rng = jax.random.split(rng, 3)
    params["kappa"] = jax.random.uniform(kappa_rng, (1,), maxval=0.1)
    params["p"] = jax.random.uniform(p_rng, (1,), maxval=1)


class CuckerSmale:
    """Cucker-Smale dynamics."""

    def __init__(self, seed=0, rayleigh=False, r0=jnp.inf):
        self.seed = seed
        self.rayleigh = rayleigh
        self.r0 = r0
        self.rng = jax.random.PRNGKey(self.seed)
        K_rng, beta_rng = jax.random.split(self.rng, 2)
        self.params = {
            "K": jax.random.uniform(K_rng, (1,), maxval=0.5),
            "beta": jax.random.uniform(beta_rng, (1,), maxval=3),
        }
        if self.rayleigh:
            get_rayleigh_params(self.params, self.rng)

    def phi(self, r, params):
        return params["K"] * (1.0 + r**2) ** (-params["beta"])

    def f(self, state, time, params, control):
        x, v = state
        r = pdist(x, x)
        a = jnp.where(r < self.r0, self.phi(r, params), 0)[..., None]
        N = jnp.count_nonzero(a, axis=1)
        F = rayleigh_friction(v, params) if self.rayleigh else 0.0
        dxdt = v
        dvdt = F + (1 / N) * jnp.sum(a * -pdisp(v, v), axis=1)
        return dxdt, dvdt


class CuckerSmalePoly:
    """Cucker-Smale dynamics with polynomial approximation of phi."""

    def __init__(self, seed=0, rayleigh=False, chebyshev=False, n=20, r0=jnp.inf):
        self.seed = seed
        self.rayleigh = rayleigh
        self.r0 = r0
        self.chebyshev = chebyshev
        self.evaluate_poly = chebval if self.chebyshev else jnp.polyval
        self.rng = jax.random.PRNGKey(self.seed)
        self.params = {
            "c": 1e-6 * jax.random.normal(self.rng, (n,)),
        }
        if self.rayleigh:
            get_rayleigh_params(self.params, self.rng)

    def phi(self, r, params):
        return self.evaluate_poly(params["c"], r)

    def f(self, state, time, params, control):
        x, v = state
        r = pdist(x, x)
        a = jnp.where(r < self.r0, self.phi(r, params), 0)[..., None]
        N = jnp.count_nonzero(a, axis=1)
        F = rayleigh_friction(v, params) if self.rayleigh else 0.0
        dxdt = v
        dvdt = F + (1 / N) * jnp.sum(a * -pdisp(v, v), axis=1)
        return dxdt, dvdt


class CuckerSmaleNN:
    """Cucker-Smale dynamics with neural network approximation of phi."""

    def __init__(
        self,
        seed=0,
        rayleigh=False,
        hidden_layer_sizes=[8],
        activation="elu",
        dropout_rate=0.0,
        r0=jnp.inf,
    ):
        self.seed = seed
        self.rayleigh = rayleigh
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.r0 = r0
        self.rng = jax.random.PRNGKey(self.seed)

        def network_fn(r):
            nn = hk.nets.MLP(
                self.hidden_layer_sizes + [1],
                activation=getattr(jax.nn, self.activation),
            )
            return nn(r)

        self.mlp = hk.transform(network_fn)
        self.params = {"nn": self.mlp.init(self.rng, jnp.array([0.0]))}
        if self.rayleigh:
            get_rayleigh_params(self.params, self.rng)

    def phi(self, r, params):
        out = self.mlp.apply(params["nn"], self.rng, r.reshape(-1, 1))
        return out.reshape(r.shape)

    def f(self, state, time, params, control):
        x, v = state
        r = pdist(x, x)
        a = jnp.where(r < self.r0, self.phi(r, params), 0)[..., None]
        N = jnp.count_nonzero(a, axis=1)
        F = rayleigh_friction(v, params) if self.rayleigh else 0.0
        dxdt = v
        dvdt = F + (1 / N) * jnp.sum(a * -pdisp(v, v), axis=1)
        return dxdt, dvdt


class CuckerSmaleKNN:
    """Cucker-Smale dynamics with k nearest neighbours."""

    def __init__(self, seed=0, k=7):
        self.seed = seed
        self.k = k
        self.rng = jax.random.PRNGKey(self.seed)
        K_rng, beta_rng = jax.random.split(self.rng, 2)
        self.params = {
            "K": jax.random.uniform(K_rng, (1,), maxval=0.5),
            "beta": jax.random.uniform(beta_rng, (1,), maxval=3),
        }

    def phi(self, r, params):
        return params["K"] * (1.0 + r**2) ** (-params["beta"])

    def f(self, state, time, params, control):
        x, v = state
        r = pdist(x, x)
        rknn, idx = nearest_neighbours(r, self.k)
        dvdt = jnp.mean(self.phi(rknn, params)[..., None] * -pdisp(v, v, idx), axis=1)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleAnticipation:
    """Cucker-Smale dynamics with anticipation."""

    def __init__(self, seed, tau=0.0):
        self.seed = seed
        self.tau = tau
        self.rng = jax.random.PRNGKey(self.seed)
        K_rng, beta_rng = jax.random.split(self.rng, 2)
        self.params = {
            "K": jax.random.uniform(K_rng, (1,), maxval=0.5),
            "beta": jax.random.uniform(beta_rng, (1,), maxval=3),
        }

    def phi(self, r, params):
        return params["K"] * (1.0 + r**2) ** (-params["beta"])

    def f(self, state, time, params, control):
        _x, v = state
        x = _x + self.tau * v
        r = pdist(x, x)
        dvdt = jnp.mean(self.phi(r, params)[..., None] * -pdisp(v, v), axis=1)
        dxdt = v
        return dxdt, dvdt


class MotschTadmor:
    """Motsch-Tadmor dynamics with CS interaction kernel."""

    def __init__(self, seed=0, rayleigh=False):
        self.seed = seed
        self.rayleigh = rayleigh
        self.rng = jax.random.PRNGKey(self.seed)
        K_rng, beta_rng = jax.random.split(self.rng, 2)
        self.params = {
            "K": jax.random.uniform(K_rng, (1,), maxval=0.5),
            "beta": jax.random.uniform(beta_rng, (1,), maxval=3),
        }
        if self.rayleigh:
            get_rayleigh_params(self.params, self.rng)

    def phi(self, r, params):
        return params["K"] * (1.0 + r**2) ** (-params["beta"])

    def f(self, state, time, params, control):
        x, v = state
        r = pdist(x, x)
        phi = self.phi(r, params)
        S = jnp.sum(phi, axis=1, keepdims=True)
        # S = x.shape[0]
        dvdt = (1 / S) * jnp.sum(phi[..., None] * -pdisp(v, v), axis=1)
        dxdt = v
        return dxdt, dvdt


class FirstOrderPredatorPrey:
    """First order predator prey model from Chen & Kolokolnikov (2014)."""

    def __init__(self, a=0.5, b=1.0, c=1.0, p=2.0):
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

    def f(self, state, time, params, control):
        x, u = state[:-1], state[-1]
        xx = pdisp(x, x)
        xu = pdisp(x, u)
        rxx = pdist(xx)
        rxu = pdist(xu)
        Fa = jnp.mean(self.phi_a(rxx, params)[..., None] * -xx, axis=0)
        Fb = self.phi_b(rxu, params) * xu.squeeze()
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))


class FirstOrderNeuralODE:
    """Entire right-hand side of first order ODE approximated as a neural network.

    Input and output size should be N x d."""

    def __init__(self, N, d, hidden_layer_sizes=[64], activation="tanh", seed=0):
        self.output_sizes = hidden_layer_sizes + [N * d]
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(
            self.output_sizes, activation=getattr(jax.nn, activation)
        )(r)
        self.mlp = hk.transform(_mlp)
        dummy_input = jnp.zeros(self.output_sizes[-1])
        self.params = self.mlp.init(self.rng, dummy_input)

    def f(self, state, time, params, control):
        state_and_time = jnp.concatenate((state.ravel(), jnp.array(time)))
        return self.mlp.apply(params, self.rng, state_and_time).reshape(state.shape)


class SecondOrderNeuralODE:
    """Entire right-hand side of second order ODE approximated as a neural network.

    Input and output size should be 2 x N x d."""

    def __init__(
        self, N, d, hidden_layer_sizes=[64], activation="tanh", dropout_rate=0.0, seed=0
    ):
        self.output_sizes = hidden_layer_sizes + [2 * N * d]
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(
            self.output_sizes, activation=getattr(jax.nn, activation)
        )(r, self.dropout_rate, self.rng)
        self.mlp = hk.transform(_mlp)
        dummy_input = jnp.zeros(self.output_sizes[-1])
        self.params = self.mlp.init(self.rng, dummy_input)

    def f(self, state, time, params, control):
        x_and_v = jnp.concatenate(state)
        state_and_time = jnp.concatenate((x_and_v.ravel(), time))
        forward = self.mlp.apply(params, self.rng, state_and_time)
        return jnp.vsplit(forward.reshape(x_and_v.shape), 2)


class FirstOrderPredatorPreyNN3:
    """First order predator prey model from Chen & Kolokolnikov (2014) with
    neural network approximation of all three kernels."""

    def __init__(self, hidden_layer_sizes=[8], activation="tanh", seed=0):
        self.output_sizes = hidden_layer_sizes + [1]
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(
            self.hidden_layer_sizes + [1], activation=getattr(jax.nn, activation)
        )(r)
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

    def f(self, state, time, params, control):
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
        self.output_sizes = hidden_layer_sizes + [1]
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _mlp = lambda r: hk.nets.MLP(
            self.output_sizes, activation=getattr(jax.nn, activation)
        )(r)
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

    def f(self, state, time, params, control):
        x, u = state[:-1], state[-1]
        xx = pdisp(x, x)
        xu = pdisp(x, u)
        rxx = pdist(xx)
        rxu = pdist(xu)
        Fa = jnp.mean(self.phi_a(rxx, params)[..., None] * -xx, axis=0)
        Fb = self.phi_b(rxu, params) * xu.squeeze()
        Fc = jnp.mean(self.phi_c(rxu, params)[..., None] * xu, axis=0)
        return jnp.vstack((Fa + Fb, Fc))


class SecondOrderSheep:
    """Just sheep with NN approximation of inter-sheep forces and
    force between sheep and dog."""

    def __init__(
        self,
        k=None,
        N=1,
        hidden_layer_sizes=[8],
        activation="tanh",
        dropout_rate=0.0,
        seed=0,
    ):
        self.k = k
        self.N = N
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _train_mlp = lambda r: hk.nets.MLP(
            self.hidden_layer_sizes + [1], activation=getattr(jax.nn, activation)
        )(r, self.dropout_rate, self.rng)
        _mlp = lambda r: hk.nets.MLP(
            self.hidden_layer_sizes + [1], activation=getattr(jax.nn, activation)
        )(
            r
        )  # no dropout
        self.Ftrain_mlp = hk.transform(_train_mlp)
        self.Gtrain_mlp = hk.transform(_train_mlp)
        self.Fmlp = hk.transform(_mlp)
        self.Gmlp = hk.transform(_mlp)
        Frng, Grng = jax.random.split(self.rng)
        self.params = {
            "mu": jax.random.uniform(self.rng, (self.N, 1)),  # friction coefficients
            "F": self.Fmlp.init(Frng, jnp.array([0.0])),
            "G": self.Gmlp.init(Grng, jnp.array([0.0])),
        }

    def F(self, r, params, training=False):
        if training:
            return self.Ftrain_mlp.apply(
                params["F"], self.rng, r.reshape(-1, 1)
            ).reshape(r.shape)
        else:
            return self.Fmlp.apply(params["F"], self.rng, r.reshape(-1, 1)).reshape(
                r.shape
            )

    def G(self, r, params, training=False):
        if training:
            return self.Gtrain_mlp.apply(
                params["G"], self.rng, r.reshape(-1, 1)
            ).reshape(r.shape)
        else:
            return self.Gmlp.apply(params["G"], self.rng, r.reshape(-1, 1)).reshape(
                r.shape
            )

    def _f(self, state, time, params, control, training):
        x, v = state
        u = map_coordinates(
            control.squeeze(), jnp.array([[time, time], [0.0, 1.0]]), order=1
        )
        xx = pdisp(x, x)
        xu = pdisp(x, u)
        rxx = pdist(xx)
        rxu = pdist(xu)
        xxhat = normalise(xx)
        xuhat = normalise(xu)
        Fprey = jnp.mean(self.F(rxx, params, training)[..., None] * xxhat, axis=1)
        Fpred = self.G(rxu, params, training) * xuhat.squeeze()
        dvdt = -params["mu"] * v + Fprey + Fpred
        dxdt = v
        return dxdt, dvdt

    f = partialmethod(_f, training=False)
    f_training = partialmethod(_f, training=True)


class SecondOrderDog:
    """Just dog with NN approximation of force."""

    def __init__(
        self,
        k=None,
        mu0=0.0,
        hidden_layer_sizes=[8],
        activation="tanh",
        dropout_rate=0.0,
        seed=0,
    ):
        self.k = k
        self.mu0 = mu0  # friction coefficient
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)
        _train_mlp = lambda r: hk.nets.MLP(
            self.hidden_layer_sizes + [1], activation=getattr(jax.nn, activation)
        )(r, self.dropout_rate, self.rng)
        _mlp = lambda r: hk.nets.MLP(
            self.hidden_layer_sizes + [1], activation=getattr(jax.nn, activation)
        )(
            r
        )  # no dropout
        self.Htrain_mlp = hk.transform(_train_mlp)
        self.Hmlp = hk.transform(_mlp)
        self.params = {
            "mu0": self.mu0,
            "H": self.Hmlp.init(self.rng, jnp.array([0.0])),
        }

    def H(self, r, params, training=False):
        if training:
            return self.Htrain_mlp.apply(
                params["H"], self.rng, r.reshape(-1, 1)
            ).reshape(r.shape)
        else:
            return self.Hmlp.apply(params["H"], self.rng, r.reshape(-1, 1)).reshape(
                r.shape
            )

    def _f(self, state, time, params, control, training):
        x, v = state
        N = control.shape[1]
        u = jnp.array(
            [
                map_coordinates(
                    control, jnp.array([[time, time], [i, i], [0.0, 1.0]]), order=1
                ).tolist()
                for i in range(N)
            ]
        )
        xu = pdisp(x, u)
        rxu = pdist(xu)
        xuhat = normalise(xu)
        force = jnp.mean(self.H(rxu, params, training) * xuhat.squeeze(), axis=1)
        dvdt = -params["mu0"] * v + force
        dxdt = v
        return dxdt, dvdt

    f = partialmethod(_f, training=False)
    f_training = partialmethod(_f, training=True)
