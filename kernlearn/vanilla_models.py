import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)

from kernlearn.utils import pdist, pdisp


class CuckerSmale:
    """Cucker-Smale dynamics."""

    def __init__(self, K=0.0, beta=0.5, sigma=1.0):
        self.string = "cucker-smale"
        self.K = K
        self.beta = beta
        self.sigma = sigma
        self.params = {"K": self.K, "beta": self.beta, "sigma": self.sigma}

    def phi(self, r, params):
        return params["K"] * (params["sigma"] + r**2) ** (-params["beta"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        dvdt = jnp.mean(self.phi(r, params)[..., None] * pdisp(v, v), axis=0)
        dxdt = v
        return dxdt, dvdt


class CuckerSmaleRayleigh:
    """Cucker-Smale dynamics with Rayleigh-type friction force."""

    def __init__(self, K=0.0, beta=0.5, p=0.0, kappa=0.0):
        self.string = "cucker-smale-rayleigh"
        self.K = K
        self.beta = beta
        self.p = p
        self.kappa = kappa
        self.params = {"K": self.K, "beta": self.beta, "p": self.p, "kappa": self.kappa}

    def phi(self, r, params):
        return params["K"] * (1 + r**2) ** (-params["beta"])

    def f(self, state, time, params):
        x, v = state
        r = pdist(x, x)
        F = params["kappa"] * v * (1 - jnp.linalg.norm(v) ** params["p"])
        dvdt = F + jnp.mean(self.phi(r, params)[..., None] * pdisp(v, v), axis=1)
        dxdt = v
        return dxdt, dvdt
