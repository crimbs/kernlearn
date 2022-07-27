import jax.numpy as jnp
from jax import jit
from jax import random
from jax.experimental.ode import odeint
from jax import config

config.update("jax_enable_x64", True)

from kernlearn.utils import min_max_scaler


def generate_data(model, seed=0, nsteps=95, N=50, d=2, ss=0):
    """Generate synthetic data.

    Params
    ------
    model : Class
    seed : Random seed for PRNG key
    nsteps : number of time steps
    N : number of agents
    d : spatial dimensions
    ss : Time to reach steady state

    Returns
    -------
    data : Dictionary of data
    """
    key = random.PRNGKey(seed)
    x0 = jnp.zeros((N, d))
    v0 = random.uniform(key, (N, d))
    initial_state = (x0, v0)
    params = model.params
    dynamics = jit(model.f)
    t = jnp.arange(nsteps + ss, dtype=jnp.float64)
    x, v = odeint(dynamics, initial_state, t, params)
    x_out = min_max_scaler(x[ss:])
    v_out = min_max_scaler(v[ss:])
    t_out = t[ss:]
    data = {"x": x_out, "v": v_out, "t": t_out}
    return data
