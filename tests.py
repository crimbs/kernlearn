from kernlearn import utils

# from kernlearn.nn_models import *

from scipy.spatial.distance import pdist, squareform
import numpy as np
import jax.numpy as jnp
from jax import random
from jax import config

config.update("jax_enable_x64", True)


def test_pdist():
    seed = 0
    rng = random.PRNGKey(seed)
    N = 10
    d = 2
    x = random.uniform(rng, (N, d))
    scipy_pdist = squareform(pdist(np.asarray(x)))
    utils_pdist = utils.pdist(x, x)
    assert jnp.allclose(scipy_pdist, utils_pdist)
