import pytest
import jax.numpy as jnp
from jax import random
from scipy.spatial import KDTree
from scipy.spatial.distance import pdist, squareform

from kernlearn import utils

rng = random.PRNGKey(0)


@pytest.mark.parametrize("N", [10, 50])
def test_pdist(N):
    N = 10
    d = 2
    x = random.uniform(rng, (N, d))
    scipy_pdist = squareform(pdist(x))
    utils_pdist = utils.pdist(x, x)
    assert jnp.allclose(scipy_pdist, utils_pdist)


@pytest.mark.parametrize("N", [10, 50])
def test_nearest_neighbours(N):
    d = 2
    k = N // 2
    x = random.uniform(rng, (N, d))
    scipy_dists, scipy_inds = KDTree(x).query(x, k)
    utils_dists, utils_inds = utils.nearest_neighbours(utils.pdist(x, x), k)
    assert jnp.allclose(scipy_dists, utils_dists)
    assert jnp.allclose(scipy_inds, utils_inds)
