import json
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import top_k
from jax.tree_util import tree_map
from jax import config

config.update("jax_enable_x64", True)


def min_max_scaler(X):
    """Transform state by scaling each dimension to [0, 1]."""
    X_std = (X - X.min(axis=(0, 1))) / (X.max(axis=(0, 1)) - X.min(axis=(0, 1)))
    return X_std


def jax2list(params):
    return tree_map(
        lambda x: x.tolist(), params, is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )


def data_loader(fname):
    """Loads data from json file of lists and converts them into jax arrays."""
    data = json.load(open(fname, "r"))
    return tree_map(jnp.asarray, data, is_leaf=lambda x: isinstance(x, list))


def train_test_split(data, ind):
    train_data = tree_map(
        lambda x: x[:ind], data, is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )
    test_data = tree_map(
        lambda x: x[ind:], data, is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )
    return train_data, test_data


@jit
def pdist(a, b):
    """Computes pairwise distance matrix between a and b.

    Does so in a way that avoids nans when jax.grad is used. Makes use
    of the 'double where trick' [1] to ensure autodifferentiability.

    References
    ----------
    [1] https://github.com/google/jax/issues/5798#issuecomment-782862747
    """
    ab = a[:, None] - b
    ab_valid = jnp.where(ab != 0, ab, 1)
    return jnp.where(ab[..., 0] != 0, jnp.linalg.norm(ab_valid, axis=-1), 0)


@jit
def pdisp(a, b):
    """Computes pairwise displacement a-b."""
    return a[:, None] - b


def nearest_neighbours(x, k, c0=jnp.inf):
    # Test with r1, i1 = KDTree(x).query(x, k)
    full_dist_squared = jnp.sum((x[:, None] - x) ** 2, axis=-1)
    # Upper bound of perception (Shvydkoy & Tadmor, 2020, p. 5796)
    clipped_dist_squared = jnp.where(full_dist_squared < c0**2, full_dist_squared, 0)
    ind = jnp.argsort(clipped_dist_squared, axis=-1)[:, 1:k]
    dist_squared = jnp.take_along_axis(clipped_dist_squared, ind, axis=-1)
    return jnp.sqrt(dist_squared)


@partial(vmap, in_axes=(0, None))
def nearest_neighbours(x, hparams):
    full_dist_sq = jnp.sum((x[:, None] - x) ** 2, axis=-1)
    _, knn_idxs = top_k(-full_dist_sq, hparams["k"] + 1)
    dist_sq = jnp.take_along_axis(full_dist_sq, knn_idxs[:, 1:], axis=-1)
    return jnp.sqrt(dist_sq)
