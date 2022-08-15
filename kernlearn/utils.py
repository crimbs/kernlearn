import os
import json

import jax.numpy as jnp
from jax import jit
from jax.lax import top_k
from jax.tree_util import tree_map


@jit
def pdisp(a: jnp.array, b: jnp.array, idx: jnp.array = None):
    """Computes pairwise displacement (a - b).

    Parameters
    ----------
    a : shape (N, d)
    b : shape (N, d)
    idx : (optional) indices of k-nearest neighbours
    """
    if idx is not None:
        b = jnp.take(b, idx, axis=0)
    return a[:, None, :] - b


@jit
def pdist(a: jnp.array, b: jnp.array = None, idx: jnp.array = None):
    """Computes pairwise distance between a and b.

    If b is not given then assumes a is pairwise displacement.

    Parameters
    ----------
    a : shape (N, d)
    b : (optional) shape (N, d)
    idx : (optional) indices of k-nearest neighbours

    Returns
    -------
    distance : shape (N, N)

    Notes
    -----
    Makes use of the 'double where trick' [1] to ensure automatic
    differentiation doesn't produce NaNs since the gradient of sqrt(0)
    is undefined.
    [1] https://github.com/google/jax/issues/5798#issuecomment-782862747
    """
    displacement = a if b is None else pdisp(a, b, idx)
    masked = jnp.where(displacement != 0, displacement, 1)
    mask = displacement[..., 0] != 0
    distance = jnp.where(mask, jnp.linalg.norm(masked, axis=-1), 0)
    return distance


def nearest_neighbours(distance: jnp.array, k: int):
    """Given a distance matrix, finds the k-nearest neighbours (inc. self)"""
    dists, inds = top_k(-distance, k)
    return -dists, inds


def normalise(x):
    x_nonzero = x != 0
    masked = jnp.where(x_nonzero, x, 1)
    _normalise = lambda x: x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    xhat = jnp.where(x_nonzero, _normalise(masked), 0)
    return xhat


def data_loader(fname: str):
    """Loads data from json file of lists and converts them into jax arrays."""
    data = json.load(open(fname, "r"))
    return tree_map(jnp.asarray, data, is_leaf=lambda x: isinstance(x, list))


def train_test_split(data: dict, ind: int):
    """Splits a dataset of jax arrays into train and test at a given index."""
    train_data = tree_map(
        lambda x: x[:ind], data, is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )
    test_data = tree_map(
        lambda x: x[ind:], data, is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )
    return train_data, test_data


def save_hyperparameters(opt, path):
    json.dump(opt.__dict__, open(os.path.join(path, type(opt).__name__ + ".json"), "w"))
