import os
import json
from functools import partial

import jax.numpy as jnp
from jax import jit, vmap
from jax.lax import top_k
from jax.tree_util import tree_map
from jax import config

from kernlearn.plot_utils import loss_plot, trajectory_comparison_plot

config.update("jax_enable_x64", True)


def save_individual(path, model, opt, train_data, test_data):
    try:
        os.mkdir(path)
    except OSError:
        pass
    print(f"Saving to: {path}")
    json.dump(opt.hparams, open(os.path.join(path, "opt_hparams.txt"), "w"))
    # Print model-based hyperparameters (these might not exist for vanilla models)
    if hasattr(model, "hparams"):
        json.dump(model.hparams, open(os.path.join(path, "model_hparams.txt"), "w"))
    params_list = tree_map(
        lambda x: x.tolist(), model.params, is_leaf=lambda x: isinstance(x, jnp.ndarray)
    )
    json.dump(params_list, open(os.path.join(path, "params.txt"), "w"))
    json.dump(opt.train_loss, open(os.path.join(path, "train_loss.txt"), "w"))
    json.dump(opt.test_loss, open(os.path.join(path, "test_loss.txt"), "w"))
    loss_plot(path, opt.train_loss, opt.test_loss)
    x_train, v_train = opt.predict(train_data, model)
    x_test, v_test = opt.predict(test_data, model)
    trajectory_comparison_plot(
        path, train_data["x"], train_data["v"], x_train, v_train, "train_comparison.pdf"
    )
    trajectory_comparison_plot(
        path, test_data["x"], test_data["v"], x_test, v_test, "test_comparison.pdf"
    )


def min_max_scaler(X):
    """Transform state by scaling each dimension to [0, 1]."""
    X_std = (X - X.min(axis=(0, 1))) / (X.max(axis=(0, 1)) - X.min(axis=(0, 1)))
    return X_std


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
def pdist(x):
    """Computes pairwise distance matrix from pairwise displacement x.

    Makes use of the 'double where trick' [1] to ensure automatic
    differentiation doesn't produce NaNs since the gradient of sqrt(0)
    is undefined.

    References
    ----------
    [1] https://github.com/google/jax/issues/5798#issuecomment-782862747
    """
    x_valid = jnp.where(x != 0, x, 1)  # final argument here is arbitrary
    return jnp.where(x[..., 0] != 0, jnp.linalg.norm(x_valid, axis=-1), 0)


@jit
def pdisp(a, b):
    """Computes pairwise displacement (a - b).

    Last index must be dimension."""
    return a[:, None, :] - b


def nearest_neighbours(x, k, c0=jnp.inf):
    # Test with r1, i1 = KDTree(x).query(x, k)
    full_dist_squared = jnp.sum(pdisp(x, x) ** 2, axis=-1)
    # Upper bound of perception (Shvydkoy & Tadmor, 2020, p. 5796)
    clipped_dist_squared = jnp.where(full_dist_squared < c0**2, full_dist_squared, 0)
    ind = jnp.argsort(clipped_dist_squared, axis=-1)[:, 1:k]
    dist_squared = jnp.take_along_axis(clipped_dist_squared, ind, axis=-1)
    return jnp.sqrt(dist_squared)


@partial(vmap, in_axes=(0, None))
def nearest_neighbours(x, hparams):
    full_dist_sq = jnp.sum(pdisp(x, x) ** 2, axis=-1)
    _, knn_idxs = top_k(-full_dist_sq, hparams["k"] + 1)
    dist_sq = jnp.take_along_axis(full_dist_sq, knn_idxs[:, 1:], axis=-1)
    return jnp.sqrt(dist_sq)
