import os
from time import perf_counter

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import jit
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from kernlearn import plot_utils

SEED = 0
PATH = os.path.join("figures", "other")


@jit
def jax_pdist(x):
    return jnp.linalg.norm(x[:, None] - x, axis=-1)


def scipy_pdist(x):
    return squareform(pdist(x))


def main():
    N = 10000
    M = 100
    rng = jax.random.PRNGKey(SEED)
    x = jax.random.uniform(rng, (N, 2))
    iterations = np.arange(M)
    jax_times = np.zeros(M)
    scipy_times = np.zeros(M)
    _ = jax_pdist(x)
    for i in tqdm(iterations):
        jax_tic = perf_counter()
        _ = jax_pdist(x)
        jax_toc = perf_counter()
        jax_times[i] = jax_toc - jax_tic
        scipy_tic = perf_counter()
        _ = scipy_pdist(x)
        scipy_toc = perf_counter()
        scipy_times[i] = scipy_toc - scipy_tic

    # Plot results
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.semilogy(
        iterations,
        np.cumsum(jax_times),
        color="k",
        label="Custom JIT-compiled pairwise distance",
    )
    ax.semilogy(
        iterations,
        np.cumsum(scipy_times),
        color="k",
        alpha=0.5,
        label=r"\texttt{scipy.spatial.distance.pdist}",
    )
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Wall Time (s)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(os.path.join(PATH, "jit.pdf"))


if __name__ == "__main__":
    main()
