"""The following is taken from a pull request [1] by user gderossi to include
Chebyshev polynomials as part of Jax, under Apache License, Version 2.0. It is
essentially a Jax implementation of the numpy.chebval function.

[1] https://github.com/google/jax/pull/11093
"""

from functools import partial
import jax.numpy as jnp
from jax import jit
from jax import lax


@partial(jit, static_argnames=("tensor",))
def chebval(c, x, tensor=True):
    """
    Evaluate a Chebyshev series at points x.

    Parameters
    ----------
    c : Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n].
    x : If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar.
    tensor : If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`.
    """
    c = jnp.asarray(c)
    x = jnp.asarray(x)
    if x.ndim != 0 and tensor:
        c = c.reshape(c.shape + (1,) * x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2 * x
        c0 = c[-2]
        c1 = c[-1]

        tmp = c0
        c0 = c[-3] - c1
        c1 = tmp + c1 * x2
        if len(c) == 3:
            return c0 + c1 * x

        tmp = c0
        c0 = c[-4] - c1
        c1 = tmp + c1 * x2

        def body_fun(i, val):
            return (c[-i] - val[1], val[0] + val[1] * x2)

        c0, c1 = lax.fori_loop(5, len(c) + 1, body_fun, (c0, c1))

        return c0 + c1 * x
