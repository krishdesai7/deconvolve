"""Weighted maximum mean discrepancy, with everything constant precomputed.

This exists because a GAN's loss is not a model-selection signal. It
oscillates around its equilibrium by construction; a flat curve cannot be
told from a stalled one; and `log 2 - BCE` estimates a divergence only when
`d` is optimal, which nothing reports. MMD has the property the loss was
assumed to have: zero iff the distributions match, monotone in mismatch, and
no adversary or optimization involved -- it is a closed-form functional of
the weights.

Only the MC-side weights change between epochs, and almost everything else
collapses:

* `term_xx` is data-side only, so it is a *scalar*, computed once.
* `k_xy` never has to be stored. The data-side weights are uniform and fixed,
  so `mean_i k(x_i, y_.)` is a *vector* of length m. Storing the matrix costs
  1.07 GB at m=16384 and 10x the per-evaluation time, for nothing.
* `k_yy` is the only matrix that must survive.
* `diag(k_yy)` is exactly `len(sigmas)` for a sum of RBFs. It is stored
  anyway so the estimator stays correct if the kernel is ever swapped.

Squared distances use `||a||^2 + ||b||^2 - 2ab^T` rather than the obvious
broadcast subtraction, which would materialize an (n, m, d) intermediate --
6.4 GB at m=16384.
"""

from __future__ import annotations

import functools
import operator
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

    from jaxtyping import Array, Float
    from numpy.typing import NDArray

# Bracketing the median heuristic. A single bandwidth is blind at every other
# scale, and a sum of RBFs is still a characteristic kernel, so this is free.
_SCALES: Final[tuple[float, ...]] = (0.5, 2.0**-0.5, 1.0, 2.0**0.5, 2.0)

# Below this the unbiased estimator's denominator is meaningless.
_MIN_DENOM: Final[float] = 1e-6

# Every dot in this module is a cancellation: `||a||^2 + ||b||^2 - 2ab^T` and
# `term_xx + term_yy - 2 v_xy w` both have answers far smaller than the O(1)
# terms they are assembled from. XLA's *default* for a float32 dot on an A100
# is TF32 -- a 10-bit mantissa on the inputs -- so the error is relative to the
# large terms and lands whole on the small answer. Measured by emulating TF32
# rounding on these exact inputs: squared distances come out ~1e-2 off the
# broadcast form and MMD^2 ~5e-5 relative, one to two orders past what
# `tests/test_mmd.py` pins. On CPU this flag is a no-op, which is why a green
# local suite said nothing and the GPU suite did.
_PRECISION: Final[jax.lax.Precision] = jax.lax.Precision.HIGHEST


class MMDCache(NamedTuple):
    """Everything about a comparison that does not depend on the weights.

    Four arrays and nothing else, so this is a clean pytree and
    `weighted_mmd` can take it as a jitted argument. The bandwidths are
    static and are deliberately not a field: as leaves they would be traced.
    """

    k_yy: Float[Array, "m m"]
    v_xy: Float[Array, " m"]
    diag_yy: Float[Array, " m"]
    term_xx: Float[Array, ""]


def squared_distances(
    a: Float[Array | NDArray[np.single], "n d"],
    b: Float[Array | NDArray[np.single], "m d"],
    /,
) -> Float[Array, "n m"]:
    """Pairwise squared distances via expansion, never an (n, m, d) tensor."""
    return (
        jnp.sum(a=a**2, axis=1)[:, None]
        + jnp.sum(a=b**2, axis=1)[None, :]
        - 2.0 * jnp.matmul(a, b=b.T, precision=_PRECISION)
    )


def median_bandwidth(x: Float[Array | NDArray[np.single], "n d"], /) -> float:
    """`sigma` such that the kernel at the median distance is `exp(-1)`.

    Computed from the data side alone. `x_data` is fixed by `data_seed`, so
    every hyperparameter arm shares an identical kernel; a pooled heuristic
    would drift with the MC side and make arms incomparable.
    """
    return float(jnp.sqrt(jnp.median(squared_distances(x, x)) / 2.0))


def bandwidths(
    x: Float[Array | NDArray[np.single], "n d"], /, *, scales: Sequence[float] = _SCALES
) -> tuple[float, ...]:
    median: float = median_bandwidth(x)
    return tuple(median * s for s in scales)


def subsample_indices(seed: int, n: int, m: int, /) -> NDArray[np.intp]:
    """A fixed, reproducible draw of at most `m` of `n` rows, without replacement."""
    return (
        np.random.default_rng(seed).permutation(x=n)[: min(m, n)].astype(dtype=np.intp)
    )


def _kernel(
    a: Float[Array | NDArray[np.single], "n d"],
    b: Float[Array | NDArray[np.single], "m d"],
    sigmas: Sequence[float],
    /,
) -> Float[Array, "n m"]:
    d2: Float[Array, "n m"] = squared_distances(a, b)
    # Not `sum(...)`: its zero-valued start defaults to `Literal[0]`, which
    # both checkers correctly refuse to unify with `Array`.
    return functools.reduce(operator.add, (jnp.exp(-d2 / (2.0 * s**2)) for s in sigmas))


def build_cache(
    x_data: Float[Array | NDArray[np.single], "n d"],
    y_mc: Float[Array | NDArray[np.single], "m d"],
    /,
    *,
    sigmas: Sequence[float],
) -> MMDCache:
    """Precompute every weight-independent term of the comparison."""
    n: int = x_data.shape[0]
    k_xx: Float[Array, "n n"] = _kernel(x_data, x_data, sigmas)
    # The standard unbiased U-statistic: the diagonal is a self-comparison and
    # carries no information about the distribution.
    term_xx: Float[Array, ""] = (jnp.sum(a=k_xx) - jnp.trace(a=k_xx)) / (n * (n - 1))
    del k_xx
    k_yy: Float[Array, "m m"] = _kernel(y_mc, y_mc, sigmas)
    return MMDCache(
        k_yy=k_yy,
        v_xy=jnp.mean(a=_kernel(x_data, y_mc, sigmas), axis=0),
        diag_yy=cast(typ="Array", val=jnp.diagonal(k_yy)),
        term_xx=term_xx,
    )


@jax.jit
def weighted_mmd(
    cache: MMDCache, raw_w: Float[Array, " m"], /
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Unbiased weighted MMD-squared and the effective sample size.

    `raw_w` is the generator's raw output; normalization happens here so no
    caller has to remember which convention applies.

    ESS is returned rather than folded into the score. The biased estimator
    would absorb `sum(w^2)` into the metric and silently penalize weight
    concentration; given that the adversarial objective is linear in `w` and
    maximized at a simplex vertex, concentration is a thing to *measure*, not
    to mix into the number being minimized.
    """
    w: Float[Array, " m"] = raw_w / jnp.sum(a=raw_w)
    sum_w_sq: Float[Array, ""] = jnp.sum(a=w**2)
    denom: Float[Array, ""] = 1.0 - sum_w_sq

    # Double `where`: the guarded branch must not be evaluated at denom = 0,
    # because jnp.where computes both sides and a NaN would propagate.
    safe: Float[Array, ""] = jnp.where(denom > _MIN_DENOM, denom, 1.0)
    k_w: Float[Array, " m"] = jnp.matmul(a=cache.k_yy, b=w, precision=_PRECISION)
    term_yy: Float[Array, ""] = (
        jnp.matmul(a=w, b=k_w, precision=_PRECISION) - jnp.sum(a=w**2 * cache.diag_yy)
    ) / safe
    mmd2: Float[Array, ""] = (
        cache.term_xx
        + term_yy
        - 2.0 * jnp.matmul(a=cache.v_xy, b=w, precision=_PRECISION)
    )
    return jnp.where(denom > _MIN_DENOM, mmd2, jnp.inf), 1.0 / sum_w_sq


def mmd_curve(
    cache: MMDCache, raw_w: Float[Array, "epochs m"], /
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """`weighted_mmd` over a stack of per-epoch weight vectors."""
    mmds, esss = jax.vmap(weighted_mmd, in_axes=(None, 0))(cache, raw_w)
    return np.asarray(a=mmds, dtype=np.double), np.asarray(esss, dtype=np.double)
