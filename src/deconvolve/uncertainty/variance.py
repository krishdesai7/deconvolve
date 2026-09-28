"""The two-way variance decomposition, and the covariance the field assumes away.

A run of RAN is a function of two independent random draws: the dataset `D` it
saw and the initialization seed `S` it started from. Write one run's output as
`T(D, S)`. The law of total variance splits its variance exactly:

    Var[T]  =  E_D[ Var_S(T | D) ]  +  Var_D[ E_S(T | D) ]

Both terms need *both* axes to compute, which is why two one-dimensional
sweeps do not answer the question. Decompose a run as

    T = mu + a(D) + b(S) + eps(D, S)

and the natural pair of one-dimensional measurements --- vary the seed at one
fixed dataset, vary the dataset at one fixed seed --- estimate
`sigma_b^2 + sigma_eps^2` and `sigma_a^2 + sigma_eps^2`. Adding them in
quadrature gives `sigma_a^2 + sigma_b^2 + 2 sigma_eps^2`, which overstates the
total by exactly the interaction term: the part of a run that depends on the
*combination* of dataset and seed and is attributable to neither. In a min-max
game that term is not small: the effect of an init seed already fails to
transfer across `lr_g` arms (measured `r = +0.04`), which is the same
phenomenon in a different coordinate. So the naive sum is not a conservative
approximation to quote --- it is a wrong number in a known direction, and the
only way to know by how much is to run the grid.

`decompose` therefore reads a full `B x S` grid and returns all three
components, from the balanced two-way crossed random-effects ANOVA:

    E[MS_data]        = sigma_eps^2 + S * sigma_a^2
    E[MS_init]        = sigma_eps^2 + B * sigma_b^2
    E[MS_interaction] = sigma_eps^2

`S` is a crossed factor rather than a nested one because a seed means the same
thing in every cell: `keras.utils.set_random_seed(s)` puts the identical
initial weights on the network whichever dataset it is about to see. A seed
main effect is therefore a real thing that can exist, and the design can see it.

There is one run per cell, so `sigma_eps^2` is the interaction together with
everything else that varies from cell to cell and with neither axis: the
per-cell split, batch order and selection subsamples (`design.split_seed`) and
any run-to-run hardware noise. Nothing here can separate them, and all of it
is method variance.

The evaluation set's own sampling variance is common to every cell, so it
cancels from all three components; `evaluation_variance` and
`evaluation_covariance` supply it separately. See `notes/agent/seeding.md`
for the two axes and `XLA_FLAGS=--xla_gpu_deterministic_ops=true` for the residual GPU
nondeterminism.

Components are moment estimators, not variances of anything, so an unlucky
grid can return a small negative value where the truth is near zero. They are
reported raw rather than clamped: a negative component is information about
the precision of the estimate, and hiding it behind a `max(0, .)` turns "we
cannot resolve this" into "this is exactly zero".
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..coretypes import EventArray


class VarianceComponents(NamedTuple):
    """The three sources, elementwise over whatever trailing shape was passed.

    Every field is a variance, not a standard deviation, because variances are
    what add. Take the square root at the point of reporting.
    """

    data: NDArray[np.double]
    init: NDArray[np.double]
    interaction: NDArray[np.double]

    @property
    def total(self) -> NDArray[np.double]:
        return self.data + self.init + self.interaction

    @property
    def naive_quadrature(self) -> NDArray[np.double]:
        """What summing the two one-dimensional sweeps would have given.

        `(sigma_a^2 + sigma_eps^2) + (sigma_b^2 + sigma_eps^2)`, which exceeds
        the true total by one interaction term. Reported next to `total` so
        the size of the double-count is visible rather than argued about.
        """
        return self.total + self.interaction


class Covariances(NamedTuple):
    """The same three components as `K x K` matrices over binned observables.

    Diagonals agree with `VarianceComponents` by construction;
    `tests/test_uncertainty.py` checks exactly that. The off-diagonals are the
    point: a bin-by-bin error bar throws them away.
    """

    data: NDArray[np.double]
    init: NDArray[np.double]
    interaction: NDArray[np.double]

    @property
    def total(self) -> NDArray[np.double]:
        return self.data + self.init + self.interaction


def _grid(t: NDArray[np.double] | EventArray, /) -> NDArray[np.double]:
    """Validate a `(B, S, ...)` design grid and widen it to float64.

    Both axes need at least two levels: with `B = 1` there is no dataset
    contrast to measure and with `S = 1` the interaction is confounded with
    the seed effect, so the mean squares have zero degrees of freedom and the
    components are undefined rather than merely imprecise.
    """
    grid: NDArray[np.double] = np.asarray(a=t, dtype=np.double)
    if grid.ndim < 2:
        raise ValueError(f"expected a (B, S, ...) grid of runs, got shape {grid.shape}")
    n_data, n_init = grid.shape[:2]
    if n_data < 2 or n_init < 2:
        raise ValueError(
            f"the decomposition needs at least 2 datasets and 2 seeds, got "
            f"B={n_data} and S={n_init}; two one-dimensional sweeps cannot "
            "separate the interaction from the main effects"
        )
    if not np.all(a=np.isfinite(grid)):
        raise ValueError("the design grid contains non-finite values")
    return grid


def _residual(grid: NDArray[np.double], /) -> NDArray[np.double]:
    """`t - mean_over_seeds - mean_over_datasets + grand_mean`, per cell."""
    grand: NDArray[np.double] = grid.mean(axis=(0, 1))
    return grid - grid.mean(axis=1)[:, None] - grid.mean(axis=0)[None, :] + grand


def decompose(t: NDArray[np.double] | EventArray, /) -> VarianceComponents:
    """Split the variance of a `(B, S, ...)` grid into its three sources.

    Elementwise over the trailing axes, so one call handles a scalar summary
    per observable, a `K`-vector of bin contents, or anything else with a run
    at each `(dataset, seed)`.
    """
    grid: NDArray[np.double] = _grid(t)
    n_data, n_init = grid.shape[:2]
    grand: NDArray[np.double] = grid.mean(axis=(0, 1))

    ms_data: NDArray[np.double] = (
        n_init * np.square(grid.mean(axis=1) - grand).sum(axis=0) / (n_data - 1)
    )
    ms_init: NDArray[np.double] = (
        n_data * np.square(grid.mean(axis=0) - grand).sum(axis=0) / (n_init - 1)
    )
    ms_interaction: NDArray[np.double] = np.square(_residual(grid)).sum(axis=(0, 1)) / (
        (n_data - 1) * (n_init - 1)
    )

    return VarianceComponents(
        data=cast(typ="NDArray[np.double]", val=(ms_data - ms_interaction) / n_init),
        init=cast(typ="NDArray[np.double]", val=(ms_init - ms_interaction) / n_data),
        interaction=ms_interaction,
    )


def _cov(rows: NDArray[np.double], /, *, ddof: int) -> NDArray[np.double]:
    """`np.cov` with the orientation pinned and a 1x1 result kept 2-D."""
    return np.atleast_2d(np.cov(m=rows, rowvar=False, ddof=ddof))


def component_covariances(t: NDArray[np.double] | EventArray, /) -> Covariances:
    """The three components as full `K x K` matrices, from a `(B, S, K)` grid.

    The dataset-averaged spectrum carries its own share of the interaction ---
    `mean_S eps` does not vanish at finite `S` --- so the raw between-dataset
    covariance estimates `Cov_a + Cov_eps / S` and has to be corrected by the
    interaction covariance before it means what its name says. Skipping that
    step inflates the very off-diagonals this function exists to measure, in
    the direction that makes the argument look better.
    """
    grid: NDArray[np.double] = _grid(t)
    if grid.ndim != 3:
        raise ValueError(
            f"expected a (B, S, K) grid of binned spectra, got shape {grid.shape}"
        )
    n_data, n_init, n_bins = grid.shape

    residual: NDArray[np.double] = _residual(grid).reshape(-1, n_bins)
    # Sum of squares over B*S cells, but only (B-1)(S-1) of them are free.
    interaction: NDArray[np.double] = (residual.T @ residual) / (
        (n_data - 1) * (n_init - 1)
    )

    between_data: NDArray[np.double] = _cov(grid.mean(axis=1), ddof=1)
    between_init: NDArray[np.double] = _cov(grid.mean(axis=0), ddof=1)
    return Covariances(
        data=cast(typ="NDArray[np.double]", val=between_data - interaction / n_init),
        init=cast(typ="NDArray[np.double]", val=between_init - interaction / n_data),
        interaction=interaction,
    )


def correlation(cov: NDArray[np.double], /) -> NDArray[np.double]:
    """Normalize a covariance to unit diagonal, leaving `nan` where it cannot.

    A corrected component covariance can carry a non-positive diagonal entry
    in a bin whose true variance is near zero, and there is no correlation to
    quote there. `nan` says so; a clamp would draw a confident zero.
    """
    variance: NDArray[np.double] = np.diag(v=cov)
    # `np.where` would evaluate `sqrt` on the negative entries as well and warn
    # about it; the mask keeps the warning-free path and the `nan` both.
    scale: NDArray[np.double] = np.full(shape=variance.shape, fill_value=np.nan)
    positive: NDArray[np.bool] = variance > 0
    scale[positive] = np.sqrt(variance[positive])
    return cov / np.outer(a=scale, b=scale)


def quantile_edges(column: EventArray, /, *, n_bins: int) -> NDArray[np.double]:
    """Equal-occupancy bin edges, so the covariance is not dominated by tails.

    A `K x K` covariance estimated from `B` replicates needs every bin to carry
    enough events that its content is a measurement rather than a coin flip;
    linear edges over a jet observable put most bins in a tail where it is the
    latter. Duplicate edges (a discrete observable such as `mult`, or a spike)
    collapse, so the returned array can be shorter than `n_bins + 1` --- read
    the bin count off the result rather than assuming it.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be at least 1, got {n_bins}")
    edges: NDArray[np.double] = np.quantile(
        a=np.asarray(a=column, dtype=np.double),
        q=np.linspace(start=0.0, stop=1.0, num=n_bins + 1),
    )
    # Nudge the outer edges so `np.histogram`'s half-open bins keep the
    # extreme events, which the quantile puts exactly on the boundary.
    unique: NDArray[np.double] = np.unique(ar=edges)
    if unique.size < 2:
        raise ValueError("column is constant; there is nothing to bin")
    span: np.double = unique[-1] - unique[0]
    unique[0] -= span * 1e-9  # pyrefly: ignore[unknown-argument-type]
    unique[-1] += span * 1e-9  # pyrefly: ignore[unknown-argument-type]
    return unique


def binned_spectra(
    column: EventArray,
    weights: NDArray[np.double] | EventArray,
    /,
    *,
    edges: NDArray[np.double],
) -> NDArray[np.double]:
    """Histogram one observable under each run's weights, normalized to unit sum.

    `weights` is `(..., n_events)` and the result is `(..., K)`, so a `(B, S)`
    design comes back as `(B, S, K)` ready for `decompose`. Every run weights
    the *same* `column`, so the across-run variance is a property of the
    unfolding rather than of the evaluation sample: the finite size of the
    common set shifts all runs together and cancels out of the contrast.
    """
    values: NDArray[np.double] = np.asarray(a=column, dtype=np.double)
    stack: NDArray[np.double] = np.asarray(a=weights, dtype=np.double)
    if stack.shape[-1] != values.shape[0]:
        raise ValueError(
            f"weights has {stack.shape[-1]} per run but the column has "
            f"{values.shape[0]} events; every run must weight the same events"
        )
    flat: NDArray[np.double] = stack.reshape(-1, values.shape[0])
    out: NDArray[np.double] = np.empty(shape=(flat.shape[0], edges.size - 1))
    for i, w in enumerate(iterable=flat):
        counts: NDArray[np.double] = np.histogram(a=values, bins=edges, weights=w)[0]
        out[i] = counts / (counts.sum() or 1.0)
    return out.reshape(*stack.shape[:-1], edges.size - 1)


def weighted_means(
    column: EventArray,
    weights: NDArray[np.double] | EventArray,
    /,
) -> NDArray[np.double]:
    """Each run's unfolded mean of one observable: the scalar summary.

    Binning is a choice, and a decomposition that depends on it invites the
    reply that a different binning would say something else. The weighted mean
    depends on none, so the summary table reports it; the binned covariance
    carries the off-diagonal argument.
    """
    values: NDArray[np.double] = np.asarray(a=column, dtype=np.double)
    stack: NDArray[np.double] = np.asarray(a=weights, dtype=np.double)
    return (stack @ values) / stack.sum(axis=-1)


def evaluation_variance(
    column: EventArray,
    weights: NDArray[np.double] | EventArray,
    /,
) -> NDArray[np.double]:
    """Each run's variance of its unfolded mean over repeated evaluation sets.

    The unfolded result is the evaluation set, reweighted; a different draw of
    those events would move it even with the weight function held fixed. That
    fluctuation is common to every cell, so it cancels from every component of
    `decompose` and has to be added to them separately. For a ratio estimator
    `m = sum(w z) / sum(w)` the delta method gives
    `sum(w^2 (z - m)^2) / sum(w)^2`, which grows as the weights concentrate.
    """
    values: NDArray[np.double] = np.asarray(a=column, dtype=np.double)
    stack: NDArray[np.double] = np.asarray(a=weights, dtype=np.double)
    flat: NDArray[np.double] = stack.reshape(-1, values.shape[0])
    means: NDArray[np.double] = weighted_means(column, flat)
    out: NDArray[np.double] = np.empty(shape=flat.shape[0])
    for i in range(flat.shape[0]):
        w: NDArray[np.double] = flat[i]
        spread: NDArray[np.double] = w * (values - means[i])
        total: np.double = w.sum()
        variance: np.double = np.sum(a=np.square(spread)) / np.square(total)
        out[i] = variance
    return out.reshape(stack.shape[:-1])


def evaluation_covariance(
    column: EventArray,
    weights: NDArray[np.double] | EventArray,
    /,
    *,
    edges: NDArray[np.double],
) -> NDArray[np.double]:
    """Each run's `K x K` covariance of `binned_spectra` over evaluation sets.

    The delta method applied to the normalized bin fractions
    `f_k = N_k / T`, with `N_k` the weight in bin `k` and `T` the weight in
    range: `(diag(h2) - f h2^T - h2 f^T + f f^T W2) / T^2`, where `h2` holds
    each bin's sum of squared weights and `W2` their total. Every row sums to
    zero, as the normalization requires. Binned with `np.histogram`, exactly
    as `binned_spectra` is, so the two describe the same bins.
    """
    values: NDArray[np.double] = np.asarray(a=column, dtype=np.double)
    stack: NDArray[np.double] = np.asarray(a=weights, dtype=np.double)
    flat: NDArray[np.double] = stack.reshape(-1, values.shape[0])
    n_bins: int = edges.size - 1
    out: NDArray[np.double] = np.empty(shape=(flat.shape[0], n_bins, n_bins))
    for i in range(flat.shape[0]):
        w: NDArray[np.double] = flat[i]
        w_sq: NDArray[np.double] = np.square(w)
        h: NDArray[np.double] = np.histogram(a=values, bins=edges, weights=w)[0]
        h2: NDArray[np.double] = np.histogram(a=values, bins=edges, weights=w_sq)[0]
        total: np.double = h.sum()
        if total <= 0:
            raise ValueError("a run puts no weight inside the bin edges")
        f: NDArray[np.double] = h / total
        cov: NDArray[np.double] = (
            np.diag(v=h2)
            - np.outer(a=f, b=h2)
            - np.outer(a=h2, b=f)
            + np.outer(a=f, b=f) * h2.sum()
        )
        scale: np.double = np.square(total)
        out[i] = cov / scale
    return out.reshape(*stack.shape[:-1], n_bins, n_bins)
