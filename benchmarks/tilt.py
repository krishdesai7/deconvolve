"""Exponential-tilt reweighting: the most regularized generator there is.

's generator is an arbitrary function `z -> w`, and `benchmarks/ceiling.py`
diagnostic D shows what that costs: the detector-level objective does not
identify the truth. The oracle particle-level likelihood ratio scores *worse*
on detector-level MMD than a trained Deconvolve does, on held-out events. Many weight
functions match `p(x)`; the right one is not the one the objective prefers.

This replaces the network with a `d`-parameter exponential family:

    w(z; b) = exp(-b . T(z)) / P(b)

`log P(b) = log E_gen[exp(-b . T(z))]` is a cumulant generating function, so it
is strictly convex, `grad log P(b) = -E_w[T(z)]`, and the gradient map is
therefore injective: **exactly one `b` reproduces any achievable value of
`E_w[T(z)]`**. With `T(z) = z` that is the Boltzmann/Gibbs reweighting -- the
maximum-entropy correction to `p_gen` subject to a first-moment constraint.

Two things this does *not* assume, both of which the output measures:

* **That the higher moments come along.** They do not. They are not left free
  either -- the tilt determines them from `p_gen` and `b`, and they are
  generally wrong. That is the price of the regularization.
* **That matching at detector level matches at particle level.** `b` cannot be
  fitted against `E[z_true]`; nothing has it. It is fitted against `E[S(x)]`,
  and that transfers to particle level only if the response is mean-preserving
  and shared between data and MC. For jets that is an assumption. The
  `first-moment transfer` block is the test of it, and it is the number this
  benchmark exists to produce.

**Why the restriction should help at all** is a dimension count, not an appeal
to smoothness. The unconstrained problem has a whole manifold of weight
functions matching `p(x)`, and D shows truth is not the point on it the
objective prefers. A `p`-parameter family intersected with the
detector-matching set is generically a set of isolated points: matching `p`
moments with `p` parameters is a square system, and the underdetermined problem
becomes an overdetermined one.

That also gives a *principled* capacity ladder, which `--degree` walks. Degree 1
matches first moments; degree 2 adds every `z_i z_j` and matches the second
moments too. Each rung buys one more matched moment and stays identifiable so
long as the parameter count is far below the number of events.

**The validation case has a known answer.** Every config in `params/` gives
`z_gen` and `z_true` different *covariances*, so their exact likelihood ratio is
`exp(quadratic in z)`. A linear tilt provably cannot represent it; a quadratic
tilt contains it exactly. So `--degree 2` on a Gaussian config should recover
essentially the whole discrepancy, and if it does not, the implementation is
wrong rather than the physics.

No adversary and no training: `P(b)` never has to be evaluated, because the
weights are normalized anyway, and the moment equations are a square root-find
whose Jacobian is available in closed form (`-Cov_w(S(x), T(z))`).

```zsh
uv run benchmarks/tilt.py --config params/2d_correlated.yaml --degree 1
uv run benchmarks/tilt.py --config params/2d_correlated.yaml --degree 2  # exact
uv run benchmarks/tilt.py --dataset jets --degree 1
uv run benchmarks/tilt.py --dataset jets --degree 2 --n-samples 1000000
```
"""

# pyrefly: ignore-errors[unknown-argument-type]
# -- argparse Namespace and numpy elementwise ops are Any under the stubs
from __future__ import annotations

import argparse
import logging
from itertools import combinations_with_replacement, starmap
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import deconvolve  # ruff: ignore[unused-import]  -- pins JAX_ENABLE_X64
import numpy as np
from deconvolve.coretypes import SUBSTRUCTURE_VARIABLES, Split
from deconvolve.data import DeconvolveDataset, load_jet_dataset
from deconvolve.evaluation.evaluate import _improvement, _wd_per_dim
from deconvolve.instrumentation.logging_config import configure_logging
from scipy.optimize import root

if TYPE_CHECKING:
    from deconvolve.coretypes import DatasetSplits, EventArray, Populations
    from numpy.typing import NDArray
    from scipy.optimize._root import OptimizeResult


logger: logging.Logger = logging.getLogger(name="deconvolve.tilt")

# The solve runs in float64 regardless of EVENT_DTYPE. Second moments of
# standardized data are O(1) but their *differences* are the residual, and
# float32 leaves only ~7 digits to find a root inside.
_SOLVE_DTYPE = np.double


class Tilt(NamedTuple):
    """A fitted tilt and everything needed to reproduce its weights."""

    beta: NDArray[np.double]
    degree: int
    center: NDArray[np.double]
    scale: NDArray[np.double]
    residual: float
    converged: bool
    message: str

    @property
    def n_params(self) -> int:
        return self.beta.size


def _design(a: NDArray[np.double], degree: int, /) -> NDArray[np.double]:
    """Sufficient statistics `T(a)`: every monomial in `a` up to `degree`.

    Each multiset of column indices appears once --
    `combinations_with_replacement`, not `product` -- because `a_i a_j` and
    `a_j a_i` are the same column, and a duplicated column makes the moment
    system singular rather than merely redundant.

    The count grows as `C(d + k, k) - 1`: in six dimensions that is 6, 27 and
    83 parameters at degrees 1, 2 and 3, against a sample of ~700k events.
    Identifiability needs the first number far below the second, which is what
    lets the ladder be walked rather than jumped.
    """
    blocks = [a]
    for k in range(2, degree + 1):
        idx = list(combinations_with_replacement(range(a.shape[1]), k))
        blocks.append(
            np.stack([np.prod(a[:, list(combo)], axis=1) for combo in idx], axis=1)
        )
    return np.concatenate(blocks, axis=1) if len(blocks) > 1 else a


def _standardize(
    a: NDArray[np.double], /
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """Center and scale, for conditioning only.

    A linear reparametrization of `T` is absorbed exactly by `b`, so this
    changes no weight the solve can produce -- it only stops the quadratic
    block from spanning several orders of magnitude, which is what a
    finite-difference-free Newton step needs to stay well-conditioned.
    """
    center = a.mean(axis=0)
    scale = a.std(axis=0)
    # A constant column carries no information and would divide by zero.
    return center, np.where(scale > 0.0, scale, 1.0)


def _weights(
    t_z: NDArray[np.double], beta: NDArray[np.double], /
) -> NDArray[np.double]:
    """`exp(-b . T(z))`, normalized to preserve the event count.

    Shifted by its own maximum before exponentiating. The shift cancels in the
    normalization exactly, and without it a moderately large `b` overflows to
    `inf` long before it reaches anything the solver would reject on merit.
    """
    log_w = -(t_z @ beta)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return cast("NDArray[np.double]", w * (len(w) / w.sum()))


def _moment_residual(
    beta: NDArray[np.double],
    t_z: NDArray[np.double],
    s_x: NDArray[np.double],
    target: NDArray[np.double],
    /,
) -> NDArray[np.double]:
    """`E_w[S(x_sim)] - E[S(x_data)]`: zero is a matched set of moments."""
    w = _weights(t_z, beta)
    return (w @ s_x) / w.sum() - target


def _moment_jacobian(
    beta: NDArray[np.double],
    t_z: NDArray[np.double],
    s_x: NDArray[np.double],
    _target: NDArray[np.double],
    /,
) -> NDArray[np.double]:
    """`-Cov_w(S(x), T(z))`, the exact derivative of `_moment_residual`.

    Differentiating a self-normalized expectation gives a covariance under the
    tilted measure, which is why this is closed-form rather than numeric: at
    degree 2 in six dimensions the parameter vector has 27 entries, and a
    finite-difference Jacobian would cost 28 passes over the sample per step.
    """
    w = _weights(t_z, beta)
    p = w / w.sum()
    mean_s = p @ s_x
    mean_t = p @ t_z
    return -((s_x * p[:, None]).T @ t_z - np.outer(mean_s, mean_t))


def fit_tilt(
    z_gen: EventArray, x_sim: EventArray, x_data: EventArray, /, *, degree: int
) -> Tilt:
    """Solve for the tilt whose reweighted `x_sim` moments match `x_data`.

    Square by construction: the statistics applied to `z` and to `x` have the
    same functional form, so there are exactly as many moment equations as
    parameters.
    """
    z: NDArray[np.double] = np.asarray(a=z_gen, dtype=_SOLVE_DTYPE)
    xs: NDArray[np.double] = np.asarray(a=x_sim, dtype=_SOLVE_DTYPE)
    xd: NDArray[np.double] = np.asarray(a=x_data, dtype=_SOLVE_DTYPE)

    center, scale = _standardize(z)
    x_center, x_scale = _standardize(xs)
    t_z: NDArray[np.double] = _design((z - center) / scale, degree)
    s_x: NDArray[np.double] = _design((xs - x_center) / x_scale, degree)
    target: NDArray[np.double] = _design((xd - x_center) / x_scale, degree).mean(axis=0)

    beta0: NDArray[np.double] = np.zeros(t_z.shape[1], dtype=_SOLVE_DTYPE)
    logger.info(
        "   degree %d: %d parameters, %d moment equations, %d MC events",
        degree,
        beta0.size,
        target.size,
        len(z),
    )
    sol: OptimizeResult[np.double, tuple[int]] = root(
        fun=_moment_residual,
        x0=beta0,
        args=(t_z, s_x, target),
        jac=_moment_jacobian,
        method="hybr",
        tol=1e-12,
    )
    residual = float(np.abs(np.asarray(a=sol.fun)).max())
    return Tilt(
        beta=np.asarray(a=sol.x, dtype=_SOLVE_DTYPE),
        degree=degree,
        center=center,
        scale=scale,
        residual=residual,
        converged=sol.success,
        message=sol.message,
    )


def tilt_weights(tilt: Tilt, z: EventArray, /) -> EventArray:
    """Apply a fitted tilt to any sample of nominal-level events."""
    scaled: NDArray[np.double] = (
        np.asarray(a=z, dtype=_SOLVE_DTYPE) - tilt.center
    ) / tilt.scale
    return cast("EventArray", _weights(_design(scaled, tilt.degree), tilt.beta))


def _report_first_moments(
    truth: EventArray,
    z_gen: EventArray,
    w: EventArray,
    variables: tuple[str, ...],
    /,
) -> None:
    """The claim under test: does matching `E[x]` deliver `E[z_true]`?

    Boltzmann guarantees a `b` exists reproducing any achievable
    `E_w[T(z)]`. It says nothing about whether the `b` found at *detector*
    level is that one. This block is the difference.
    """
    logger.info(msg="  first-moment transfer (the assumption, not the theorem)")
    logger.info(
        "    %-6s %10s %10s %10s %10s %8s",
        "var",
        "E[z_true]",
        "E[z_gen]",
        "E_w[z_gen]",
        "gap after",
        "closed",
    )
    before: NDArray[np.floating] = truth.mean(axis=0) - z_gen.mean(axis=0)
    after: NDArray[np.floating] = truth.mean(axis=0) - (w @ z_gen) / w.sum()
    for i, var in enumerate(iterable=variables):
        closed: float = (
            100.0 * (1.0 - abs(after[i]) / abs(before[i])) if before[i] else np.nan
        )
        logger.info(
            "    %-6s %10.5f %10.5f %10.5f %10.2e %7.1f%%",
            var,
            truth.mean(axis=0)[i],
            z_gen.mean(axis=0)[i],
            (w @ z_gen)[i] / w.sum(),
            after[i],
            closed,
        )
    logger.info(
        "    %-6s %10s %10s %10s %10.2e %7.1f%%",
        "max",
        "",
        "",
        "",
        np.abs(after).max(),
        100.0 * (1.0 - np.abs(after).max() / np.abs(before).max()),
    )


def _report_level(
    name: str,
    reference: EventArray,
    comparison: EventArray,
    w: EventArray,
    variables: tuple[str, ...],
    /,
) -> float:
    before = _wd_per_dim(ref=reference, comp=comparison)
    after = _wd_per_dim(ref=reference, comp=comparison, weights=w)
    logger.info("  %s level", name)
    for i, var in enumerate(variables):
        logger.info(
            "    %-6s W %.5f -> %.5f  (%+.1f%%)",
            var,
            before[i],
            after[i],
            _improvement(before[i], after[i]),
        )
    mean = float(np.mean(list(starmap(_improvement, zip(before, after, strict=True)))))
    logger.info("    %-6s %+.1f%%", "mean", mean)
    return mean


def _load(args: argparse.Namespace) -> tuple[DatasetSplits, tuple[str, ...]]:
    """Either data source, reduced to splits plus the column names."""
    if args.dataset == "jets":
        chosen = set(args.variables or SUBSTRUCTURE_VARIABLES)
        # Canonical order, as `cli._canonical_variables` fixes it: these names
        # index columns, so a permutation is a different dataset.
        variables = tuple(v for v in SUBSTRUCTURE_VARIABLES if v in chosen)
        splits, _dim, _std = load_jet_dataset(
            n_samples=args.n_samples,
            batch_size=args.batch_size,
            variables=variables,
            seed=args.data_seed,
        )
        return splits, variables
    if args.config is None:
        raise SystemExit("--config is required for --dataset gaussian")
    splits = DeconvolveDataset(
        batch_size=args.batch_size, seed=args.data_seed
    ).generate_gaussian_dataset(config_path=args.config, n_samples=args.n_samples)
    dim = splits.train.as_arrays().z.shape[1]
    return splits, tuple(f"z{i}" for i in range(dim))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ = parser.add_argument(
        "--dataset", choices=("gaussian", "jets"), default="gaussian"
    )
    _ = parser.add_argument("--config", type=Path, default=None)
    _ = parser.add_argument(
        "--var", action="append", dest="variables", choices=SUBSTRUCTURE_VARIABLES
    )
    _ = parser.add_argument("--degree", type=int, choices=(1, 2, 3), default=1)
    _ = parser.add_argument("--n-samples", type=int, default=1_000_000)
    _ = parser.add_argument("--batch-size", type=int, default=1024)
    _ = parser.add_argument("--data-seed", type=int, default=42)
    args = parser.parse_args()

    configure_logging(level="info")
    splits, variables = _load(args)
    # Fitted on train, scored on test -- the same protocol `ceiling.py`
    # diagnostic B uses, so the two sets of numbers are comparable.
    train_pop: Populations = splits.select(Split.TRAIN).partition()
    test_pop: Populations = splits.select(Split.TEST).partition()
    logger.info(
        "%d events over %s, %d train / %d test (MC side)",
        args.n_samples,
        ", ".join(variables),
        len(train_pop.mc),
        len(test_pop.mc),
    )

    logger.info("")
    logger.info("Fitting the tilt on detector-level moments (train split)")
    tilt = fit_tilt(train_pop.mc.z, train_pop.mc.x, train_pop.data, degree=args.degree)
    logger.info(
        "   %s  max |residual| %.3e  (%s)",
        "converged" if tilt.converged else "DID NOT CONVERGE",
        tilt.residual,
        tilt.message.strip(),
    )
    logger.info(
        "   b = %s", np.array2string(tilt.beta, precision=4, max_line_width=100)
    )

    w = tilt_weights(tilt, test_pop.mc.z)
    ess = float(w.sum() ** 2 / np.square(w).sum())
    logger.info("")
    logger.info("Scored on the test split")
    logger.info(
        "  weights: min %.3g  max %.3g  ESS %.0f / %d (%.1f%%)",
        w.min(),
        w.max(),
        ess,
        len(w),
        100.0 * ess / len(w),
    )
    _report_first_moments(test_pop.require_truth(), test_pop.mc.z, w, variables)
    particle = _report_level(
        "particle", test_pop.require_truth(), test_pop.mc.z, w, variables
    )
    detector = _report_level("detector", test_pop.data, test_pop.mc.x, w, variables)

    logger.info("")
    logger.info("SUMMARY  degree %d, %d parameters", args.degree, tilt.n_params)
    logger.info("  particle %+.1f%%   detector %+.1f%%", particle, detector)
    logger.info(
        "  A Gaussian config's exact likelihood ratio is exp(quadratic), so "
        "degree 2 should"
    )
    logger.info(
        "  recover nearly all of it there. On jets, compare against the unconstrained"
    )
    logger.info(
        "  ceiling from `ceiling.py` diagnostic B "
        "(%+.1f%% particle, %+.1f%% detector).",
        93.2,
        82.8,
    )


if __name__ == "__main__":
    main()
