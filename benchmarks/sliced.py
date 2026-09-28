"""Are the per-axis metrics missing joint structure?

Every number in `deconvolve.evaluation.evaluate` is computed one coordinate
axis at a time -- `_wd_per_dim`, `_js_per_dim`, `_triangular_per_dim` all
loop over columns. That
makes the whole metric suite blind to correlation by construction: two
distributions with identical marginals and different joint structure score
identically on all of it. A reweighting that fixes every marginal and leaves
the correlations wrong would be reported as a complete success.

The sliced Wasserstein distance is the cheapest fix. Project both samples onto
random unit directions, take the 1-D Wasserstein distance along each, average.
Axis-aligned metrics are the special case where the directions are the basis
vectors; drawing them uniformly on the sphere is what makes the metric see the
off-diagonal structure.

**Weighted, because Deconvolve emits weights rather than events.** The textbook
`mean(|sort(x) - sort(y)|)` is the equal-size uniform-weight case. Using it
here would mean discarding `w` -- measuring the wrong distribution -- or
resampling events to represent it, which injects sampling noise into the
quantity being measured. `w1_weighted` is the same estimator `scipy` computes
and `deconvolve.evaluation.evaluate` already relies on, vectorised over projections.

**Standardised against the reference.** Directions are drawn on the sphere, so
the axes have to be commensurable; without it whichever observable carries the
largest numerical scale dominates every projection and the metric quietly
becomes a measurement of that one axis. Improvement percentages are reported
rather than raw distances, which makes them comparable with the per-axis
numbers in `metrics.json` despite the change of units.

float64 on the host, not JAX: `CLAUDE.md`'s Precision section pins the *data*
to float32 and leaves the *scores* in float64, which is what scipy already
returns for every other metric here. A 400k-element cumulative sum is also
exactly where float32 accumulation error would show up.

```zsh
uv run benchmarks/sliced.py --run-dir runs/2026-…
uv run benchmarks/sliced.py --run-dir runs/… --n-projections 512 --repeats 8
uv run benchmarks/sliced.py --run-dir runs/… --subsample 50000
```
"""

# pyrefly: ignore-errors[unknown-argument-type]
# -- run config is JSON (dict[str, Any]); numpy stubs return Any for elementwise ops
from __future__ import annotations

import os

os.environ.setdefault(key="KERAS_BACKEND", value="jax")  # pyrefly: ignore[unused-call-result]

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import deconvolve  # ruff: ignore[unused-import]  -- pins the backend and the dtype
import keras
import numpy as np
from deconvolve.coretypes import Split, artifacts_dir
from deconvolve.data import DeconvolveDataset, load_jet_dataset
from deconvolve.data.config import gaussian_config_from_run_config
from deconvolve.instrumentation.logging_config import configure_logging
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from typing import Any

    from deconvolve.coretypes import DatasetSplits, Populations
    from numpy.typing import NDArray

logger: logging.Logger = logging.getLogger("deconvolve.sliced")


def w1_weighted(
    u: NDArray[np.floating],
    v: NDArray[np.floating],
    /,
    *,
    u_weights: NDArray[np.floating] | None = None,
    v_weights: NDArray[np.floating] | None = None,
) -> np.double:
    """1-D Wasserstein-1 distance between two weighted empirical measures.

    `integral |F_u(t) - F_v(t)| dt`, evaluated on the merged support: sort the
    pooled values, accumulate each side's normalised weights into a step CDF,
    and integrate the gap. Identical to `scipy.stats.wasserstein_distance`,
    written out here so it can be vectorised over projections in one pass
    rather than called once per direction.

    Weights are normalised, so they are a measure and not a count: scaling all
    of them by a constant cannot change a distance.
    """
    u = np.asarray(a=u, dtype=np.double)
    v = np.asarray(a=v, dtype=np.double)
    wu: NDArray[np.double] = (
        np.full(u.shape, fill_value=1.0 / u.size)
        if u_weights is None
        else np.asarray(a=u_weights, dtype=np.double) / np.sum(u_weights)
    )
    wv: NDArray[np.double] = (
        np.full(v.shape, fill_value=1.0 / v.size)
        if v_weights is None
        else np.asarray(a=v_weights, dtype=np.double) / np.sum(a=v_weights)
    )

    pooled: NDArray[np.double] = np.concatenate([u, v])
    order: NDArray[np.intp] = np.argsort(a=pooled, kind="stable")
    values: NDArray[np.double] = pooled[order]
    cdf_u: NDArray[np.double] = np.cumsum(
        a=np.concatenate([wu, np.zeros_like(a=wv)])[order]
    )
    cdf_v: NDArray[np.double] = np.cumsum(
        a=np.concatenate([np.zeros_like(a=wu), wv])[order]
    )
    return np.sum(a=np.abs(cdf_u[:-1] - cdf_v[:-1]) * np.diff(a=values))


def _directions(seed: int, dim: int, n_projections: int) -> NDArray[np.double]:
    """Unit vectors drawn uniformly on the sphere."""
    raw: NDArray[np.double] = np.random.default_rng(seed).normal(
        size=(n_projections, dim)
    )
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def sliced_wasserstein(
    ref: NDArray[np.floating],
    comp: NDArray[np.floating],
    /,
    *,
    seed: int,
    n_projections: int = 128,
    comp_weights: NDArray[np.floating] | None = None,
) -> float:
    """Mean 1-D Wasserstein distance over random projections.

    Both samples are standardised by the *reference's* per-axis mean and
    standard deviation first, so a direction on the sphere weights the
    observables comparably rather than by whatever units they arrived in.
    """
    ref = np.asarray(a=ref, dtype=np.double)
    comp = np.asarray(a=comp, dtype=np.double)
    centre: NDArray[np.double] = ref.mean(axis=0)
    scale: NDArray[np.double] = np.where(ref.std(axis=0) > 0.0, ref.std(axis=0), 1.0)

    dirs: NDArray[np.double] = _directions(seed, ref.shape[1], n_projections)
    ref_proj: NDArray[np.double] = ((ref - centre) / scale) @ dirs.T
    comp_proj: NDArray[np.double] = ((comp - centre) / scale) @ dirs.T
    return float(
        np.mean(
            a=[
                w1_weighted(ref_proj[:, k], comp_proj[:, k], v_weights=comp_weights)
                for k in range(n_projections)
            ]
        )
    )


class Comparison(NamedTuple):
    """One level's before/after, with the projection Monte-Carlo spread."""

    before: float
    after: float
    before_sd: float
    after_sd: float

    @property
    def improvement_pct(self) -> float:
        return 100.0 * (self.before - self.after) / self.before


def compare(
    ref: NDArray[np.floating],
    comp: NDArray[np.floating],
    weights: NDArray[np.floating],
    /,
    *,
    n_projections: int,
    repeats: int,
) -> Comparison:
    """Sliced distance before and after reweighting, over several draws.

    Repeated across projection seeds because the estimate is an average over
    finitely many directions and carries Monte-Carlo error of its own; a single
    number would invite reading a difference that is only the draw.
    """
    before: list[float] = [
        sliced_wasserstein(ref, comp, seed=s, n_projections=n_projections)
        for s in range(repeats)
    ]
    after: list[float] = [
        sliced_wasserstein(
            ref, comp, seed=s, n_projections=n_projections, comp_weights=weights
        )
        for s in range(repeats)
    ]
    return Comparison(
        before=float(np.mean(a=before)),
        after=float(np.mean(a=after)),
        before_sd=float(np.std(a=before, ddof=1)) if repeats > 1 else 0.0,
        after_sd=float(np.std(a=after, ddof=1)) if repeats > 1 else 0.0,
    )


def _standardise(
    ref: NDArray[np.floating], comp: NDArray[np.floating], /
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    ref = np.asarray(a=ref, dtype=np.double)
    comp = np.asarray(a=comp, dtype=np.double)
    centre: NDArray[np.double] = ref.mean(axis=0)
    scale: NDArray[np.double] = np.where(ref.std(axis=0) > 0.0, ref.std(axis=0), 1.0)
    return (ref - centre) / scale, (comp - centre) / scale


def axis_wasserstein(
    ref: NDArray[np.floating],
    comp: NDArray[np.floating],
    /,
    *,
    comp_weights: NDArray[np.floating] | None = None,
) -> np.double:
    """Mean 1-D Wasserstein distance over the coordinate axes.

    What `metrics.json` reports, standardised the same way as the sliced number
    so the two live on one scale and comparing them is about geometry rather
    than units.
    """
    r, c = _standardise(ref, comp)
    return np.mean(
        a=[
            w1_weighted(r[:, i], c[:, i], v_weights=comp_weights)
            for i in range(r.shape[1])
        ]
    )


class Floors(NamedTuple):
    """What each metric reads when both samples are the same distribution."""

    sliced: float
    axis: float


def null_floors(
    pool: NDArray[np.floating], /, *, n: int, seed: int, n_projections: int
) -> Floors:
    """Both metrics on two disjoint halves of one sample.

    The residuals are what decide whether joint structure is being left wrong,
    and a residual only means something against the floor its own metric reaches
    when there is nothing left to fix. The two floors differ -- they measure
    different directions -- so quoting one residual against the other's floor is
    exactly the error to avoid.

    Both draws are `n`, the size of the comparison being calibrated, taken from
    a larger pool. Splitting an n-event sample in half would measure the floor
    for n/2 instead: W1 between two empirical measures of one distribution falls
    as n^-1/2, so that floor is ~1.41x too large and every residual quoted
    against it comes out that much too small.
    """
    pool = np.asarray(a=pool, dtype=np.double)
    if pool.shape[0] < 2 * n:
        raise ValueError(
            f"need two disjoint draws of {n} to calibrate a comparison of that "
            f"size, but the reference pool has {pool.shape[0]} rows"
        )
    order: NDArray[np.long] = np.random.default_rng(seed).permutation(x=pool.shape[0])
    left, right = pool[order[:n]], pool[order[n : 2 * n]]
    return Floors(
        sliced=sliced_wasserstein(left, right, seed=seed, n_projections=n_projections),
        axis=axis_wasserstein(left, right),
    )


def _load_splits(config: dict[str, Any], /) -> tuple[DatasetSplits, tuple[str, ...]]:
    """Rebuild exactly the dataset the run trained on."""
    if config["dataset"] == "jets":
        variables: tuple[str, ...] = tuple(config["variables"])
        splits, _dim, _std = load_jet_dataset(
            n_samples=config["n_samples"],
            batch_size=config["batch_size"],
            variables=variables,
            seed=config.get("data_seed", 42),
        )
        return splits, variables
    splits = DeconvolveDataset(
        # Runs predating seed recording used the then-hardcoded 42; see
        # CLAUDE.md's Seeding section. `averaging.py` still indexes this
        # directly and raises KeyError on those runs.
        batch_size=config["batch_size"],
        seed=config.get("data_seed", 42),
    ).generate_gaussian_dataset(
        params=gaussian_config_from_run_config(
            params=config["gaussian_params"], dim=config["dim"]
        ),
        n_samples=config["n_samples"],
    )
    return splits, tuple(f"dim_{i}" for i in range(config["dim"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ = parser.add_argument("--run-dir", type=Path, required=True)
    _ = parser.add_argument("--n-projections", type=int, default=128)
    _ = parser.add_argument("--repeats", type=int, default=4)
    _ = parser.add_argument(
        "--subsample",
        type=int,
        default=100_000,
        help="cap on events per side; the merged sort is the cost",
    )
    _ = parser.add_argument("--seed", type=int, default=0)
    args: argparse.Namespace = parser.parse_args()
    configure_logging(level="info")

    run_dir: Path = args.run_dir
    config: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    splits, variables = _load_splits(config)
    pop: Populations = splits.select(Split.TEST).partition()
    generator: keras.Model = keras.saving.load_model(
        artifacts_dir(run_dir) / "generator.keras"
    )

    rng: np.random.Generator = np.random.default_rng(args.seed)
    n = min(args.subsample, pop.mc.z.shape[0], pop.data.shape[0])
    i_mc: NDArray[np.long] = rng.permutation(x=pop.mc.z.shape[0])[:n]
    i_data: NDArray[np.long] = rng.permutation(x=pop.data.shape[0])[:n]
    weights: NDArray[np.double] = np.asarray(
        a=generator(pop.mc.z[i_mc], training=False), dtype=np.double
    ).ravel()

    logger.info(
        "%s: %d events per side, %d observables, %d projections x %d draws",
        run_dir.name,
        n,
        len(variables),
        args.n_projections,
        args.repeats,
    )

    levels: dict[str, tuple[Any, Any]] = {
        "detector": (pop.data[i_data], pop.mc.x[i_mc]),
    }
    # The floor is drawn from the *full* reference population so both null
    # draws can be `n` -- the size of the comparison it calibrates.
    floor_pools: dict[str, NDArray[Any]] = {"detector": pop.data}
    if pop.has_truth:
        levels["particle"] = (pop.truth[i_data], pop.mc.z[i_mc])
        floor_pools["particle"] = pop.truth
    else:
        logger.warning(msg="no truth recorded; particle level skipped")
    if pop.data.shape[0] < 2 * n:
        raise SystemExit(
            f"--subsample {n} needs {2 * n} reference events to calibrate a "
            f"floor at that size; only {pop.data.shape[0]} are available"
        )

    table = Table(title=f"{run_dir.name} — residuals against each metric's floor")
    table.add_column(header="Level")
    table.add_column(header="Metric")
    table.add_column(header="before", justify="right")
    table.add_column(header="after", justify="right")
    table.add_column(header="improvement", justify="right")
    # The residual, not the improvement, is what says whether joint structure
    # was left wrong -- and only in units of the floor its own metric reaches
    # when there is nothing left to fix. Improvement percentages have different
    # denominators for the two metrics and cannot be compared directly.
    table.add_column(header="after / floor", justify="right")
    for level, (ref, comp) in levels.items():
        floors: Floors = null_floors(
            floor_pools[level], n=n, seed=args.seed, n_projections=args.n_projections
        )
        c: Comparison = compare(
            ref, comp, weights, n_projections=args.n_projections, repeats=args.repeats
        )
        axis_before: np.double = axis_wasserstein(ref, comp)
        axis_after: np.double = axis_wasserstein(ref, comp, comp_weights=weights)
        table.add_row(
            level,
            "sliced",
            f"{c.before:.5f}",
            f"{c.after:.5f}",
            f"{c.improvement_pct:+.2f}%",
            f"{c.after / floors.sliced:.2f}x",
        )
        table.add_row(
            "",
            "per-axis",
            f"{axis_before:.5f}",
            f"{axis_after:.5f}",
            f"{100.0 * (axis_before - axis_after) / axis_before:+.2f}%",
            f"{axis_after / floors.axis:.2f}x",
        )
        logger.info(
            "%s floors (same distribution both sides): sliced %.5f, per-axis %.5f",
            level,
            floors.sliced,
            floors.axis,
        )
    Console().print(table)
    logger.info(
        msg="Compare the two 'after / floor' figures, not the two improvements. "
        "Similar multiples mean the residual is isotropic and the axis metrics "
        "were not flattering; a larger sliced multiple means the reweighting "
        "fixed the marginals and left joint structure behind."
    )


if __name__ == "__main__":
    main()
