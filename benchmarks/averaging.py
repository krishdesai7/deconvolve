"""Average the epochs the criterion cannot tell apart, instead of picking one.

Two measured facts motivate this, both from saved runs rather than theory.

**The criterion cannot rank most epochs.** `benchmarks/ceiling.py` diagnostic C
shows detector level is nearly saturated after reweighting -- a converged
classifier finds 0.6% of the mismatch it finds unweighted -- and roughly 55 of
100 epochs sit within one estimator floor of the detector-MMD minimum. Their
particle-level MMD spans almost the whole range of the run, and the Spearman
correlation between the two criteria is +0.18.

**The resulting spread is larger than any effect worth measuring.** Nine
six-variable jet runs differing only in initialization give particle-level jet
mass between 10.0% and 34.1% improvement, an SD of 9.1 points on a mean of
23.9. Resolving a 7-point difference against that needs ~25 runs per arm.

Picking one member of a set the criterion declares equivalent is a coin flip,
and the spread above is largely the cost of flipping it. The principled
response to candidates that cannot be separated is not to choose -- it is to
average. Each epoch defines a reweighted distribution `q_e`; the mixture
`(1/K) sum_e q_e` is a distribution too, and in weight space it is the mean of
the per-epoch normalized weights. That is what this measures.

Averaging **weights**, not parameters. Two networks' parameters have no
meaningful midpoint -- the loss surface is not convex and the hidden units of
one have no correspondence with the other's -- but the weights they induce live
in the space the objective is actually defined on, where a mixture is exactly a
mixture.

The tied set is chosen on the **validation** detector MMD, which is what
selection already uses and needs no truth, and everything is then scored on
the **test** split. `particle-best` is reported alongside as the achievable
reference; it reads `z_true` to pick its epoch and is a ceiling, not a method.

```zsh
uv run benchmarks/averaging.py --run-dir runs/2026-08-29T213609Z
uv run benchmarks/averaging.py --run-dir runs/2026-08-29T213609Z --floor 1e-3
```
"""

# pyrefly: ignore-errors[unknown-argument-type]
# -- run config is JSON (dict[str, Any]); numpy stubs return Any for elementwise ops
from __future__ import annotations

import os

os.environ.setdefault(key="KERAS_BACKEND", value="jax")  # pyrefly: ignore[unused-call-result]

import json
import logging
from itertools import starmap
from typing import TYPE_CHECKING, Annotated, NamedTuple

import deconvolve  # ruff: ignore[unused-import]  -- pins JAX_ENABLE_X64
import jax.numpy as jnp
import numpy as np
import typer
from deconvolve.coretypes import Split, artifacts_dir
from deconvolve.data import DeconvolveDataset, load_jet_dataset
from deconvolve.data.config import gaussian_config_from_run_config
from deconvolve.evaluation.evaluate import _improvement, _wd_per_dim
from deconvolve.instrumentation.logging_config import configure_logging
from deconvolve.training.engine import MMD_SUBSAMPLE, _weights_per_epoch, load_params
from deconvolve.training.mmd import (
    MMDCache,
    bandwidths,
    build_cache,
    subsample_indices,
    weighted_mmd,
)
from deconvolve.training.models import build_generator

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from deconvolve.coretypes import (
        DatasetSplits,
        DeconvolveModel,
        EventArray,
        Populations,
    )
    from numpy.typing import NDArray


logger: Logger = logging.getLogger(name="deconvolve.averaging")

# The MMD^2 estimator's resolution at m = 16384, from `benchmarks/`. Two epochs
# closer than this in detector MMD are not distinguishable by the criterion,
# which is the definition of the tied set.
DEFAULT_FLOOR: float = 5e-4


class Strategy(NamedTuple):
    name: str
    weights: EventArray
    n_epochs: int
    note: str


def _normalized(raw: NDArray[np.double], /) -> NDArray[np.double]:
    """Scale each epoch's weights to mean 1, so no epoch dominates the mixture.

    A mixture of reweighted distributions is the mean of their *normalized*
    weights. Averaging raw generator output instead would weight each epoch by
    whatever overall scale its softplus happened to settle at, which is not a
    property of the distribution it represents.
    """
    return raw / raw.mean(axis=-1, keepdims=True)


def _strategies(
    per_epoch: NDArray[np.double],
    val_mmd: NDArray[np.double],
    val_mmd_particle: NDArray[np.double] | None,
    best_epoch: int,
    floor: float,
    /,
) -> list[Strategy]:
    """The single-epoch pick, the two averages, and the truth-fitted ceiling."""
    normalized: NDArray[np.double] = _normalized(per_epoch)
    tied: NDArray[np.intp] = np.flatnonzero(a=val_mmd <= val_mmd.min() + floor)
    out: list[Strategy] = [
        Strategy(
            name="selected",
            weights=normalized[best_epoch],
            n_epochs=1,
            note=f"epoch {best_epoch}, what the run shipped",
        ),
        Strategy(
            name="tied-mean",
            weights=normalized[tied].mean(axis=0, dtype=np.single),
            n_epochs=len(tied),
            note=f"epochs within {floor:.0e} of the detector minimum",
        ),
        Strategy(
            name="all-mean",
            weights=normalized.mean(axis=0, dtype=np.single),
            n_epochs=len(normalized),
            note="every epoch",
        ),
    ]
    if val_mmd_particle is not None:
        best_p = int(val_mmd_particle.argmin())
        out.append(
            Strategy(
                name="particle-best",
                weights=normalized[best_p],
                n_epochs=1,
                note=f"epoch {best_p}, chosen with truth -- a ceiling, not a method",
            )
        )
    return out


def _mmd_scorer(test_pop: Populations, seed: int, /) -> Callable[[EventArray], float]:
    """The selection criterion itself, ready to score any weight vector.

    The cache holds every weight-independent term, so it is built once and
    reused across strategies -- which is the entire reason `MMDCache` exists.
    Rebuilding it per strategy costs an m x m kernel each time for a value that
    cannot have changed.

    Reported per strategy because an average is only interesting if it is still
    a solution the criterion accepts: a mixture that has left the tied set
    traded detector-level agreement for whatever else it gained, which is a
    different claim from the one under test.
    """
    n_nat, n_mc = len(test_pop.data), len(test_pop.mc)
    m: int = min(MMD_SUBSAMPLE, n_nat, n_mc)
    i_nat: NDArray[np.intp] = subsample_indices(seed, n_nat, m)
    i_mc: NDArray[np.intp] = subsample_indices(seed + 1, n_mc, m)
    ref, comp = test_pop.data[i_nat], test_pop.mc.x[i_mc]
    cache: MMDCache = build_cache(ref, comp, sigmas=bandwidths(jnp.asarray(a=ref)))

    def score(w: EventArray, /) -> float:
        return float(weighted_mmd(cache, jnp.asarray(a=w[i_mc]))[0])

    return score


def _report(
    strategy: Strategy,
    test_pop: Populations,
    variables: tuple[str, ...],
    mmd: Callable[[EventArray], float],
    /,
) -> tuple[float, float]:
    """Per-variable Wasserstein improvement at both levels."""
    w: EventArray = strategy.weights
    ess: float = float(w.sum() ** 2 / np.square(w).sum())
    logger.info(
        "  %-14s %2d epoch(s)  ESS %5.1f%%  detector MMD2 %+.3e   %s",
        strategy.name,
        strategy.n_epochs,
        100.0 * ess / len(w),
        mmd(w),
        strategy.note,
    )
    means: list[float] = []
    for level, ref, comp in (
        ("particle", test_pop.require_truth(), test_pop.mc.z),
        ("detector", test_pop.data, test_pop.mc.x),
    ):
        before: NDArray[np.double] = _wd_per_dim(ref=ref, comp=comp)
        after: NDArray[np.double] = _wd_per_dim(ref=ref, comp=comp, weights=w)
        per_var: list[float] = list(
            starmap(_improvement, zip(before, after, strict=True))
        )
        means.append(float(np.mean(a=per_var)))
        detail: str = "  ".join(
            f"{v}={p:+.1f}%" for v, p in zip(variables, per_var, strict=True)
        )
        logger.info("      %-9s mean %+6.1f%%   %s", level, means[-1], detail)
    return means[0], means[1]


def _load_splits(config: dict[str, Any], /) -> tuple[DatasetSplits, tuple[str, ...]]:
    """Rebuild exactly the dataset the run trained on."""
    if config["dataset"] == "jets":
        variables: tuple[str, ...] = tuple(config["variables"])
        splits, _dim, _std = load_jet_dataset(
            n_samples=config["n_samples"],
            batch_size=config["batch_size"],
            variables=variables,
            seed=config["data_seed"],
        )
        return splits, variables
    # `_save_run` records the *parsed* config, not the path it came from, so
    # the run reproduces even if the YAML has since moved or changed. Rebuilt
    # through the same helper `deconvolve evaluate` uses: `model_dump` turns the
    # covariance arrays into lists, and the constructor does not coerce them
    # back, so `GaussianConfig(**dumped)` yields a config whose fields are
    # lists and fails on the first `.tolist()`.
    splits: DatasetSplits = DeconvolveDataset(
        batch_size=config["batch_size"], seed=config["data_seed"]
    ).generate_gaussian_dataset(
        params=gaussian_config_from_run_config(
            config["gaussian_params"], config["dim"]
        ),
        n_samples=config["n_samples"],
    )
    return splits, tuple(f"z{i}" for i in range(config["dim"]))


def main(
    path: Annotated[Path, typer.Argument(help="The run directory")],
    floor: Annotated[
        float,
        typer.Option("--floor", "-f", help="detector-MMD width defining the tied set"),
    ] = DEFAULT_FLOOR,
) -> None:
    configure_logging(level="info")

    config: dict[str, Any] = json.loads((path / "config.json").read_text())
    history: np.lib.npyio.NpzFile = np.load(file=artifacts_dir(path) / "history.npz")
    splits, variables = _load_splits(config)
    test_pop: Populations = splits.select(Split.TEST).partition()

    # The architecture only; `load_params` supplies every epoch's values.
    generator: DeconvolveModel = build_generator(
        dim=config["dim"],
        hidden_units=config["hidden_units"],
        n_layers=config["n_layers"],
    )
    per_epoch: NDArray[np.double] = np.asarray(
        a=_weights_per_epoch(generator, load_params(path), test_pop.mc.z),
        dtype=np.double,
    )
    logger.info(
        "%s: %d epochs x %d test events, %d variables",
        path.name,
        *per_epoch.shape,
        len(variables),
    )

    val_mmd: NDArray[np.double] = np.asarray(history["val_mmd"], dtype=np.double)
    particle: NDArray[np.double] | None = (
        np.asarray(a=history["val_mmd_particle"], dtype=np.double)
        if "val_mmd_particle" in history
        else None
    )
    logger.info(msg="")
    logger.info(msg="Scored on the test split, weights averaged over the tied set")
    mmd: Callable[[EventArray], float] = _mmd_scorer(test_pop, config["data_seed"])
    results: dict[str, tuple[float, float]] = {}
    for strategy in _strategies(
        per_epoch, val_mmd, particle, int(config["best_epoch"]), floor
    ):
        results[strategy.name] = _report(strategy, test_pop, variables, mmd)

    logger.info(msg="")
    logger.info(msg="SUMMARY  particle-level mean improvement")
    base: float = results["selected"][0]
    for name, (p, _d) in results.items():
        logger.info("  %-14s %+6.1f%%   (%+.1f vs selected)", name, p, p - base)
    logger.info(
        msg="  Averaging is truth-free: the tied set comes from validation detector"
    )
    logger.info(
        msg="  MMD. `particle-best` reads z_true to pick its epoch and bounds the rest."
    )


if __name__ == "__main__":
    typer.run(function=main)
