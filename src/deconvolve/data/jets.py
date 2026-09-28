"""Load jet substructure data for RAN training.

Checks `CACHE_DIR` (`.cache/`, or wherever `DECONVOLVE_CACHE_DIR` points) for
per-variable `.npz` files. If missing, invokes `download_jet_data` to fetch
from Zenodo. Loads, subsamples, z-score standardizes (using MC gen-level
statistics only), and builds the train/val/test splits via
`DeconvolveDataset`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import numpy as np

from ..coretypes import (
    CACHE_DIR,
    CACHE_FILENAMES,
    EVENT_DTYPE,
    SUBSTRUCTURE_VARIABLES,
    ZXY,
    DatasetSplits,
    Events,
    Populations,
)
from ..instrumentation import note
from .datasets import DeconvolveDataset
from .download import download_jet_data

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from ..coretypes import EventArray

logger: Logger = logging.getLogger(name=__name__)


def _reject_unordered(variables: Sequence[str], /) -> None:
    """Refuse the containers that silently lose the column order.

    A `set` costs nothing to iterate and everything to reproduce; a duplicate
    name would quietly give two columns the same data under one label; an
    unknown name would only surface as a `KeyError` deep in the load. All three
    are cheap to catch here and expensive to debug from a metrics table.
    """
    if isinstance(variables, (set, frozenset)):
        raise TypeError(
            "variables must be an ordered sequence, not a set: column order is "
            "recorded in config.json and has to survive into another process"
        )
    if len(set(variables)) != len(variables):
        raise ValueError(f"variables contains duplicates: {list(variables)}")
    unknown: list[str] = [v for v in variables if v not in CACHE_FILENAMES]
    if unknown:
        raise ValueError(
            f"unknown jet variables {unknown}; expected from {list(CACHE_FILENAMES)}"
        )
    if not variables:
        raise ValueError("variables must name at least one observable")


def load_jet_dataset(
    n_samples: int = 500_000,
    batch_size: int = 1024,
    cache_dir: Path = CACHE_DIR,
    variables: Sequence[str] = SUBSTRUCTURE_VARIABLES,
    seed: int = 42,
) -> tuple[DatasetSplits, int, dict[str, tuple[np.single, np.single]]]:
    """Build jet splits with column `i` taken from `variables[i]`.

    `variables` is a `Sequence` and the order is load-bearing: it is the column
    order of every array downstream, `_save_run` records it, and a later
    `deconvolve evaluate` or `deconvolve baseline ibu` must reproduce it exactly
    to label those columns --- or to feed a trained generator its own features.
    Passing a `set` or `frozenset` here is a bug, not a convenience: its
    iteration order depends on per-process randomized string hashes and so
    cannot survive into the second process. A set is refused outright, and
    duplicate and unknown names are refused too.

    Each selected substructure variable is z-score standardized using the MC
    gen-level (`z_gen`) mean and std. The same parameters are applied to all
    four arrays (`z_true`, `x_data`, `z_gen`, `x_sim`) to avoid information
    leakage and preserve correlations.

    Args:
        n_samples: Number of events to use per class (data and MC).
        batch_size: Batch size for the returned splits.
        cache_dir: Directory containing per-variable `.npz` files.
        variables: Which substructure variables to use, **in column order**.
        seed: Dataset seed, controlling the shuffle, the train/val/test split
            and the per-epoch batch order. Independent of the weight-init
            seed passed to `train`.

    Returns:
        The splits, the feature dimensionality, and the standardization
            parameters `{var_name: (mu, sigma)}`.
    """
    _reject_unordered(variables)
    # The npz caches on disk are float64, as the Zenodo release ships them.
    # Narrowing happens once here, on the way into the pipeline, so the
    # standardization statistics below are float32 reductions (pairwise
    # summation keeps them accurate).
    scalar: np.dtype[np.single] = np.dtype(EVENT_DTYPE)

    missing: list[str] = [
        v for v in variables if not (cache_dir / f"{CACHE_FILENAMES[v]}.npz").exists()
    ]
    if missing:
        note("downloaded from Zenodo", to="data")
        logger.info(msg="Jet data not found in cache; downloading from Zenodo.")
        download_jet_data(cache_dir)
    else:
        note("cache hit", to="data")

    n_features: int = len(variables)

    with np.load(file=cache_dir / f"{CACHE_FILENAMES[variables[0]]}.npz") as f:
        n_avail: int = min(
            len(cast(typ="NDArray[Any]", val=f["z_true"])),
            len(cast(typ="NDArray[Any]", val=f["z_gen"])),
        )
    if n_samples > n_avail:
        raise ValueError(f"Requested {n_samples} samples but only {n_avail} available")

    z_true: EventArray = np.empty(shape=(n_samples, n_features), dtype=scalar)
    x_data: EventArray = np.empty(shape=(n_samples, n_features), dtype=scalar)
    z_gen: EventArray = np.empty(shape=(n_samples, n_features), dtype=scalar)
    x_sim: EventArray = np.empty(shape=(n_samples, n_features), dtype=scalar)

    std_params: dict[str, tuple[np.single, np.single]] = {}
    for i, var in enumerate(iterable=variables):
        with np.load(file=cache_dir / f"{CACHE_FILENAMES[var]}.npz") as f:
            col: Mapping[str, NDArray[Any]] = cast(
                typ="Mapping[str, NDArray[Any]]", val=f
            )
            z_true[:, i] = col["z_true"][:n_samples]
            x_data[:, i] = col["x_data"][:n_samples]
            z_gen[:, i] = col["z_gen"][:n_samples]
            x_sim[:, i] = col["x_sim"][:n_samples]

        # Standardize using MC gen-level statistics only
        mu: np.single = z_gen[:, i].mean()
        sigma: np.single = z_gen[:, i].std()
        std_params[var] = (mu, sigma)

        z_true[:, i] = (z_true[:, i] - mu) / sigma
        x_data[:, i] = (x_data[:, i] - mu) / sigma
        z_gen[:, i] = (z_gen[:, i] - mu) / sigma
        x_sim[:, i] = (x_sim[:, i] - mu) / sigma

    data: ZXY = Populations(
        mc=Events(z_gen, x_sim), data=x_data, truth=z_true
    ).interleave()

    splits: DatasetSplits = DeconvolveDataset(batch_size, seed).splits_from_data(data)
    return splits, n_features, std_params
