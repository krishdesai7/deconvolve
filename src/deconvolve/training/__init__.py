from __future__ import annotations

from typing import TYPE_CHECKING

from . import engine, mmd, models
from .engine import (
    EPS,
    MMD_SUBSAMPLE,
    PARAMS_FILE,
    EpochParams,
    RunCarry,
    TrainResult,
    TrainState,
    bce_sums,
    load_params,
    normalize_weights,
    save_params,
    train,
    weight_dispersion,
    weighted_bce,
)
from .mmd import (
    MMDCache,
    bandwidths,
    build_cache,
    median_bandwidth,
    mmd_curve,
    squared_distances,
    subsample_indices,
    weighted_mmd,
)
from .models import build_discriminator, build_generator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "EPS",
    "MMD_SUBSAMPLE",
    "PARAMS_FILE",
    "EpochParams",
    "MMDCache",
    "RunCarry",
    "TrainResult",
    "TrainState",
    "bandwidths",
    "bce_sums",
    "build_cache",
    "build_discriminator",
    "build_generator",
    "engine",
    "load_params",
    "median_bandwidth",
    "mmd",
    "mmd_curve",
    "models",
    "normalize_weights",
    "save_params",
    "squared_distances",
    "subsample_indices",
    "train",
    "weight_dispersion",
    "weighted_bce",
    "weighted_mmd",
)
