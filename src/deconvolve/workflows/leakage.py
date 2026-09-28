"""Quick leakage check: poison z_true and verify training is unaffected.

Sets z_true to a silly value in one arm and compares against a clean arm; if
any network can see z_true, the two diverge.

Both arms must use the same `init_seed` or the comparison is meaningless: with
random initialization the run-to-run spread swamps the effect being tested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..coretypes import (
    EVENT_DTYPE,
    METRIC_DISPLAY_SCALE,
    POISON_SENTINEL,
    TRUTH_SENTINEL,
    Events,
    Populations,
)
from ..data import DeconvolveDataset
from ..evaluation.evaluate import (
    _collect_test_data,
    _improvement,
    _triangular_per_dim,
    _wd_per_dim,
)
from ..training.engine import train

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any, Literal

    from ..coretypes import ZXY, DatasetSplits, DeconvolveModel, EventArray

logger: Logger = logging.getLogger(name=__name__)


def run_leakage_check(poison: bool, sentinel: float, seed: int, init_seed: int) -> None:
    _: Any
    if poison and sentinel == TRUTH_SENTINEL:
        # Caught here rather than 100k events and a full training run later,
        # where it surfaces as `require_truth()` refusing for a reason that has
        # nothing obvious to do with the flag that caused it.
        raise ValueError(
            f"--sentinel {sentinel} is TRUTH_SENTINEL, which marks a sample as "
            "having no particle-level truth at all -- so the particle-level "
            "comparison this check exists to make would be refused. Pick any "
            f"other far-off-manifold value; the default is {POISON_SENTINEL}."
        )
    tag: Literal["CLEAN", "POISONED"] = "POISONED" if poison else "CLEAN"
    logger.info(
        "Running %s (z_true = %s), init seed %d",
        tag,
        f"Poisoned: {sentinel}" if poison else "N(0,1)",
        init_seed,
    )

    # Generate: z_true ~ N(0,1), z_gen ~ N(-0.5, 1), sigma_det = 0.25
    rng: np.random.Generator = np.random.default_rng(seed)
    n: int = 100_000

    # `Generator.normal` is float64 whatever the caller wants, so the check's
    # own sample narrows here, at the one boundary where it enters the pipeline.
    z_true: EventArray = rng.normal(loc=0.0, scale=1.0, size=(n, 1)).astype(EVENT_DTYPE)
    z_gen: EventArray = rng.normal(loc=-0.5, scale=1.0, size=(n, 1)).astype(EVENT_DTYPE)
    x_data: EventArray = (z_true + rng.normal(loc=0, scale=0.25, size=(n, 1))).astype(
        EVENT_DTYPE
    )
    x_sim: EventArray = (z_gen + rng.normal(loc=0, scale=0.25, size=(n, 1))).astype(
        EVENT_DTYPE
    )

    if poison:
        z_true[:] = sentinel

    data: ZXY = Populations(
        mc=Events(z_gen, x_sim), data=x_data, truth=z_true
    ).interleave()

    splits: DatasetSplits = DeconvolveDataset(
        batch_size=1024, seed=seed
    ).splits_from_data(data)

    # Fixed init_seed: both arms must start from identical weights, or the
    # comparison measures initialization variance rather than leakage.
    g: DeconvolveModel = train(
        splits, dim=1, hidden_units=32, n_layers=2, seed=init_seed
    ).g

    test: Populations = _collect_test_data(test_ds=splits.test).partition()

    raw_w: EventArray = np.asarray(a=g(test.mc.z)).flatten()
    w: EventArray = raw_w / raw_w.mean()

    for level, ref, comp in [
        ("DETECTOR", test.data, test.mc.x),
        ("PARTICLE", test.require_truth(), test.mc.z),
    ]:
        wd_b: float = _wd_per_dim(ref, comp)[0]
        wd_a: float = _wd_per_dim(ref, comp, weights=w)[0]
        td_b: float = _triangular_per_dim(ref, comp)[0]
        td_a: float = _triangular_per_dim(ref, comp, weights=w)[0]
        logger.info(
            "%10s (x%g)  Wasserstein: %.2f → %.2f (%+.1f%%)  Δ: %.2f → %.2f (%+.1f%%)",
            level,
            METRIC_DISPLAY_SCALE,
            wd_b * METRIC_DISPLAY_SCALE,
            wd_a * METRIC_DISPLAY_SCALE,
            _improvement(before=wd_b, after=wd_a),
            td_b * METRIC_DISPLAY_SCALE,
            td_a * METRIC_DISPLAY_SCALE,
            _improvement(before=td_b, after=td_a),
        )
