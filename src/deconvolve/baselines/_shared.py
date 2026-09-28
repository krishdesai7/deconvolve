"""The part of a baseline that is not the unfolding method.

Reading a run's config, rebuilding its populations, and scoring the resulting
weights with the same metrics RAN is scored by.

A baseline attempts the same task RAN does -- generate weights that reweight
Generation, using only the relationship between Data and Simulation -- so it
needs the same run config, the same event populations, and the same metric
record. Keeping those here means a comparison is a comparison of unfolding
methods and nothing else; both IBU and OmniFold are callers into this shared
scoring path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ..coretypes import DatasetName, RunConfig, Split, UnfoldingPopulations
from ..evaluation.evaluate import (
    _improvement,
    _js_per_dim,
    _load_splits,
    _triangular_per_dim,
    _wd_per_dim,
)

if TYPE_CHECKING:
    from typing import Any

    from ..coretypes import ZXY, DatasetSplits, EventArray, MetricRecord, Populations


def _positive_int(value: object, key: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _parse_dataset(raw: dict[str, Any]) -> DatasetName:
    value: object = raw.get("dataset", DatasetName.gaussian.value)
    try:
        return DatasetName(value=value)
    except ValueError as error:
        known: str = ", ".join(name.value for name in DatasetName)
        raise ValueError(
            f"Unknown dataset {value!r}, expected one of {known}"
        ) from error


def _parse_variable_names(
    raw: dict[str, Any], dataset: DatasetName, dim: int, /
) -> tuple[str, ...]:
    if dataset == DatasetName.gaussian:
        return tuple(f"dim_{i}" for i in range(dim))

    variables: object = raw.get("variables")
    if not isinstance(variables, (list, tuple)) or any(
        not isinstance(name, str) or not name for name in variables
    ):
        raise ValueError("variables must be a sequence of nonempty strings")
    variable_names: tuple[str, ...] = cast(typ=tuple[str, ...], val=tuple(variables))
    if len(variable_names) != dim:
        raise ValueError(
            f"variables has length {len(variable_names)}, expected dim={dim}"
        )
    return variable_names


def parse_run_config(raw: object) -> RunConfig:
    """Validate a run's `config.json`, already parsed from JSON, into a `RunConfig`."""
    if not isinstance(raw, dict) or not all(isinstance(k, str) for k in raw):
        raise ValueError("run config must be a JSON object")

    config: dict[str, Any] = cast(typ="dict[str, Any]", val=raw)
    dim: int = _positive_int(value=config.get("dim"), key="dim")
    n_samples: int = _positive_int(value=config.get("n_samples"), key="n_samples")
    batch_size: int = _positive_int(value=config.get("batch_size"), key="batch_size")
    data_seed: object = config.get("data_seed", 42)
    if not isinstance(data_seed, int):
        raise TypeError("data_seed must be an integer")

    dataset: DatasetName = _parse_dataset(raw=config)
    variable_names: tuple[str, ...] = _parse_variable_names(config, dataset, dim)

    return RunConfig(
        source=dict(config),
        dataset=dataset,
        dim=dim,
        n_samples=n_samples,
        batch_size=batch_size,
        data_seed=data_seed,
        variable_names=variable_names,
    )


def _partitioned(data: ZXY, expected_dim: int, label: str, /) -> Populations:
    """Check the shape assumptions a baseline relies on, then partition."""
    if data.z.ndim != 2 or data.x.ndim != 2 or data.z.shape != data.x.shape:
        raise ValueError(
            f"{label}: z and x must be identically shaped two-dimensional arrays"
        )
    if data.z.shape[1] != expected_dim:
        raise ValueError(
            f"{label}: array dimension {data.z.shape[1]}, expected dim={expected_dim}"
        )
    if not np.all(a=np.isfinite(data.z)) or not np.all(a=np.isfinite(data.x)):
        raise ValueError(f"{label}: z and x values must be finite")
    try:
        return data.partition()
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error


def prepare_populations(
    splits: DatasetSplits, expected_dim: int
) -> UnfoldingPopulations:
    """Partition a dataset into the populations a baseline fits and is scored on.

    Returns an `UnfoldingPopulations`, which unpacks as `(fit, test)`. Both are
    `Populations`, and they are disjoint. `fit` is train+val and supplies the
    response (`fit.mc.z` and `fit.mc.x`, paired per event) and the measurement
    (`fit.data`). `test` is the held-out split alone, where the metrics are
    computed: detector level scores `test.data` against `test.mc.x`, particle
    level scores `test.truth` against `test.mc.z`. `test.truth` is the only
    place a baseline touches the answer key, and it appears only in scoring.

    By construction, `fit` is `Split.TRAIN | Split.VAL`, not `Split.ALL`. A
    baseline fitted on every event and then scored on the test split would be
    scored on data it had already used and would be handed information RAN is
    denied: `train` does read the test split, to compute a test-level MMD
    diagnostic, but nothing weight-bearing depends on that read, so the test
    split still cannot influence the returned model or its selection
    (`tests/test_train.py::TestTrainingNeverSeesTheTestSplit`). The comparison
    is only a comparison if both sides see the same events.

    Arrays arrive at the pipeline's pinned `EVENT_DTYPE` and are not cast here.
    """
    return UnfoldingPopulations(
        fit=_partitioned(
            splits.select(Split.TRAIN | Split.VAL), expected_dim, "train and val splits"
        ),
        test=_partitioned(splits.select(Split.TEST), expected_dim, "test split"),
    )


def load_populations(config: RunConfig) -> UnfoldingPopulations:
    """Rebuild the run's dataset and split it into the baseline populations.

    The populations come at the pipeline's pinned `EVENT_DTYPE`. Baselines
    that need another precision call `astype` at their own boundary.
    """
    return prepare_populations(
        _load_splits(config=config.source), expected_dim=config.dim
    )


def evaluate_dimension(
    reference: EventArray,
    comparison: EventArray,
    weights: EventArray,
) -> MetricRecord:
    """Score one dimension before and after reweighting `comparison`."""
    wasserstein_before: float = _wd_per_dim(ref=reference, comp=comparison)[0]
    wasserstein_after: float = _wd_per_dim(
        ref=reference, comp=comparison, weights=weights
    )[0]
    jensenshannon_before: float = _js_per_dim(ref=reference, comp=comparison)[0]
    jensenshannon_after: float = _js_per_dim(
        ref=reference, comp=comparison, weights=weights
    )[0]
    triangular_before: float = _triangular_per_dim(ref=reference, comp=comparison)[0]
    triangular_after: float = _triangular_per_dim(
        ref=reference, comp=comparison, weights=weights
    )[0]
    return {
        "wasserstein_before": wasserstein_before,
        "wasserstein_after": wasserstein_after,
        "wasserstein_improvement_pct": _improvement(
            wasserstein_before, wasserstein_after
        ),
        "jensenshannon_before": jensenshannon_before,
        "jensenshannon_after": jensenshannon_after,
        "jensenshannon_improvement_pct": _improvement(
            jensenshannon_before, jensenshannon_after
        ),
        "triangular_before": triangular_before,
        "triangular_after": triangular_after,
        "triangular_improvement_pct": _improvement(triangular_before, triangular_after),
    }
