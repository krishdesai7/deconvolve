from __future__ import annotations

from typing import TYPE_CHECKING

from . import design, report, variance
from .design import (
    Design,
    DesignSpec,
    EvaluationSet,
    base_populations,
    bootstrap,
    bootstrap_multiplicities,
    cell_path,
    freeze_design,
    load_cells,
    load_frozen,
    replicate_splits,
    reserve_evaluation_set,
    run_cell,
    split_seed,
)
from .report import collect, multinomial_off_diagonal
from .variance import (
    Covariances,
    VarianceComponents,
    binned_spectra,
    component_covariances,
    correlation,
    decompose,
    evaluation_covariance,
    evaluation_variance,
    quantile_edges,
    weighted_means,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "Covariances",
    "Design",
    "DesignSpec",
    "EvaluationSet",
    "VarianceComponents",
    "base_populations",
    "binned_spectra",
    "bootstrap",
    "bootstrap_multiplicities",
    "cell_path",
    "collect",
    "component_covariances",
    "correlation",
    "decompose",
    "design",
    "evaluation_covariance",
    "evaluation_variance",
    "freeze_design",
    "load_cells",
    "load_frozen",
    "multinomial_off_diagonal",
    "quantile_edges",
    "replicate_splits",
    "report",
    "reserve_evaluation_set",
    "run_cell",
    "split_seed",
    "variance",
    "weighted_means",
)
