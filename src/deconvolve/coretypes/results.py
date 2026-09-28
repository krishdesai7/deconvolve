from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Literal

    import numpy as np
    from numpy.typing import NDArray

    from .events import Populations
    from .types import MetricRecord


class UnfoldingPopulations(NamedTuple):
    """What a baseline may fit on, and what it is scored on --- kept apart.

    `fit` is train+val; `test` is the held-out split. They are disjoint, and
    deliberately so: it is conventional in the unfolding literature to fit the
    response and iterate the prior on every event and then quote metrics on a
    subset of those same events, which scores an estimator on data it has
    already seen. Naming the two populations separately makes that mistake
    harder to reach for by accident.
    """

    fit: Populations
    test: Populations


@dataclass(frozen=True)
class VariableOutcome:
    variable_name: str
    status: Literal["completed", "skipped"]
    n_bins: int
    _: KW_ONLY
    skip_reason: str | None = None


@dataclass(frozen=True)
class IBUResult:
    metrics: dict[str, MetricRecord]
    variable_names: tuple[str, ...]
    weights: NDArray[np.single]
    outcomes: tuple[VariableOutcome, ...]
