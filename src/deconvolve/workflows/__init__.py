"""The orchestration behind `deconvolve train` and `deconvolve leakage-check`.

These live apart from `training` and `evaluation` because they need both.
Dependencies point one way: `workflows` imports `training`, `evaluation` and
`baselines`, and none of those import `workflows`. Putting either module inside
`training/` or `evaluation/` instead would make one of those packages import
the other's sibling, closing a cycle.

`run` and `run_leakage_check` are re-exported here, so
`from deconvolve.workflows import run` works without knowing which submodule
it lives in.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import leakage, train
from .leakage import run_leakage_check
from .train import run

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "leakage",
    "run",
    "run_leakage_check",
    "train",
)
