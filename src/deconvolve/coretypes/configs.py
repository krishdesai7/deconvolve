"""Validated views of a run's Gaussian parameter set and `config.json`.

Together they configure a run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from typing import Any, Final, LiteralString

    import numpy as np
    from numpy.typing import NDArray

    from .enums import DatasetName
REQUIRED_KEYS: Final[frozenset[LiteralString]] = frozenset(
    (
        "mu_gen",
        "mu_true",
        "sigma_gen",
        "sigma_true",
        "sigma_detector",
    )
)


class GaussianConfig(NamedTuple):
    dim: int
    mu_gen: NDArray[np.double]
    mu_true: NDArray[np.double]
    cov_gen: NDArray[np.double]
    cov_true: NDArray[np.double]
    cov_detector: NDArray[np.double]

    def model_dump(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "mu_gen": self.mu_gen.tolist(),
            "mu_true": self.mu_true.tolist(),
            "cov_gen": self.cov_gen.tolist(),
            "cov_true": self.cov_true.tolist(),
            "cov_detector": self.cov_detector.tolist(),
        }


@dataclass(frozen=True)
class RunConfig:
    """A validated view of a run's `config.json`.

    Attributes:
        source: The raw `config.json`, kept because `_load_splits` reconstructs
            the dataset from it and must see exactly what the run recorded.
        variable_names: The names of the observables, one per dimension.
    """

    source: dict[str, Any]
    dataset: DatasetName
    dim: int
    n_samples: int
    batch_size: int
    data_seed: int
    variable_names: tuple[str, ...]
