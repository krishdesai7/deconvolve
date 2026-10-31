"""CLI choice enums.

They live here rather than beside the code they select for so that a choice
type is not tied to the module that consumes it -- `DatasetName` names an
option `deconvolve.data` implements, and `LogLevel` one that
`deconvolve.instrumentation.logging_config` does.
"""

from __future__ import annotations

from enum import StrEnum, auto


class LogLevel(StrEnum):
    debug = auto()
    info = auto()
    warning = auto()
    error = auto()
    critical = auto()


class DatasetName(StrEnum):
    gaussian = auto()
    jets = auto()


class Objective(StrEnum):
    """Which divergence the adversarial game minimizes at detector level.

    `bce` is the weighted BCE against a sigmoid discriminator: Jensen-Shannon at
    the optimum. `wasserstein` is the Kantorovich-Rubinstein dual against an
    unbounded, gradient-penalized critic: Wasserstein-1 at the optimum. Either
    way `g` only reweights, so the critic's input gradient never reaches `g`.
    """

    bce = auto()
    wasserstein = auto()


class Resample(StrEnum):
    """Which samples an uncertainty design's bootstrap resamples.

    `both` gives the combined statistical uncertainty. `data` and `mc` resample
    one side and leave the other as collected, so separate designs can report
    the data and simulation statistics apart, as analyses usually quote them.
    """

    both = auto()
    data = auto()
    mc = auto()
