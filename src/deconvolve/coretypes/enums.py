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


class Resample(StrEnum):
    """Which samples an uncertainty design's bootstrap resamples.

    `both` gives the combined statistical uncertainty. `data` and `mc` resample
    one side and leave the other as collected, so separate designs can report
    the data and simulation statistics apart, as analyses usually quote them.
    """

    both = auto()
    data = auto()
    mc = auto()
