from __future__ import annotations

from typing import TYPE_CHECKING

from . import logging_config, timing
from .logging_config import configure_logging
from .timing import (
    TIMING_ENV_VAR,
    Phase,
    enable,
    is_enabled,
    note,
    phase,
    phases,
    record,
    report,
    reset,
    write,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "TIMING_ENV_VAR",
    "Phase",
    "configure_logging",
    "enable",
    "is_enabled",
    "logging_config",
    "note",
    "phase",
    "phases",
    "record",
    "report",
    "reset",
    "timing",
    "write",
)
