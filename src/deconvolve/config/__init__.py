"""Layered configuration: discovery and merge (`layers`), the command tree it
validates against (`spec`), and `deconvolve config show` rendering (`show`).

Only `layers` is re-exported here, so importing this package stays stdlib-only;
`spec` pulls in Typer and `show` pulls in Rich, and are imported by path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .layers import (
    ConfigError,
    Layer,
    Resolved,
    default_map,
    discover,
    load,
    origins_for,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "ConfigError",
    "Layer",
    "Resolved",
    "default_map",
    "discover",
    "load",
    "origins_for",
)
