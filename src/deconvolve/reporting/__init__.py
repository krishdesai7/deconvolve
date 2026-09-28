from __future__ import annotations

from typing import TYPE_CHECKING

from . import report
from .report import (
    TEMPLATE_TOKEN,
    build_report,
    config_rows,
    decimal,
    latex_text,
    load_template,
    metrics_table,
    render,
    skipped_variables,
    timing_rows,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "TEMPLATE_TOKEN",
    "build_report",
    "config_rows",
    "decimal",
    "latex_text",
    "load_template",
    "metrics_table",
    "render",
    "report",
    "skipped_variables",
    "timing_rows",
)
