import logging

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging with Rich terminal rendering."""
    normalized: str = level.upper()
    numeric_level: int | None = logging.getLevelNamesMapping().get(normalized)
    if numeric_level is None:
        raise ValueError(f"Unknown log level: {level!r}")

    handler = RichHandler(
        console=Console(stderr=True),
        markup=False,
        rich_tracebacks=True,
        show_path=False,
    )
    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[handler],
        force=True,
    )
