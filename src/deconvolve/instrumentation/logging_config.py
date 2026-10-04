import logging

from rich.console import Console
from rich.logging import RichHandler

APP_LOGGER: str = "deconvolve"
"""The namespace `--log-level` governs. Every module logs under it via
`getLogger(__name__)`, and the benchmarks name theirs `deconvolve.<script>`."""

_FONT_SUBSETTER: str = "fontTools.subset"
"""Driven by matplotlib on every PDF save. Besides its INFO stream it warns
`TeX  NOT subset; don't know how to subset; dropped` for a font table it does
not recognise, once or twice per figure -- harmless, and nothing a user can
act on -- so it is held to errors below DEBUG."""


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging with Rich terminal rendering.

    `level` applies to the `deconvolve` loggers. Everything else -- JAX,
    matplotlib, and above all `fontTools.subset`, which logs every step of the
    font subsetting behind each saved PDF at INFO -- is held at WARNING, so an
    ordinary run's terminal is this package's own output plus any library that
    actually has something to warn about.

    `DEBUG` is the exception and opens every logger, libraries included: it
    is the level asked for when the question is what a dependency is doing.
    The font subsetter's warnings are held back too, except at DEBUG.
    """
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
    root_level: int = (
        numeric_level
        if numeric_level <= logging.DEBUG
        else max(numeric_level, logging.WARNING)
    )
    logging.basicConfig(
        level=root_level,
        format="%(message)s",
        handlers=[handler],
        force=True,
    )
    logging.getLogger(APP_LOGGER).setLevel(numeric_level)
    logging.getLogger(_FONT_SUBSETTER).setLevel(
        logging.NOTSET if numeric_level <= logging.DEBUG else logging.ERROR
    )
