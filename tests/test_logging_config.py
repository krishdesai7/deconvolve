from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
from rich.logging import RichHandler

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture(autouse=True)
def restore_root_logging() -> Iterator[None]:
    root = logging.getLogger()
    pinned = [logging.getLogger(name) for name in ("deconvolve", "fontTools.subset")]
    old_handlers = root.handlers[:]
    old_level = root.level
    old_pinned_levels = [logger.level for logger in pinned]
    yield
    root.handlers[:] = old_handlers
    root.setLevel(old_level)
    for logger, level in zip(pinned, old_pinned_levels, strict=True):
        logger.setLevel(level)


def test_configure_logging_installs_one_rich_handler_and_level() -> None:
    from deconvolve.instrumentation.logging_config import configure_logging

    configure_logging("debug")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], RichHandler)


def test_configure_logging_is_deterministic_when_called_twice() -> None:
    from deconvolve.instrumentation.logging_config import configure_logging

    configure_logging("debug")
    configure_logging("WARNING")

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert logging.getLogger("deconvolve").level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], RichHandler)


def test_configure_logging_rejects_unknown_level() -> None:
    from deconvolve.instrumentation.logging_config import configure_logging

    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("verbose")


def test_info_is_scoped_to_this_package() -> None:
    """`fontTools.subset` logs every glyph table of every saved PDF at INFO."""
    from deconvolve.instrumentation.logging_config import configure_logging

    configure_logging("INFO")

    assert logging.getLogger("deconvolve.workflows.train").isEnabledFor(logging.INFO)
    assert logging.getLogger("deconvolve.sliced").isEnabledFor(logging.INFO)
    assert not logging.getLogger("fontTools.subset").isEnabledFor(logging.INFO)
    # Other libraries still get to warn; only the font subsetter is quieter.
    assert logging.getLogger("jax").isEnabledFor(logging.WARNING)
    assert not logging.getLogger("fontTools.subset").isEnabledFor(logging.WARNING)


def test_debug_opens_every_logger() -> None:
    """Debug is for asking what a dependency is doing, so nothing is held back."""
    from deconvolve.instrumentation.logging_config import configure_logging

    configure_logging("DEBUG")

    assert logging.getLogger("fontTools.subset").isEnabledFor(logging.DEBUG)
    assert logging.getLogger("deconvolve.evaluation").isEnabledFor(logging.DEBUG)


def test_error_raises_the_floor_for_libraries_too() -> None:
    from deconvolve.instrumentation.logging_config import configure_logging

    configure_logging("ERROR")

    assert not logging.getLogger("jax").isEnabledFor(logging.WARNING)
    assert not logging.getLogger("deconvolve").isEnabledFor(logging.WARNING)
