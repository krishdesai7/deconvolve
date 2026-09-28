"""The hand-written CLI reference matches the CLI it documents.

`docs/user-guide/cli.md` is maintained by hand rather than generated, so this
is what keeps it honest: every long option a command defines has to appear in
that command's section, and every option a section documents has to exist on
that command. A section is everything from a `` `deconvolve ...` `` heading to
the next one; `Global Options` is the root command's.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import typer
from deconvolve.cli import app
from typer.core import TyperGroup

if TYPE_CHECKING:
    from typer._click.core import Command

PAGE: Path = Path(__file__).parents[1] / "docs" / "user-guide" / "cli.md"
_COMMAND_HEADING: re.Pattern[str] = re.compile(r"^#{2,3} `deconvolve ?([^`]*)`$")
_GLOBAL_HEADING: str = "## Global Options"
_OPTION: re.Pattern[str] = re.compile(r"`(--[a-z][a-z0-9-]*)")


def _sections() -> dict[str, str]:
    """Page text keyed by command path (`""` for the root, `"baseline ibu"`)."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in PAGE.read_text().splitlines():
        if line == _GLOBAL_HEADING:
            current = ""
        elif match := _COMMAND_HEADING.match(line):
            current = match.group(1)
        if current is not None:
            sections.setdefault(current, []).append(line)
    return {path: "\n".join(lines) for path, lines in sections.items()}


def _commands() -> dict[str, Command]:
    """Every command in the tree, keyed by path."""
    found: dict[str, Command] = {}

    def walk(command: Command, path: str) -> None:
        found[path] = command
        if isinstance(command, TyperGroup):
            for name, child in command.commands.items():
                walk(child, f"{path} {name}".strip())

    walk(typer.main.get_command(app), "")
    return found


def _long_options(command: Command) -> set[str]:
    # `opts` only: a `--force / --no-force` flag is documented by its first
    # spelling, and `--help` is on every command without being documented.
    return {
        opt
        for param in command.params
        for opt in getattr(param, "opts", ())
        if opt.startswith("--") and opt != "--help"
    }


COMMANDS: dict[str, Command] = _commands()
SECTIONS: dict[str, str] = _sections()
DOCUMENTED: list[str] = sorted(
    path for path, command in COMMANDS.items() if _long_options(command)
)


def test_every_leaf_command_has_a_section() -> None:
    leaves: set[str] = {
        path
        for path, command in COMMANDS.items()
        if not isinstance(command, TyperGroup)
    }
    assert leaves - SECTIONS.keys() == set()


@pytest.mark.parametrize("path", DOCUMENTED, ids=lambda path: path or "root")
def test_every_option_is_documented_in_its_section(path: str) -> None:
    section: str = SECTIONS.get(path, "")
    missing: set[str] = {
        opt for opt in _long_options(COMMANDS[path]) if f"`{opt}" not in section
    }
    assert missing == set()


@pytest.mark.parametrize("path", sorted(SECTIONS), ids=lambda path: path or "root")
def test_every_documented_option_exists(path: str) -> None:
    command: Command | None = COMMANDS.get(path)
    assert command is not None, f"section for a command that does not exist: {path}"
    # Click adds `--help` itself, outside `params`.
    live: set[str] = {"--help"} | {
        opt
        for param in command.params
        for opt in (*getattr(param, "opts", ()), *getattr(param, "secondary_opts", ()))
    }
    assert set(_OPTION.findall(SECTIONS[path])) - live == set()
