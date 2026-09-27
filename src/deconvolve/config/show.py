"""Render what the config *files* said, and where each such value came from.

This answers "what did the config files say", not "what is every option's
effective value": `resolved.values` only ever holds keys a file actually
supplied, so an option left at its code default, or set only through a
`DECONVOLVE_*` environment variable, never appears here. See `render`'s docstring
for the fully-resolved alternatives.

`rich` with an injectable `Console`, matching `uncertainty/report.py`; `print`
is banned in `src/` by `tests/test_source_hygiene.py`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Column, Table

from .coretypes.constants import CACHE_ENV_VAR

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import Layer, Resolved

# Resolved at import in their own modules, so they cannot join the layered
# stack. See the spec's "Deferred" section.
_ENV_ONLY: tuple[tuple[str, str], ...] = (
    (CACHE_ENV_VAR, ".cache"),
    ("DECONVOLVE_TIMING", "off"),
)


def _dotted(path: tuple[str, ...], key: str, /) -> str:
    return ".".join((*path, key.replace("_", "-")))


def _nest(
    source: Mapping[tuple[str, ...], Mapping[str, object]], /
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for path, entries in source.items():
        label: str = ".".join(path) or "."
        out[label] = {key.replace("_", "-"): value for key, value in entries.items()}
    return out


def _scoped(
    source: Mapping[tuple[str, ...], Mapping[str, object]], command: str | None, /
) -> Mapping[tuple[str, ...], Mapping[str, object]]:
    """The same per-path scoping `_values_table` applies, before nesting."""
    if command is None:
        return source
    return {
        path: entries for path, entries in source.items() if path and path[0] == command
    }


def _payload(
    resolved: Resolved, command: str | None, /
) -> dict[str, dict[str, dict[str, object]]]:
    """The same content as the table, shaped for `--json` -- and scoped like it.

    `command` narrows this the same way it narrows `_values_table`: without
    it, `config show train --json` silently ignored the scope argument and
    emitted every command's values.
    """
    return {
        "values": _nest(_scoped(resolved.values, command)),
        "origins": _nest(_scoped(resolved.origins, command)),
    }


def _roles(layers: tuple[Layer, ...], /) -> tuple[str | None, ...]:
    """The role of each layer, when position alone determines it.

    `discover()` has a fixed order -- global layer, then project layer -- so
    two layers are unambiguously (global, project). One layer could be
    either, so its role is unknown from the tuple alone.
    """
    return ("global", "project") if len(layers) == 2 else (None,) * len(layers)


def _layers_table(resolved: Resolved, /) -> Table:
    table = Table(
        Column("path", overflow="fold"),
        Column("role"),
        title="Layers (nearest last)",
        show_header=False,
        box=None,
    )
    if not resolved.layers:
        table.add_row("(no configuration files found)")
    for layer, role in zip(resolved.layers, _roles(resolved.layers), strict=True):
        kind, _, path = layer.origin.partition(":")
        table.add_row(path, role or kind)
    return table


def _short_origins(layers: tuple[Layer, ...], /) -> dict[str, str]:
    """Map each layer's full origin to a short label for the values table.

    Usually just the filename (`deconvolve.toml`), but `discover()` walks the global
    layer before the project one, and both are commonly named `deconvolve.toml` --
    the exact case that made two different files render identically before
    this fix. When a filename repeats, the role disambiguates it.
    """
    names: list[str] = [layer.origin.partition(":")[0] for layer in layers]
    roles: tuple[str | None, ...] = _roles(layers)
    labels: dict[str, str] = {}
    for layer, name, role in zip(layers, names, roles, strict=True):
        if names.count(name) > 1 and role is not None:
            labels[layer.origin] = f"{name} ({role})"
        else:
            labels[layer.origin] = name
    return labels


def _values_table(resolved: Resolved, command: str | None, /) -> Table:
    short: dict[str, str] = _short_origins(resolved.layers)
    table = Table(show_header=True, box=None)
    table.add_column("setting")
    table.add_column("value")
    table.add_column("origin")
    for path in sorted(resolved.values):
        if command is not None and (not path or path[0] != command):
            continue
        for key, value in sorted(resolved.values[path].items()):
            origin: str = resolved.origins[path][key]
            table.add_row(
                _dotted(path, key), str(object=value), short.get(origin, origin)
            )
    return table


def _environment_table(environ: Mapping[str, str], /) -> Table:
    table = Table(title="Environment-only (not layered)", show_header=False, box=None)
    for name, fallback in _ENV_ONLY:
        table.add_row(name, environ.get(name, f"{fallback} (unset)"))
    return table


def render(
    resolved: Resolved,
    environ: Mapping[str, str],
    *,
    console: Console | None = None,
    command: str | None = None,
    as_json: bool = False,
) -> None:
    """Print what the config files said, as a table or as JSON.

    Not a full accounting of every option's effective value: a value at its
    code default, or set only through a `DECONVOLVE_*` environment variable, is
    absent from `resolved.values` and so absent here too. For that, use
    `deconvolve <command> --help` (which renders the effective default Click would
    apply) or a run's `config.json` `_origin` block, which records where
    every value that actually reached that run came from.
    """
    active: Console = console or Console()
    if as_json:
        active.print_json(json=json.dumps(obj=_payload(resolved, command), default=str))
        return
    active.print(_layers_table(resolved))
    active.print(_values_table(resolved, command))
    active.print(_environment_table(environ))
