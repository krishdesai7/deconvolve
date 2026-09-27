"""Layered configuration discovery and merge.

A leaf module: it imports only the standard library, so it can be unit-tested
without paying for JAX, Keras or Typer. The command tree it validates against
is built separately in `config_spec.py` and passed in.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from typing import Any, Final, LiteralString

    from .config_spec import CommandSpec

# `tool.deconvolve` in a pyproject; the whole document in a `deconvolve.toml`.
_PYPROJECT_TABLE: Final[tuple[LiteralString, ...]] = ("tool", "deconvolve")


class ConfigError(Exception):
    """A configuration file exists but cannot be used as written."""


@dataclass(frozen=True)
class Layer:
    """One config file's contribution, with the string that names its source."""

    origin: str
    data: dict[str, Any]


def _parse(path: Path, /) -> dict[str, Any]:
    """Read one TOML document, or explain why it could not be read."""
    try:
        with path.open(mode="rb") as handle:
            return tomllib.load(handle)
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"{path}: invalid TOML: {error}") from error
    except OSError as error:
        raise ConfigError(f"{path}: cannot be read: {error.strerror}") from error


def _table(data: Mapping[str, Any], keys: tuple[str, ...], /) -> dict[str, Any] | None:
    """Follow a dotted table path, or return None if any level is absent."""
    current: object = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        table: dict[str, Any] = current
        if key not in table:
            return None
        current = table[key]
    return current if isinstance(current, dict) else None


def _global_layer(environ: Mapping[str, str], /) -> Layer | None:
    """`$XDG_CONFIG_HOME/deconvolve/deconvolve.toml`, defaulting to `~/.config`."""
    base: str | None = environ.get("XDG_CONFIG_HOME")
    root: Path = (
        Path(base) if base else Path(environ.get("HOME", "~")).expanduser() / ".config"
    )
    path: Path = root / "deconvolve" / "deconvolve.toml"
    if not path.is_file():
        return None
    return Layer(origin=f"deconvolve.toml:{path}", data=_parse(path))


def _project_layer_in(directory: Path, /) -> Layer | None:
    """This directory's contribution, `deconvolve.toml` shadowing `pyproject.toml`."""
    standalone: Path = directory / "deconvolve.toml"
    if standalone.is_file():
        return Layer(origin=f"deconvolve.toml:{standalone}", data=_parse(standalone))

    pyproject: Path = directory / "pyproject.toml"
    if not pyproject.is_file():
        return None
    table: dict[str, Any] | None = _table(_parse(pyproject), _PYPROJECT_TABLE)
    if table is None:
        # A pyproject with no `[tool.deconvolve]` is not a config file, so it does not
        # stop the walk -- an outer directory may still hold the real one.
        return None
    return Layer(origin=f"pyproject.toml:{pyproject}", data=table)


def _project_layer(cwd: Path, /) -> Layer | None:
    """Walk up from `cwd`, first hit wins, never past the git root."""
    for directory in (cwd.resolve(), *cwd.resolve().parents):
        layer: Layer | None = _project_layer_in(directory)
        if layer is not None:
            return layer
        # Checked *after* this directory's own files: the repository root is
        # allowed to hold the config, it is just the last place that may.
        if (directory / ".git").exists():
            return None
    return None


def discover(cwd: Path, environ: Mapping[str, str]) -> tuple[Layer, ...]:
    """The layer files that apply here, lowest precedence first."""
    found: list[Layer] = []
    if (layer := _global_layer(environ)) is not None:
        found.append(layer)
    if (layer := _project_layer(cwd)) is not None:
        found.append(layer)
    return tuple(found)


@dataclass(frozen=True)
class Resolved:
    """Merged values, where each came from, and the files consulted."""

    values: dict[tuple[str, ...], dict[str, Any]]
    origins: dict[tuple[str, ...], dict[str, str]]
    layers: tuple[Layer, ...]


def _normalize(key: str, /) -> str:
    return key.replace("-", "_")


def _display(key: str, /) -> str:
    """Config files are written in kebab-case, so errors speak it."""
    return key.replace("_", "-")


def _suggestion(key: str, candidates: Iterable[str], /) -> str:
    matches: list[str] = get_close_matches(
        word=_normalize(key), possibilities=[_normalize(c) for c in candidates], n=1
    )
    return f", did you mean `{_display(matches[0])}`?" if matches else ""


def _coerce_int(value: Any, key: str, where: str, /) -> Any:
    """Reject a float that would lose information becoming an int."""
    if isinstance(value, bool) or not isinstance(value, float):
        return value
    if not value.is_integer():
        raise ConfigError(
            f"{where}: `{_display(key)} = {value}` is not a whole number, "
            f"but this option is an integer"
        )
    return int(value)


def _check_option(key: str, value: Any, spec: CommandSpec, where: str, /) -> Any:
    """Validate one scalar key against the command it was written under."""
    name: str = _normalize(key)
    if name in spec.excluded:
        raise ConfigError(
            f"{where}: `{_display(key)}` cannot be configured -- it names a "
            f"specific run or a destructive action, so it must be given on the "
            f"command line"
        )
    if name not in spec.options:
        raise ConfigError(
            f"{where}: unknown option `{_display(key)}`"
            f"{_suggestion(key, (*spec.options, *spec.excluded))}"
        )
    if spec.options[name] is int:
        return _coerce_int(value, key, where)
    return value


def _walk(
    data: Mapping[str, Any],
    spec: CommandSpec,
    path: tuple[str, ...],
    origin: str,
    into: Resolved,
    /,
) -> None:
    """Validate one table and fold it into the accumulating result."""
    where: str = f"{origin.split(sep=':', maxsplit=1)[1]}{_table_label(path)}"
    for key, value in data.items():
        name: str = _normalize(key)
        if isinstance(value, dict):
            # Option keys always normalize `-` -> `_`; command names never do
            # -- no registered command name contains an underscore standing in
            # for a hyphen (e.g. `leakage-check`), so a table name is looked
            # up literally, with no normalized fallback. `[leakage_check]` is
            # therefore an unknown table, caught below with a `did you mean`
            # suggestion, not silently resolved to `leakage-check`.
            child: CommandSpec | None = spec.children.get(key)
            if child is None:
                raise ConfigError(
                    f"{where}: unknown command table `{_display(key)}`"
                    f"{_suggestion(key, spec.children)}"
                )
            _walk(value, child, (*path, key), origin, into)
            continue
        into.values.setdefault(path, {})[name] = _check_option(key, value, spec, where)
        into.origins.setdefault(path, {})[name] = origin


def _table_label(path: tuple[str, ...], /) -> str:
    return f" [{'.'.join(path)}]" if path else ""


def load(layers: tuple[Layer, ...], spec: CommandSpec) -> Resolved:
    """Validate every layer against the command tree and merge them in order."""
    resolved = Resolved(values={}, origins={}, layers=layers)
    for layer in layers:
        _walk(layer.data, spec, (), layer.origin, resolved)
    return resolved


def default_map(resolved: Resolved) -> dict[str, Any]:
    """Nest the merged values by command path, as `ctx.default_map` wants."""
    root: dict[str, Any] = {}
    for path, values in resolved.values.items():
        node: dict[str, Any] = root
        for part in path:
            node = node.setdefault(part, {})
        node.update(values)
    return root


# Click's own names for where a value came from. `DEFAULT_MAP` means a config
# file, and `Resolved.origins` knows which one.
_SOURCE_LABELS: Final[Mapping[str, str]] = {
    "COMMANDLINE": "command-line",
    "ENVIRONMENT": "environment",
    "PROMPT": "prompt",
    "DEFAULT": "default",
}


def origins_for(
    ctx: Any, resolved: Resolved, path: tuple[str, ...], *, names: Iterable[str]
) -> dict[str, str]:
    """Where each named parameter's value actually came from, for the record."""
    from_files: Mapping[str, str] = resolved.origins.get(path, {})
    origins: dict[str, str] = {}
    for name in names:
        source: str = ctx.get_parameter_source(name).name
        origins[name] = (
            from_files.get(name, "config-file")
            if source == "DEFAULT_MAP"
            else _SOURCE_LABELS.get(source, source.lower())
        )
    return origins
