"""The command tree, as a description of what configuration may name.

Derived from the live Typer app rather than from a hand-maintained table, so a
new command or a new flag is validated the moment it is added.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, get_args, get_origin, get_type_hints

from typer.models import ArgumentInfo

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import Any, Final, LiteralString

    import typer

# Per-invocation identity and destructive toggles. Keyed by command path; the
# empty tuple is the root. `force` is listed per command rather than globally
# so that the error message can say which command it was rejected for.
NOT_LAYERABLE: Final[Mapping[tuple[str, ...], frozenset[str]]] = {
    ("train",): frozenset({"load_run"}),
    ("evaluate",): frozenset({"force"}),
    ("report",): frozenset({"force"}),
    ("baseline", "ibu"): frozenset({"force"}),
    ("baseline", "omnifold"): frozenset({"force"}),
    # `n_datasets`, `n_seeds`, `data_seed` and `init_seed` shape the design's
    # grid, and `design.json` is their single authority once `freeze` has run
    # -- an ambient file must not be able to reach them either. `n_bins` stays
    # layerable: it is a presentation choice for `collect`'s figure, not part
    # of the grid.
    ("uncertainty", "collect"): frozenset(
        {"n_datasets", "n_seeds", "data_seed", "init_seed"}
    ),
    ("uncertainty", "freeze"): frozenset({"force"}),
    # None of `show`'s own options are layerable: `command` is positional
    # already, and `as_json` is a per-invocation output-format toggle, not a
    # preference worth inheriting from a file.
    ("config", "show"): frozenset({"as_json"}),
}

# Resolved from `design.json`, never from a config file. See the spec's
# "uncertainty freeze path".
FROZEN_COMMANDS: Final[frozenset[tuple[LiteralString, ...]]] = frozenset(
    {("uncertainty", "run")}
)

_CONTEXT_PARAMS: Final[frozenset[LiteralString]] = frozenset({"ctx", "context"})


@dataclass(frozen=True)
class CommandSpec:
    """Layerable option names and types at one node of the command tree."""

    options: dict[str, Any] = field(default_factory=dict)
    excluded: frozenset[str] = frozenset()
    children: dict[str, CommandSpec] = field(default_factory=dict)


def _is_argument(annotation: Any, /) -> bool:
    """Whether a parameter is positional, i.e. carries a `typer.Argument`."""
    if get_origin(tp=annotation) is not Annotated:
        return False
    return any(isinstance(meta, ArgumentInfo) for meta in get_args(tp=annotation)[1:])


def _base_type(annotation: Any, /) -> Any:
    """Strip `Annotated[...]` down to the declared type."""
    if get_origin(tp=annotation) is Annotated:
        return get_args(tp=annotation)[0]
    return annotation


def _options_of(
    callback: Any, path: tuple[str, ...], /
) -> tuple[dict[str, Any], frozenset[str]]:
    """Split one command's parameters into layerable and excluded."""
    hints: dict[str, Any] = get_type_hints(obj=callback, include_extras=True)
    denied: frozenset[str] = NOT_LAYERABLE.get(path, frozenset())

    options: dict[str, Any] = {}
    excluded: set[str] = set(denied)
    for name in inspect.signature(obj=callback).parameters:
        if name in _CONTEXT_PARAMS:
            continue
        annotation: Any = hints.get(name, str)
        if _is_argument(annotation) or name in denied:
            excluded.add(name)
            continue
        options[name] = _base_type(annotation)
    return options, frozenset(excluded)


def _command_children(
    app: typer.Typer, path: tuple[str, ...], /
) -> dict[str, CommandSpec]:
    """Describe each `@app.command()` registered directly on `app`."""
    children: dict[str, CommandSpec] = {}
    for command in app.registered_commands:
        callback: Callable[..., Any] | None = command.callback
        if callback is None:
            continue
        name: str = command.name or getattr(callback, "__name__", "")
        if not name:
            continue
        child_path: tuple[str, ...] = (*path, name)
        if child_path in FROZEN_COMMANDS:
            continue
        options, excluded = _options_of(callback, child_path)
        children[name] = CommandSpec(options=options, excluded=excluded)
    return children


def _group_children(
    app: typer.Typer, path: tuple[str, ...], /
) -> dict[str, CommandSpec]:
    """Describe each sub-app registered on `app` via `add_typer`."""
    children: dict[str, CommandSpec] = {}
    for group in app.registered_groups:
        if group.name is None or group.typer_instance is None:
            continue
        children[group.name] = build_spec(
            app=group.typer_instance, path=(*path, group.name)
        )
    return children


def build_spec(app: typer.Typer, path: tuple[str, ...] = ()) -> CommandSpec:
    """Describe `app` and everything registered under it."""
    options: dict[str, Any] = {}
    excluded: frozenset[str] = frozenset()
    if app.registered_callback is not None and app.registered_callback.callback:
        options, excluded = _options_of(app.registered_callback.callback, path)

    children: dict[str, CommandSpec] = {
        **_command_children(app, path),
        **_group_children(app, path),
    }
    return CommandSpec(options=options, excluded=excluded, children=children)
