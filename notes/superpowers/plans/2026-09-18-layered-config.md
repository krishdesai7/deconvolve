# Layered Configuration Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `ran` a five-layer configuration stack — code default < XDG global file < project file < environment < command line — so per-machine and per-project defaults live in a file instead of a shell alias.

**Architecture:** A new leaf module `src/ran/config.py` discovers and merges TOML files, then hands the result to Click's native `default_map` via `ctx.default_map` inside the existing `@app.callback()`. Click already implements the precedence, the type coercion and the constraint validation, so no resolver is written. Uncertainty design cells bypass the stack entirely and read a frozen `design.json`.

**Tech Stack:** Python 3.14 (`tomllib` from the stdlib), Typer 0.27.2 (vendors Click privately as `typer._click`; this plan touches no private API), `rich` for `config show` output, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-layered-config-design.md`

## Global Constraints

- **Python `>=3.14`.** `tomllib` is stdlib. Add no new dependency for any task in this plan.
- **No private Typer API.** Reach `default_map` and `get_parameter_source` through the public `typer.Context` only. Never `import typer._click` — it is vendored and unstable.
- **`print` and `fire` are banned in `src/` and `scripts/`**, enforced by `tests/test_source_hygiene.py:23`. Use `rich.console.Console`, following the injectable-console pattern at `src/ran/uncertainty/report.py:165-166`.
- **Max cyclomatic complexity is 10** (`pyproject.toml:218`, `max-complexity-allowed`). Validation code branches a lot; split helpers rather than growing one function.
- **`from __future__ import annotations` at the top of every module**, matching every existing file in `src/ran/`.
- **Type annotations on every parameter and return**, including keyword-only. `just typecheck` runs `pyrefly check --min-severity info` and `uv check` (which is `ty`). Imports used only in annotations go under `if TYPE_CHECKING:`.
- **Never layer the physics spec.** `params/*.yaml` stays reachable only through an explicit `--config` path.
- Run `uv run just test-fast` after each task; `uv run just validate` before the final commit. Nothing in this plan is marked `slow` or needs a GPU.

### Naming note for the implementer

The spec names the new module `src/ran/config.py`. Note that `src/ran/data/config.py` already exists and holds the *physics* (Gaussian) config. The two are different packages so there is no import conflict, but when writing imports prefer the explicit `from ran import config as ran_config` over a bare `from . import config` inside `src/ran/`, so a reader never has to work out which `config` is meant. The new test file is `tests/test_layered_config.py`, because `tests/test_config.py` is already the physics one.

### Merge semantics clarification

The spec's layer table says each layer overrides the ones above it but does not spell out the granularity. **Merging across layers 2 and 3 is per-key, not wholesale**: a global `~/.config/ran/ran.toml` setting `train.lr-g` and a project `ran.toml` setting `train.n-epochs` both apply. This is uv's behaviour and is what "overrides" means throughout this plan.

Note this is distinct from the *shadowing* rule inside layer 3: when one directory holds both `ran.toml` and a `pyproject.toml` with `[tool.ran]`, the `ran.toml` replaces the `pyproject.toml` entirely and the latter contributes nothing.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ran/config.py` (create) | Discovery, parsing, validation, merge, `default_map` construction. Leaf module: imports only stdlib. |
| `src/ran/config_spec.py` (create) | Introspects the live Typer app into a `CommandSpec` tree of layerable option names and types. Separated from `config.py` so the latter stays a stdlib-only leaf, importable without Typer. |
| `src/ran/cli.py` (modify) | Callback assigns `ctx.default_map`; new `config` sub-app; new `uncertainty freeze`; `uncertainty run` resolves from the frozen spec. |
| `src/ran/config_show.py` (create) | Renders the resolved stack with `rich`. Injectable `Console`, mirroring `uncertainty/report.py`. |
| `src/ran/uncertainty/design.py` (modify) | `freeze_design()` / `load_frozen()` for `design.json`. |
| `src/ran/workflows/train.py` (modify) | `_origin` block in `config.json`. |
| `tests/test_layered_config.py` (create) | Discovery, validation, merge — pure unit tests, no JAX import. |
| `tests/test_config_cli.py` (create) | End-to-end precedence via `CliRunner`; `config show`; freeze behaviour. |
| `scripts/submit_uncertainty.zsh` (modify) | Call `ran uncertainty freeze` between `mkdir` and `sbatch`. |

---

### Task 1: Discovery — finding the layer files

**Files:**
- Create: `src/ran/config.py`
- Test: `tests/test_layered_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ConfigError(Exception)`; `Layer` (frozen dataclass, fields `origin: str`, `data: dict[str, Any]`); `discover(cwd: Path, environ: Mapping[str, str]) -> tuple[Layer, ...]` returning global-then-project order.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_layered_config.py
from __future__ import annotations

from pathlib import Path

import pytest
from ran.config import ConfigError, discover


def _repo(tmp_path: Path) -> Path:
    """A directory that looks like a git repository root."""
    (tmp_path / ".git").mkdir()
    return tmp_path


def test_no_config_anywhere_discovers_nothing(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    assert discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")}) == ()


def test_project_ran_toml_is_found_from_a_subdirectory(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "ran.toml").write_text("[train]\nn-epochs = 500\n")
    deep = root / "a" / "b"
    deep.mkdir(parents=True)

    layers = discover(cwd=deep, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}
    assert layers[0].origin == f"ran.toml:{root / 'ran.toml'}"


def test_the_walk_stops_at_the_git_root(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ran.toml").write_text("[train]\nn-epochs = 1\n")
    root = outside / "repo"
    root.mkdir()
    (root / ".git").mkdir()

    assert discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")}) == ()


def test_a_git_file_bounds_the_walk_as_a_git_directory_does(tmp_path: Path) -> None:
    """A linked worktree's `.git` is a file, not a directory."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ran.toml").write_text("[train]\nn-epochs = 1\n")
    root = outside / "wt"
    root.mkdir()
    (root / ".git").write_text("gitdir: /elsewhere\n")

    assert discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")}) == ()


def test_ran_toml_shadows_pyproject_in_the_same_directory(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "ran.toml").write_text("[train]\nn-epochs = 500\n")
    (root / "pyproject.toml").write_text("[tool.ran.train]\nn-epochs = 9\nlr-g = 0.1\n")

    layers = discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}


def test_the_nearest_directory_wins_and_the_walk_stops(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "ran.toml").write_text("[train]\nn-epochs = 1\n")
    near = root / "a"
    near.mkdir()
    (near / "ran.toml").write_text("[train]\nn-epochs = 500\n")

    layers = discover(cwd=near, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}


def test_a_pyproject_without_a_tool_ran_table_does_not_stop_the_walk(
    tmp_path: Path,
) -> None:
    root = _repo(tmp_path)
    (root / "ran.toml").write_text("[train]\nn-epochs = 500\n")
    near = root / "a"
    near.mkdir()
    (near / "pyproject.toml").write_text('[project]\nname = "unrelated"\n')

    layers = discover(cwd=near, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    assert len(layers) == 1
    assert layers[0].data == {"train": {"n-epochs": 500}}


def test_the_global_layer_comes_first(tmp_path: Path) -> None:
    xdg = tmp_path / "xdg"
    (xdg / "ran").mkdir(parents=True)
    (xdg / "ran" / "ran.toml").write_text("[train]\nlr-g = 0.1\n")
    root = _repo(tmp_path / "repo")
    (root / "ran.toml").write_text("[train]\nn-epochs = 500\n")

    layers = discover(cwd=root, environ={"XDG_CONFIG_HOME": str(xdg)})

    assert [layer.data for layer in layers] == [
        {"train": {"lr-g": 0.1}},
        {"train": {"n-epochs": 500}},
    ]


def test_xdg_config_home_defaults_to_dot_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    (home / ".config" / "ran").mkdir(parents=True)
    (home / ".config" / "ran" / "ran.toml").write_text("[train]\nlr-g = 0.1\n")
    root = _repo(tmp_path / "repo")

    layers = discover(cwd=root, environ={"HOME": str(home)})

    assert [layer.data for layer in layers] == [{"train": {"lr-g": 0.1}}]


def test_malformed_toml_names_the_path_and_the_position(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    (root / "ran.toml").write_text("[train\nn-epochs = 500\n")

    with pytest.raises(ConfigError) as excinfo:
        discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})

    message = str(excinfo.value)
    assert str(root / "ran.toml") in message
    assert "line 1" in message


def test_an_unreadable_file_names_the_path(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    broken = root / "ran.toml"
    broken.write_text("[train]\n")
    broken.chmod(0o000)
    try:
        with pytest.raises(ConfigError) as excinfo:
            discover(cwd=root, environ={"XDG_CONFIG_HOME": str(tmp_path / "nope")})
        assert str(broken) in str(excinfo.value)
    finally:
        broken.chmod(0o644)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_layered_config.py -v`
Expected: every test FAILS with `ModuleNotFoundError: No module named 'ran.config'`.

- [ ] **Step 3: Write the implementation**

```python
# src/ran/config.py
"""Layered configuration discovery and merge.

A leaf module: it imports only the standard library, so it can be unit-tested
without paying for JAX, Keras or Typer. The command tree it validates against
is built separately in `config_spec.py` and passed in.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any, Final, LiteralString

# `tool.ran` in a pyproject; the whole document in a `ran.toml`.
_PYPROJECT_TABLE: Final[tuple[LiteralString, ...]] = ("tool", "ran")


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
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, dict) else None


def _global_layer(environ: Mapping[str, str], /) -> Layer | None:
    """`$XDG_CONFIG_HOME/ran/ran.toml`, defaulting to `~/.config`."""
    base: str | None = environ.get("XDG_CONFIG_HOME")
    root: Path = (
        Path(base) if base else Path(environ.get("HOME", "~")).expanduser() / ".config"
    )
    path: Path = root / "ran" / "ran.toml"
    if not path.is_file():
        return None
    return Layer(origin=f"ran.toml:{path}", data=_parse(path))


def _project_layer_in(directory: Path, /) -> Layer | None:
    """This directory's contribution, `ran.toml` shadowing `pyproject.toml`."""
    standalone: Path = directory / "ran.toml"
    if standalone.is_file():
        return Layer(origin=f"ran.toml:{standalone}", data=_parse(standalone))

    pyproject: Path = directory / "pyproject.toml"
    if not pyproject.is_file():
        return None
    table: dict[str, Any] | None = _table(_parse(pyproject), _PYPROJECT_TABLE)
    if table is None:
        # A pyproject with no `[tool.ran]` is not a config file, so it does not
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_layered_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ran/config.py tests/test_layered_config.py
git commit -m "feat(config): discover XDG and project config layers

Nearest-wins walk from cwd, bounded by the git root. ran.toml shadows
a pyproject [tool.ran] in the same directory; a pyproject without that
table does not stop the walk.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Command spec — what is layerable

**Files:**
- Create: `src/ran/config_spec.py`
- Test: `tests/test_layered_config.py` (append)

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `CommandSpec` (frozen dataclass, fields `options: dict[str, Any]` mapping option name to its annotation, `excluded: frozenset[str]`, `children: dict[str, CommandSpec]`); `build_spec(app: typer.Typer) -> CommandSpec`; `NOT_LAYERABLE: Mapping[tuple[str, ...], frozenset[str]]`.

Option names are **Python parameter names** (`n_epochs`), because that is what Click's `default_map` keys on. Normalization of the TOML's `n-epochs` happens in Task 3.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_layered_config.py
from ran.cli import app
from ran.config_spec import build_spec


def test_the_spec_mirrors_the_command_tree() -> None:
    spec = build_spec(app)

    assert set(spec.children) == {
        "train",
        "evaluate",
        "report",
        "baseline",
        "uncertainty",
        "leakage-check",
    }
    assert set(spec.children["baseline"].children) == {"ibu", "omnifold"}


def test_group_level_options_sit_at_the_root() -> None:
    assert "log_level" in build_spec(app).options


def test_hyperparameters_are_layerable() -> None:
    train = build_spec(app).children["train"]

    for name in ("n_epochs", "batch_size", "lr_g", "lr_d", "n_layers", "hidden_units"):
        assert name in train.options, name
    assert train.options["n_epochs"] is int


def test_positional_arguments_are_not_layerable() -> None:
    report = build_spec(app).children["report"]

    assert "run_dir" not in report.options
    assert "run_dir" in report.excluded


def test_force_and_identity_options_are_not_layerable() -> None:
    spec = build_spec(app)

    assert "force" in spec.children["evaluate"].excluded
    assert "load_run" in spec.children["train"].excluded


def test_uncertainty_run_is_excluded_wholesale() -> None:
    """Design cells resolve from design.json, never from an ambient file."""
    uncertainty = build_spec(app).children["uncertainty"]

    assert "run" not in uncertainty.children
    assert "collect" in uncertainty.children
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_layered_config.py -k spec -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ran.config_spec'`.

- [ ] **Step 3: Write the implementation**

Typer records commands as `app.registered_commands` (each with `.name` and `.callback`) and sub-apps as `app.registered_groups` (each with `.name` and `.typer_instance`). A parameter is a positional Argument only when its `Annotated` metadata holds a `typer.models.ArgumentInfo`; a bare default such as `force: bool = False` is an Option.

```python
# src/ran/config_spec.py
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
    from collections.abc import Mapping
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
    ("uncertainty", "collect"): frozenset({"design_dir"}),
    ("uncertainty", "freeze"): frozenset({"design_dir", "force"}),
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
    if get_origin(annotation) is not Annotated:
        return False
    return any(isinstance(meta, ArgumentInfo) for meta in get_args(annotation)[1:])


def _base_type(annotation: Any, /) -> Any:
    """Strip `Annotated[...]` down to the declared type."""
    return get_args(annotation)[0] if get_origin(annotation) is Annotated else annotation


def _options_of(
    callback: Any, path: tuple[str, ...], /
) -> tuple[dict[str, Any], frozenset[str]]:
    """Split one command's parameters into layerable and excluded."""
    hints: dict[str, Any] = get_type_hints(callback, include_extras=True)
    denied: frozenset[str] = NOT_LAYERABLE.get(path, frozenset())

    options: dict[str, Any] = {}
    excluded: set[str] = set(denied)
    for name in inspect.signature(callback).parameters:
        if name in _CONTEXT_PARAMS:
            continue
        annotation: Any = hints.get(name, str)
        if _is_argument(annotation) or name in denied:
            excluded.add(name)
            continue
        options[name] = _base_type(annotation)
    return options, frozenset(excluded)


def build_spec(app: typer.Typer, path: tuple[str, ...] = ()) -> CommandSpec:
    """Describe `app` and everything registered under it."""
    options: dict[str, Any] = {}
    excluded: frozenset[str] = frozenset()
    if app.registered_callback is not None and app.registered_callback.callback:
        options, excluded = _options_of(app.registered_callback.callback, path)

    children: dict[str, CommandSpec] = {}
    for command in app.registered_commands:
        name: str = command.name or command.callback.__name__
        child_path: tuple[str, ...] = (*path, name)
        if child_path in FROZEN_COMMANDS:
            continue
        command_options, command_excluded = _options_of(command.callback, child_path)
        children[name] = CommandSpec(options=command_options, excluded=command_excluded)

    for group in app.registered_groups:
        if group.name is None or group.typer_instance is None:
            continue
        children[group.name] = build_spec(group.typer_instance, (*path, group.name))

    return CommandSpec(options=options, excluded=excluded, children=children)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_layered_config.py -v`
Expected: all PASS except `test_uncertainty_run_is_excluded_wholesale` and the `freeze` entries in `NOT_LAYERABLE`, which describe commands Task 7 adds. Until then `uncertainty.freeze` is simply absent from the tree, which is harmless — `NOT_LAYERABLE` keys for commands that do not exist are never consulted. `test_uncertainty_run_is_excluded_wholesale` passes now because `FROZEN_COMMANDS` skips `run` regardless.

- [ ] **Step 5: Commit**

```bash
git add src/ran/config_spec.py tests/test_layered_config.py
git commit -m "feat(config): derive the layerable option tree from the Typer app

Positional arguments, per-invocation identity options and --force are
excluded. uncertainty run is excluded wholesale: design cells resolve
from a frozen spec, not from ambient files.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Validate and merge

**Files:**
- Modify: `src/ran/config.py`
- Test: `tests/test_layered_config.py` (append)

**Interfaces:**
- Consumes: `Layer`, `ConfigError` (Task 1); `CommandSpec` (Task 2).
- Produces: `Resolved` (frozen dataclass, fields `values: dict[tuple[str, ...], dict[str, Any]]`, `origins: dict[tuple[str, ...], dict[str, str]]`, `layers: tuple[Layer, ...]`); `load(layers: tuple[Layer, ...], spec: CommandSpec) -> Resolved`; `default_map(resolved: Resolved) -> dict[str, Any]`.

Rules: a TOML **table** names a subcommand, a scalar names an option — so a name that is both is disambiguated by value type. Keys are normalized `-` to `_`. Merging across layers is per-key, with later layers winning.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_layered_config.py
from ran.config import Layer, default_map, load
from ran.config_spec import CommandSpec


def _spec() -> CommandSpec:
    return CommandSpec(
        options={"log_level": str},
        children={
            "train": CommandSpec(
                options={"n_epochs": int, "lr_g": float, "n_layers": int},
                excluded=frozenset({"load_run"}),
            ),
            "baseline": CommandSpec(
                children={"omnifold": CommandSpec(options={"n_epochs": int})}
            ),
        },
    )


def _layer(data: dict[str, object], origin: str = "ran.toml:/p/ran.toml") -> Layer:
    return Layer(origin=origin, data=data)


def test_keys_normalize_from_kebab_to_snake() -> None:
    resolved = load((_layer({"train": {"n-epochs": 500}}),), _spec())

    assert resolved.values[("train",)] == {"n_epochs": 500}


def test_underscores_are_accepted_too() -> None:
    resolved = load((_layer({"train": {"n_epochs": 500}}),), _spec())

    assert resolved.values[("train",)] == {"n_epochs": 500}


def test_nested_command_tables_resolve() -> None:
    resolved = load((_layer({"baseline": {"omnifold": {"n-epochs": 80}}}),), _spec())

    assert resolved.values[("baseline", "omnifold")] == {"n_epochs": 80}


def test_group_level_options_resolve_at_the_root() -> None:
    resolved = load((_layer({"log-level": "debug"}),), _spec())

    assert resolved.values[()] == {"log_level": "debug"}


def test_a_later_layer_overrides_per_key_not_wholesale() -> None:
    resolved = load(
        (
            _layer({"train": {"lr-g": 0.1, "n-epochs": 1}}, "ran.toml:/global"),
            _layer({"train": {"n-epochs": 500}}, "ran.toml:/project"),
        ),
        _spec(),
    )

    assert resolved.values[("train",)] == {"lr_g": 0.1, "n_epochs": 500}
    assert resolved.origins[("train",)]["lr_g"] == "ran.toml:/global"
    assert resolved.origins[("train",)]["n_epochs"] == "ran.toml:/project"


def test_default_map_is_nested_by_command_path() -> None:
    resolved = load(
        (_layer({"log-level": "debug", "baseline": {"omnifold": {"n-epochs": 80}}}),),
        _spec(),
    )

    assert default_map(resolved) == {
        "log_level": "debug",
        "baseline": {"omnifold": {"n_epochs": 80}},
    }


def test_an_unknown_key_suggests_the_near_miss() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load((_layer({"train": {"n-epoch": 500}}),), _spec())

    message = str(excinfo.value)
    assert "n-epoch" in message
    assert "n-epochs" in message
    assert "/p/ran.toml" in message


def test_an_unknown_table_suggests_the_near_miss() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load((_layer({"trian": {"n-epochs": 500}}),), _spec())

    assert "trian" in str(excinfo.value)
    assert "train" in str(excinfo.value)


def test_a_non_layerable_option_says_why() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load((_layer({"train": {"load-run": "runs/x"}}),), _spec())

    message = str(excinfo.value)
    assert "load-run" in message
    assert "cannot be configured" in message


def test_a_non_integral_float_for_an_int_option_is_rejected() -> None:
    """Click would silently truncate 3.7 to 3."""
    with pytest.raises(ConfigError) as excinfo:
        load((_layer({"train": {"n-layers": 3.7}}),), _spec())

    assert "n-layers" in str(excinfo.value)
    assert "3.7" in str(excinfo.value)


def test_an_integral_float_for_an_int_option_is_accepted() -> None:
    resolved = load((_layer({"train": {"n-layers": 3.0}}),), _spec())

    assert resolved.values[("train",)] == {"n_layers": 3}


def test_a_float_option_accepts_an_int() -> None:
    resolved = load((_layer({"train": {"lr-g": 1}}),), _spec())

    assert resolved.values[("train",)] == {"lr_g": 1}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_layered_config.py -v`
Expected: the new tests FAIL with `ImportError: cannot import name 'load' from 'ran.config'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ran/config.py`. Add `from difflib import get_close_matches` to the imports, and `from .config_spec import CommandSpec` under `if TYPE_CHECKING:` (it is needed only as an annotation, which keeps `config.py` a stdlib-only leaf at runtime).

```python
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
        _normalize(key), [_normalize(c) for c in candidates], n=1
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
            child: CommandSpec | None = spec.children.get(name)
            if child is None:
                raise ConfigError(
                    f"{where}: unknown command table `{_display(key)}`"
                    f"{_suggestion(key, spec.children)}"
                )
            _walk(value, child, (*path, name), origin, into)
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
```

Note the children-of-a-command-table name lookup uses the command name as written in the CLI (`leakage-check`), so `_normalize` must not be applied to command table names when looking them up. Handle this by trying the raw key first and the normalized key second:

```python
child: CommandSpec | None = spec.children.get(key) or spec.children.get(name)
```

Add `Iterable` to the `TYPE_CHECKING` imports from `collections.abc`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_layered_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Check complexity**

Run: `uv run just complexity`
Expected: PASS. `_walk` and `_check_option` are the two that could exceed 10; if either does, extract the error-raising branches into named helpers rather than raising the limit.

- [ ] **Step 6: Commit**

```bash
git add src/ran/config.py tests/test_layered_config.py
git commit -m "feat(config): validate and merge config layers

Kebab and snake keys both normalize. Tables name commands, scalars name
options. Unknown keys fail with a difflib near-miss; a non-integral
float for an int option fails rather than being silently truncated.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Wire it into the CLI

**Files:**
- Modify: `src/ran/cli.py:37-51` (the `configure` callback)
- Test: `tests/test_config_cli.py` (create)

**Interfaces:**
- Consumes: `discover`, `load`, `default_map`, `ConfigError` (Tasks 1, 3); `build_spec` (Task 2).
- Produces: a `configure(ctx: typer.Context, log_level: ...)` callback that assigns `ctx.default_map`; `ran.cli.SPEC` (a cached `CommandSpec`).

This is where the whole stack becomes observable. After this task, `ran train --help` inside a directory with a `ran.toml` reports the configured value as the default.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config_cli.py
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ran.cli import app
from typer.testing import CliRunner

if TYPE_CHECKING:
    from pathlib import Path

runner: CliRunner = CliRunner()


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A git-root-bounded directory that is the process's cwd, with no global layer."""
    (tmp_path / ".git").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "no-global"))
    monkeypatch.delenv("RAN_LOG_LEVEL", raising=False)
    return tmp_path


def test_help_reports_the_configured_default(project: Path) -> None:
    (project / "ran.toml").write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "500" in result.stdout


def test_help_reports_the_code_default_without_config(project: Path) -> None:
    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "100" in result.stdout


def test_a_malformed_config_fails_the_command(project: Path) -> None:
    (project / "ran.toml").write_text("[train\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code != 0


def test_an_unknown_key_fails_the_command(project: Path) -> None:
    (project / "ran.toml").write_text("[train]\nn-epoch = 500\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code != 0


def test_a_pyproject_tool_ran_table_is_read(project: Path) -> None:
    (project / "pyproject.toml").write_text("[tool.ran.train]\nn-epochs = 321\n")

    result = runner.invoke(app, ["train", "--help"])

    assert result.exit_code == 0
    assert "321" in result.stdout


def test_an_out_of_range_config_value_is_rejected_by_click(project: Path) -> None:
    (project / "ran.toml").write_text("[train]\nn-epochs = -5\n")

    result = runner.invoke(app, ["train"])

    assert result.exit_code != 0
    assert "n-epochs" in result.output
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_cli.py -v`
Expected: `test_help_reports_the_configured_default` FAILS (help shows 100, not 500); the error-path tests FAIL with exit code 0.

- [ ] **Step 3: Write the implementation**

Replace `configure` at `src/ran/cli.py:37-51`:

```python
@app.callback()
def configure(
    ctx: typer.Context,
    log_level: Annotated[
        LogLevel,
        typer.Option(
            "--log-level",
            "-L",
            case_sensitive=False,
            envvar="RAN_LOG_LEVEL",
            help="Application log level.",
        ),
    ] = LogLevel.info,
) -> None:
    # Assigned here rather than through `Typer(context_settings=)` so that
    # discovery is lazy: it walks the filesystem once, at invocation, not at
    # import. `rantypes/constants.py` resolving `RAN_CACHE_DIR` at import is the
    # failure mode this avoids.
    #
    # Click then applies its own precedence to what we hand it:
    #   COMMANDLINE > ENVIRONMENT > DEFAULT_MAP > DEFAULT
    # which is layers 5, 4, 3-2, 1. It also casts these values through each
    # parameter's declared type and enforces each parameter's constraints, so
    # `n-epochs = -5` from a file is rejected by the existing `min=1`.
    ctx.default_map = _resolved_default_map(ctx)
    configure_logging(level=log_level.value)
```

Add above it, after the app definitions:

```python
@cache
def _spec() -> CommandSpec:
    """Built once per process, after every command has registered itself."""
    return build_spec(app)


def _resolved_default_map(ctx: typer.Context, /) -> dict[str, Any]:
    """The merged file layers, or an empty map if `config show` will explain why.

    A broken config file must not disable the one command whose job is to
    diagnose it, so `ran config show` proceeds with no defaults and reports the
    error itself. Every other command fails here.
    """
    try:
        return default_map(load(discover(Path.cwd(), os.environ), _spec()))
    except ConfigError as error:
        if ctx.invoked_subcommand == "config":
            return {}
        raise typer.BadParameter(str(object=error)) from error
```

New imports at the top of `cli.py`:

```python
import os
from functools import cache
from typing import TYPE_CHECKING, Annotated

from .config import ConfigError, default_map, discover, load
from .config_spec import build_spec

if TYPE_CHECKING:
    from typing import Any

    from .config_spec import CommandSpec
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Confirm nothing else regressed**

Run: `uv run just test-fast`
Expected: PASS. `tests/test_cli.py` exercises every command's `--help`; if any fails, the callback is raising where it should not.

- [ ] **Step 6: Commit**

```bash
git add src/ran/cli.py tests/test_config_cli.py
git commit -m "feat(config): resolve config layers into Click's default_map

Assigned in the group callback so discovery stays lazy. Click supplies
precedence, type coercion and constraint validation; --help now reports
the effective default rather than the code default.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: `ran config show`

**Files:**
- Create: `src/ran/config_show.py`
- Modify: `src/ran/cli.py` (register a `config` sub-app)
- Modify: `tests/test_cli.py:25-33` (the exact command-tree assertion)
- Test: `tests/test_config_cli.py` (append)

**Interfaces:**
- Consumes: `Resolved`, `discover`, `load`, `ConfigError` (Tasks 1, 3); `build_spec` (Task 2).
- Produces: `render(resolved: Resolved | None, error: ConfigError | None, environ: Mapping[str, str], *, console: Console | None = None, command: str | None = None) -> None`.

`tests/test_cli.py:25-33` asserts the command tree exactly and **will fail** until `"config"` is added to the expected set. That is intentional — it is the repo's guard against commands appearing unannounced.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_config_cli.py
def test_config_show_lists_values_and_origins(project: Path) -> None:
    (project / "ran.toml").write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "train.n-epochs" in result.stdout
    assert "500" in result.stdout
    assert "ran.toml" in result.stdout


def test_config_show_reports_the_environment_only_settings(project: Path) -> None:
    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code == 0
    assert "RAN_CACHE_DIR" in result.stdout
    assert "RAN_TIMING" in result.stdout


def test_config_show_explains_a_broken_config_instead_of_dying(project: Path) -> None:
    """The command that diagnoses a bad file must survive a bad file."""
    (project / "ran.toml").write_text("[train\n")

    result = runner.invoke(app, ["config", "show"])

    assert result.exit_code != 0
    assert "invalid TOML" in result.output


def test_config_show_scopes_to_one_command(project: Path) -> None:
    (project / "ran.toml").write_text(
        "[train]\nn-epochs = 500\n\n[uncertainty.collect]\nn-bins = 40\n"
    )

    result = runner.invoke(app, ["config", "show", "train"])

    assert result.exit_code == 0
    assert "train.n-epochs" in result.stdout
    assert "n-bins" not in result.stdout


def test_config_show_json_is_machine_readable(project: Path) -> None:
    import json

    (project / "ran.toml").write_text("[train]\nn-epochs = 500\n")

    result = runner.invoke(app, ["config", "show", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["values"]["train"]["n-epochs"] == 500
    assert "ran.toml" in payload["origins"]["train"]["n-epochs"]
```

Update the tree assertion in `tests/test_cli.py`:

```python
    assert _command_names(app) == {
        "train",
        "evaluate",
        "report",
        "baseline",
        "uncertainty",
        "leakage-check",
        "config",
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_cli.py tests/test_cli.py -v`
Expected: the `config show` tests FAIL with exit code 2 ("No such command"); `test_registered_command_trees_are_exact` FAILS on the missing `"config"`.

- [ ] **Step 3: Write the implementation**

```python
# src/ran/config_show.py
"""Render the resolved configuration stack and where each value came from.

`rich` with an injectable `Console`, matching `uncertainty/report.py`; `print`
is banned in `src/` by `tests/test_source_hygiene.py`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from .rantypes.constants import CACHE_ENV_VAR

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .config import ConfigError, Resolved

# Resolved at import in their own modules, so they cannot join the layered
# stack. See the spec's "Deferred" section.
_ENV_ONLY: tuple[tuple[str, str], ...] = (
    (CACHE_ENV_VAR, ".cache"),
    ("RAN_TIMING", "off"),
)


def _dotted(path: tuple[str, ...], key: str, /) -> str:
    return ".".join((*path, key.replace("_", "-")))


def _payload(resolved: Resolved, /) -> dict[str, dict[str, dict[str, object]]]:
    """The same content as the table, shaped for `--json`."""
    def _nest(source: Mapping[tuple[str, ...], Mapping[str, object]]) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for path, entries in source.items():
            label: str = ".".join(path) or "."
            out[label] = {key.replace("_", "-"): value for key, value in entries.items()}
        return out

    return {"values": _nest(resolved.values), "origins": _nest(resolved.origins)}


def _layers_table(resolved: Resolved, /) -> Table:
    table = Table(title="Layers (nearest last)", show_header=False, box=None)
    if not resolved.layers:
        table.add_row("(no configuration files found)")
    for layer in resolved.layers:
        kind, _, path = layer.origin.partition(":")
        table.add_row(path, kind)
    return table


def _values_table(resolved: Resolved, command: str | None, /) -> Table:
    table = Table(show_header=True, box=None)
    table.add_column("setting")
    table.add_column("value")
    table.add_column("origin")
    for path in sorted(resolved.values):
        if command is not None and (not path or path[0] != command):
            continue
        for key, value in sorted(resolved.values[path].items()):
            table.add_row(
                _dotted(path, key), str(object=value), resolved.origins[path][key]
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
    """Print the stack, as a table or as JSON."""
    active: Console = console or Console()
    if as_json:
        active.print_json(json.dumps(obj=_payload(resolved), default=str))
        return
    active.print(_layers_table(resolved))
    active.print(_values_table(resolved, command))
    active.print(_environment_table(environ))
```

In `cli.py`, register the sub-app next to the others:

```python
config_app: typer.Typer = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
app.add_typer(
    typer_instance=config_app,
    name="config",
    help="Inspect the resolved configuration stack.",
)


@config_app.command(name="show")
def config_show_command(
    command: Annotated[
        str | None, typer.Argument(help="Limit the listing to one command.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
) -> None:
    """Show every resolved setting and the file, variable or default it came from."""
    from .config_show import render

    try:
        resolved = load(discover(Path.cwd(), os.environ), _spec())
    except ConfigError as error:
        raise typer.BadParameter(str(object=error)) from error
    render(resolved, os.environ, command=command, as_json=as_json)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_cli.py tests/test_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ran/config_show.py src/ran/cli.py tests/test_config_cli.py tests/test_cli.py
git commit -m "feat(config): add ran config show

Lists the files consulted, every resolved setting with its origin, and
the two environment-only settings. Survives a broken config file, since
it is the command that diagnoses one.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Record provenance in `config.json`

**Files:**
- Modify: `src/ran/workflows/train.py:250-264`
- Modify: `src/ran/cli.py` (`train_command` passes origins through)
- Test: `tests/test_config_cli.py` (append)

**Interfaces:**
- Consumes: `Resolved` (Task 3); `ctx.get_parameter_source` from `typer.Context`.
- Produces: `origins_for(ctx: typer.Context, resolved: Resolved, path: tuple[str, ...]) -> dict[str, str]` in `src/ran/config.py`; `run(..., origins: dict[str, str] | None = None)` in `workflows/train.py`.

`origins` is keyword-only with a `None` default so that `workflows.run()`'s existing positional signature — and every caller in `benchmarks/` — is unaffected.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_config_cli.py
from ran.config import Layer, Resolved, origins_for


def test_origins_for_prefers_the_click_source_over_the_file() -> None:
    """A flag beats a file, and the recorded origin has to say so."""

    class _Ctx:
        def __init__(self) -> None:
            self._sources = {"n_epochs": "COMMANDLINE", "lr_g": "DEFAULT_MAP", "n_layers": "DEFAULT"}

        def get_parameter_source(self, name: str) -> object:
            return type("S", (), {"name": self._sources[name]})()

    resolved = Resolved(
        values={("train",): {"lr_g": 0.1}},
        origins={("train",): {"lr_g": "ran.toml:/p/ran.toml"}},
        layers=(Layer(origin="ran.toml:/p/ran.toml", data={}),),
    )

    origins = origins_for(_Ctx(), resolved, ("train",), names=("n_epochs", "lr_g", "n_layers"))

    assert origins == {
        "n_epochs": "command-line",
        "lr_g": "ran.toml:/p/ran.toml",
        "n_layers": "default",
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_config_cli.py::test_origins_for_prefers_the_click_source_over_the_file -v`
Expected: FAIL with `ImportError: cannot import name 'origins_for'`.

- [ ] **Step 3: Write the implementation**

Append to `src/ran/config.py`:

```python
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
```

In `workflows/train.py`, add `origins: dict[str, str] | None = None` as a keyword-only parameter to both `run()` and `_save_run()` (whichever builds `config_out`), and extend the dict at line 250:

```python
    config_out: dict[str, Any] = {
        "batch_size": batch_size,
        "n_samples": n_samples,
        "dim": dim,
        "dataset": dataset,
        "seed": init_seed,
        "data_seed": data_seed,
        **hyperparameters,
    }
    if origins:
        # Leading underscore: metadata *about* the run, not a parameter *of*
        # it. `baselines/_shared.py:parse_run_config` reads known keys by name
        # and keeps the rest in `source`, so older readers are unaffected and
        # older run dirs simply have no `_origin`.
        config_out["_origin"] = origins
```

In `cli.py`, `train_command` gains `ctx: typer.Context` and passes the origins:

```python
    resolved = load(discover(Path.cwd(), os.environ), _spec())
    run(
        ...,
        origins=origins_for(
            ctx, resolved, ("train",), names=sorted(_spec().children["train"].options)
        ),
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_cli.py -v`
Expected: all PASS.

- [ ] **Step 5: Confirm existing run-config readers still work**

Run: `uv run pytest tests/test_workflow.py tests/test_ibu.py tests/test_report.py -v`
Expected: PASS. These parse `config.json`; an unknown `_origin` key must not disturb them.

- [ ] **Step 6: Commit**

```bash
git add src/ran/config.py src/ran/workflows/train.py src/ran/cli.py tests/test_config_cli.py
git commit -m "feat(config): record per-key provenance in config.json

An _origin block names the file, variable or default behind every
recorded hyperparameter, so a run says where its settings came from.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: `uncertainty freeze`, and frozen resolution in cells

**Files:**
- Modify: `src/ran/uncertainty/design.py`
- Modify: `src/ran/cli.py` (`uncertainty freeze`; `uncertainty_run_command`)
- Modify: `src/ran/uncertainty/__init__.py` (export the two new functions)
- Modify: `tests/test_cli.py` (the `uncertainty_app` tree assertion)
- Modify: `scripts/submit_uncertainty.zsh:65-70`
- Test: `tests/test_config_cli.py` (append)

**Interfaces:**
- Consumes: `origins_for` (Task 6), `_spec` (Task 4).
- Produces: in `design.py`, `freeze_design(design_dir: Path, values: dict[str, Any], origins: dict[str, str], *, force: bool = False) -> Path` and `load_frozen(design_dir: Path) -> dict[str, Any]`.

`FROZEN_COMMANDS` in Task 2 already keeps `uncertainty run` out of the layer tree, so no ambient file can reach it. This task supplies what it reads instead.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_config_cli.py
def test_uncertainty_run_without_a_frozen_design_names_freeze(project: Path) -> None:
    design = project / "design"
    design.mkdir()

    result = runner.invoke(
        app, ["uncertainty", "run", "--cell", "0", "--design-dir", str(design)]
    )

    assert result.exit_code != 0
    assert "freeze" in result.output


def test_freeze_writes_the_resolved_values(project: Path) -> None:
    (project / "ran.toml").write_text("[uncertainty.freeze]\nn-epochs = 7\n")
    design = project / "design"

    result = runner.invoke(app, ["uncertainty", "freeze", "--design-dir", str(design)])

    assert result.exit_code == 0
    import json

    frozen = json.loads((design / "design.json").read_text())
    assert frozen["config"]["n_epochs"] == 7
    assert "ran.toml" in frozen["_origin"]["n_epochs"]


def test_freeze_refuses_to_overwrite_without_force(project: Path) -> None:
    design = project / "design"
    assert runner.invoke(app, ["uncertainty", "freeze", "-d", str(design)]).exit_code == 0

    result = runner.invoke(app, ["uncertainty", "freeze", "-d", str(design)])

    assert result.exit_code != 0
    assert "--force" in result.output


def test_freeze_overwrites_with_force(project: Path) -> None:
    design = project / "design"
    assert runner.invoke(app, ["uncertainty", "freeze", "-d", str(design)]).exit_code == 0

    result = runner.invoke(app, ["uncertainty", "freeze", "-d", str(design), "--force"])

    assert result.exit_code == 0


def test_a_config_edit_after_freeze_does_not_reach_a_cell(project: Path) -> None:
    """The regression test for the property the freeze path exists to protect."""
    import json

    (project / "ran.toml").write_text("[uncertainty.freeze]\nn-epochs = 7\n")
    design = project / "design"
    assert runner.invoke(app, ["uncertainty", "freeze", "-d", str(design)]).exit_code == 0

    (project / "ran.toml").write_text("[uncertainty.freeze]\nn-epochs = 999\n")

    frozen = json.loads((design / "design.json").read_text())
    assert frozen["config"]["n_epochs"] == 7
```

Update the sub-app assertion in `tests/test_cli.py`:

```python
    assert _command_names(uncertainty_app) == {"run", "collect", "freeze"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config_cli.py tests/test_cli.py -v`
Expected: the freeze tests FAIL with "No such command 'freeze'".

- [ ] **Step 3: Write the implementation**

Append to `src/ran/uncertainty/design.py`:

```python
FROZEN_NAME: Final[LiteralString] = "design.json"


def freeze_design(
    design_dir: Path,
    values: dict[str, Any],
    origins: dict[str, str],
    *,
    force: bool = False,
) -> Path:
    """Write the settings every cell of this design will use.

    Refuses to overwrite: a design whose cells were trained under different
    settings is not a variance decomposition, and rewriting this file while an
    array is in flight is exactly how that happens.
    """
    design_dir.mkdir(parents=True, exist_ok=True)
    path: Path = design_dir / FROZEN_NAME
    if path.exists() and not force:
        raise FileExistsError(
            f"{path} already exists; pass --force to overwrite it, but not while "
            f"an array is running"
        )
    _ = path.write_text(
        data=json.dumps(obj={"config": values, "_origin": origins}, indent=2)
    )
    return path


def load_frozen(design_dir: Path) -> dict[str, Any]:
    """The frozen settings for this design."""
    path: Path = design_dir / FROZEN_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `ran uncertainty freeze --design-dir "
            f"{design_dir}` once before submitting the array"
        )
    frozen: dict[str, Any] = json.loads(s=path.read_text())
    return frozen["config"]
```

Add `Final`, `LiteralString` and `Any` to the module's `TYPE_CHECKING` imports, and export both names from `src/ran/uncertainty/__init__.py`.

In `cli.py`, add the `freeze` command. Its options are `uncertainty run`'s minus `cell`:

```python
@uncertainty_app.command(name="freeze")
def uncertainty_freeze_command(
    ctx: typer.Context,
    design_dir: Annotated[Path, typer.Option("--design-dir", "-d")],
    force: Annotated[bool, typer.Option("--force")] = False,
    n_datasets: Annotated[int, typer.Option("--n-datasets", "-B", min=2)] = 8,
    n_seeds: Annotated[int, typer.Option("--n-seeds", "-S", min=2)] = 8,
    n_eval: Annotated[int, typer.Option(min=1)] = 100_000,
    dataset: Annotated[DatasetName, typer.Option("--dataset", "-D")] = DatasetName.jets,
    variable: Annotated[list[str] | None, typer.Option("--var", "-v")] = None,
    config: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", min=1)] = 1024,
    n_samples: Annotated[int, typer.Option("--n-samples", "-n", min=1)] = 500_000,
    hidden_units: Annotated[int, typer.Option("--hidden-units", "-u", min=1)] = 64,
    n_layers: Annotated[int, typer.Option("--n-layers", "-l", min=1)] = 2,
    n_epochs: Annotated[int, typer.Option("--n-epochs", "-e", min=1)] = 100,
    n_disc_steps: Annotated[int, typer.Option("--n-disc-steps", "-k", min=1)] = 5,
    lr_g: Annotated[float, typer.Option("--lr-g", min=0.0)] = 3e-5,
    lr_d: Annotated[float, typer.Option("--lr-d", min=0.0)] = 1e-4,
    lambda_dispersion: Annotated[
        float, typer.Option("--lambda-dispersion", min=0.0)
    ] = 0.015,
    data_seed: Annotated[int, typer.Option()] = 42,
    init_seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Fix the settings for a variance design before its cells are submitted.

    Run once, on the login node, between creating the design directory and
    submitting the array. Cells read this file instead of the config layers, so
    editing `ran.toml` mid-array cannot split a design.
    """
    from .uncertainty import freeze_design

    names: tuple[str, ...] = tuple(sorted(_spec().children["uncertainty"].children["freeze"].options))
    values: dict[str, Any] = {name: ctx.params[name] for name in names}
    values["variable"] = list(_canonical_variables(variable))
    values["dataset"] = dataset.value
    if config is not None:
        values["config"] = str(object=config)

    resolved = load(discover(Path.cwd(), os.environ), _spec())
    path = freeze_design(
        design_dir,
        values,
        origins_for(ctx, resolved, ("uncertainty", "freeze"), names=names),
        force=force,
    )
    logger.info("Froze design settings to %s", path)
```

`uncertainty_run_command` gains `ctx` and overlays the frozen values over anything not typed on the command line:

```python
@uncertainty_app.command(name="run")
def uncertainty_run_command(
    ctx: typer.Context,
    cell: Annotated[int, typer.Option("--cell", "-c", min=0)],
    design_dir: Annotated[Path, typer.Option("--design-dir", "-d")],
    # ... every other option unchanged ...
) -> None:
    """Train one (bootstrap dataset, init seed) cell of the design.

    Settings come from the design's frozen `design.json`, not from the config
    layers: an edited `ran.toml` must not be able to change what cell 30 of a
    64-cell array measures. An explicit flag still wins.
    """
    from .uncertainty import DesignSpec, load_frozen, run_cell

    frozen: dict[str, Any] = load_frozen(design_dir)
    settings: dict[str, Any] = {
        name: (
            ctx.params[name]
            if ctx.get_parameter_source(name).name == "COMMANDLINE"
            else frozen.get(name, ctx.params[name])
        )
        for name in frozen
        if name in ctx.params
    }
    ...
```

Then build `DesignSpec` and the `run_cell` call from `settings` rather than from the bare parameters. `FileNotFoundError` from `load_frozen` surfaces as a non-zero exit with its message; wrap it in `typer.BadParameter` for a clean one.

Finally, in `scripts/submit_uncertainty.zsh`, after `mkdir -p "${DESIGN_DIR}"` (line 66):

```zsh
# Fix the design's settings once, here on the login node, before any cell
# exists. Cells read this file rather than the config layers, so an edit to
# ran.toml while the array is in flight cannot split the design.
uv run ran uncertainty freeze --design-dir "${DESIGN_DIR}" ${RUN_ARGS} \
    -B "${B}" -S "${S}" --n-eval "${N_EVAL}"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config_cli.py tests/test_cli.py tests/test_uncertainty.py -v`
Expected: all PASS.

- [ ] **Step 5: Check the submit script still parses**

Run: `uv run pytest tests/test_scripts.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ran/uncertainty/ src/ran/cli.py tests/ scripts/submit_uncertainty.zsh
git commit -m "feat(uncertainty): freeze design settings before submitting cells

ran uncertainty freeze writes design.json once on the login node; cells
resolve from it rather than from the config layers, so editing ran.toml
mid-array cannot split a variance decomposition. An explicit flag still
wins. A cell with no frozen design errors instead of falling back.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 8: Documentation

**Files:**
- Create: `notes/agent/configuration.md`
- Modify: `CLAUDE.md` (Reference Docs table, Quick Start)
- Modify: `notes/agent/caching.md` (after line 44)
- Modify: `notes/agent/uncertainty.md`, `notes/agent/cli-and-running.md`
- Modify: `CHANGELOG.md`

`CLAUDE.md` must stay near 100 lines — add one table row and at most one Quick Start line, and put the substance in `configuration.md`.

- [ ] **Step 1: Write `notes/agent/configuration.md`**

Cover, with no more than a short paragraph each: the five-layer table from the spec; the discovery rule with the worked directory diagram; the schema including the `ran.toml`-versus-`pyproject.toml` prefix difference and the per-command-table rationale; the non-layerable denylist and why each entry is on it; `ran config show`; the freeze path; and why `cache-dir` and `timing` are environment-only.

- [ ] **Step 2: Add the Reference Docs row to `CLAUDE.md`**

```markdown
| [notes/agent/configuration.md](notes/agent/configuration.md)     | The five config layers, discovery order, `ran config show`, the design freeze |
```

- [ ] **Step 3: Extend `notes/agent/caching.md`**

After the existing environment-only discussion at lines 36-44, record that `RAN_CACHE_DIR` and `RAN_TIMING` stayed outside the layered stack because both resolve at import — `constants.py:35` and `timing.py:122` — and are bound into default arguments at `data/jets.py:62`, `data/download.py:272` and `data/datasets.py:153` before any Typer callback runs.

- [ ] **Step 4: Update the two CLI-facing docs**

`cli-and-running.md`: add `ran config show` and `ran uncertainty freeze` to the reference. `uncertainty.md`: add the freeze step, noting it runs once on the login node before `sbatch`.

- [ ] **Step 5: Add a CHANGELOG entry**

Follow the existing format in `CHANGELOG.md`. Note the breaking change: `ran uncertainty run` now requires a frozen `design.json`.

- [ ] **Step 6: Full validation**

Run: `uv run just validate`
Expected: format, lint, types, complexity and the full test suite all PASS.

- [ ] **Step 7: Commit**

```bash
git add docs/ CLAUDE.md CHANGELOG.md
git commit -m "docs: document the layered configuration stack

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage.** Every section maps to a task: layering table and discovery rule (Task 1), schema and layerable set (Tasks 2-3), architecture and the `default_map` wiring (Task 4), error handling (Task 3 for file-level cases, Task 4 for the CLI surface), `ran config show` (Task 5), `config.json` provenance (Task 6), the freeze path (Task 7), documentation including the deferred-settings note (Task 8). The "Deferred" section is a non-goal and correctly has no implementation task, only the Task 8 doc entry.

**Known deviations from the spec, both deliberate:**

1. The spec describes one new module, `src/ran/config.py`. This plan splits it into `config.py` (stdlib-only leaf) and `config_spec.py` (needs Typer), so the discovery and merge logic stays testable without importing the CLI. `config_show.py` is a third file for the same reason — it needs `rich`.
2. The spec says `freeze` takes "the same options as run". This plan has `freeze` own the layerable options and `run` own only `--cell`, `--design-dir` and the overrides, since `uncertainty run` is excluded from the layer tree. Configuration is therefore written under `[tool.ran.uncertainty.freeze]`, not `[...uncertainty.run]`; a file using the latter gets the Task 3 unknown-table error.

**Ordering note.** Tasks 1-4 are strictly sequential. Tasks 5, 6 and 7 all depend on Task 4 but not on each other, so they may be done in any order or in parallel. Task 8 is last. Two tests in `tests/test_cli.py` assert the command tree exactly and will fail by design until Task 5 and Task 7 update them.
