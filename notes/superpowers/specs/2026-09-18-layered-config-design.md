# Layered configuration resolution

**Date:** 2026-09-18
**Status:** Approved, ready for implementation planning

## Problem

`deconvolve` has no configuration layer. Every setting is a Typer default hard-coded in
a function signature, and three environment variables were each bolted on
separately at a different place and a different time:

- `cli.py:77-122` fixes `batch_size=1024`, `n_samples=500_000`, `n_epochs=100`,
  `lr_g=3e-5`, `lr_d=1e-4`, `lambda_dispersion=0.015`, `hidden_units=64`,
  `n_layers=2`. `cli.py:182-205` repeats all eight for `uncertainty run`.
- `DECONVOLVE_CACHE_DIR` is read at import of `coretypes/constants.py:35`.
- `DECONVOLVE_TIMING` is read at import of `instrumentation/timing.py:122`.
- `DECONVOLVE_LOG_LEVEL` is read by Typer's `envvar=` at `cli.py:47`.

The consequence is that a machine-level preference has nowhere to live. Running
on Perlmutter versus a laptop means retyping `-n1600000 -l3 -u128` on every
invocation, or wrapping `deconvolve` in a shell alias, or --- as
`scripts/submit_uncertainty.zsh:63` actually does --- freezing the flags into a
`RUN_ARGS` string inside a submit script, where they drift out of step with
`scripts/submit.zsh`. The script says so in a comment: "this tracks
`scripts/submit.zsh` and ... the two must be changed together."

The output of this work is a five-layer configuration stack matching the
convention used by uv, ruff and pyrefly, so that per-machine and per-project
defaults live in a file rather than in a person's shell history.

## The layering

Lowest precedence first. Each layer overrides all layers above it.

| #   | Layer          | Location                                                          |
| --- | -------------- | ----------------------------------------------------------------- |
| 1   | Code default   | the Typer signature in `cli.py`                                   |
| 2   | Global config  | `$XDG_CONFIG_HOME/deconvolve/deconvolve.toml`, default `~/.config/deconvolve/deconvolve.toml` |
| 3   | Project config | nearest `deconvolve.toml`, or `[tool.deconvolve]` in `pyproject.toml`           |
| 4   | Environment    | `DECONVOLVE_*`, via Typer's existing `envvar=`                           |
| 5   | Command line   | `--n-epochs 500`                                                  |

### Discovery rule for layer 3

Walk upward from the current working directory. The **first** directory
containing either a `deconvolve.toml` or a `pyproject.toml` with a `[tool.deconvolve]` table
wins outright, and the walk stops there. Within that one directory, `deconvolve.toml`
shadows `pyproject.toml` entirely --- they are not merged with each other.

The walk never passes the enclosing git repository root, located by walking for
a `.git` entry (file or directory, so worktrees and submodules behave). Outside
any repository the walk terminates at the filesystem root.

```
~/work/          deconvolve.toml            ignored, walk already stopped
  repo/    .git/
    a/          pyproject.toml       [tool.deconvolve]  <- wins, walk stops
      b/        (cwd)
```

Exactly one project-level file ever contributes, so "which file did this value
come from" has exactly one answer.

## Non-goals

- **Layering the physics spec.** `params/*.yaml` holds the Gaussian definition
  (`mu_gen`, `sigma_detector`, ...) parsed by `data/config.py:137`. It remains
  reachable only through an explicit `--config` path. An experiment's physics
  must be named on the command line, never inherited from an ambient file.
- **Layering `cache-dir` and `timing`.** See "Deferred" below.
- **Writing config files.** Discovery is read-only. `deconvolve` never creates a
  skeleton `deconvolve.toml`, and `deconvolve config show` never writes.
- **Merging multiple project files.** Rejected in favour of nearest-wins; see
  "Alternatives considered".
- **A migration path for existing run directories.** `config.json` gains an
  `_origin` block; older run dirs simply lack it, and every reader treats it as
  optional.

## Schema

The TOML mirrors the command tree, because Click's `default_map` is keyed by
command path.

```toml
# deconvolve.toml --- the same tables, without the `tool.` prefix
[train]
n-epochs = 500
batch-size = 2048

# pyproject.toml
[tool.deconvolve]
log-level = "debug"          # a group-level option

[tool.deconvolve.train]
n-epochs = 500
lr-g = 3e-5

[tool.deconvolve.baseline.omnifold]
n-epochs = 80

[tool.deconvolve.uncertainty.collect]
n-bins = 40
```

### Per-command tables, not one flat namespace

`n_epochs` already denotes three unrelated quantities in `cli.py`: 100 for
`train`, 50 for `baseline omnifold`, 100 for `uncertainty run`. `batch_size` is
1024 in two commands and 512 in a third. A flat `n-epochs = 500` would silently
retune the OmniFold baseline while the author believed they were configuring
training, which corrupts a comparison rather than a convenience.

Only genuinely global options --- those declared on the `@app.callback()`
itself, today just `log-level` --- sit at the top level of `[tool.deconvolve]`.

In `deconvolve.toml` the `tool.deconvolve` prefix is dropped: a top-level **key** is a
group-level option and a top-level **table** is a command. `[train]` in
`deconvolve.toml` and `[tool.deconvolve.train]` in `pyproject.toml` denote the same thing.

### Key spelling

TOML keys are kebab-case (`n-epochs`), matching uv and ruff. Underscores are
accepted and normalized, since the Python parameter name is `n_epochs` and
readers will reach for both. Normalization is `-` to `_` on lookup.

### What is layerable

Layerable: every Typer **Option** not on the denylist below.

Not layerable:

- Every positional **Argument** (`report`'s `run_dir`).
- `cell` and `design_dir` on `uncertainty run` --- per-job identity, and the
  whole point of a SLURM array is that they differ per invocation.
- `load_run` on `train` --- names a specific prior run.
- `force` on every command that has one --- a destructive-rebuild toggle should
  be typed each time, not inherited.

A key naming a non-layerable option is an error, with a message saying why
rather than merely "unknown key".

## Architecture

### `src/ran/config.py` (new)

A leaf module. It imports `tomllib`, `pathlib`, `os` and `difflib` and nothing
from `deconvolve`, so it stays cheap to import and unit-testable without pulling in
JAX or Keras.

```python
def discover(cwd: Path, environ: Mapping[str, str]) -> tuple[Layer, ...]:
    """The global file then the project file, each with its provenance."""

def load(layers: tuple[Layer, ...], spec: CommandSpec) -> Resolved:
    """Parse, validate against the command tree, merge. Raises ConfigError."""

def default_map(resolved: Resolved) -> dict[str, Any]:
    """Nested by command path, ready for `ctx.default_map`."""
```

`CommandSpec` is derived from the live Typer app --- the command names, their
option names, and the denylist --- so a new command or flag is validated
without anyone updating a parallel table.

`Resolved` carries both the merged values and a per-key origin string
(`"deconvolve.toml:/Users/krish/Developer/deconvolve/deconvolve.toml"`, `"default"`), which feeds
both `deconvolve config show` and the `_origin` block in `config.json`.

### `src/deconvolve/cli.py` (modified)

The existing `configure()` callback gains a `ctx: typer.Context` parameter and
assigns `ctx.default_map`. That is the entire integration:

```python
@app.callback()
def configure(ctx: typer.Context, log_level: ... = LogLevel.info) -> None:
    ctx.default_map = config.default_map(config.load(config.discover(Path.cwd(), os.environ), SPEC))
    configure_logging(level=log_level.value)
```

Assigning inside the callback --- rather than through `Typer(context_settings=)`
--- keeps filesystem discovery **lazy**. It happens once per invocation, at
invocation, not at import. This matters: `coretypes/constants.py:35` already
demonstrates the failure mode of resolving configuration at import time.

No other command signature changes. `workflows.run()` and
`uncertainty.run_cell()` keep their current signatures and remain unaware that
configuration layering exists.

### Why Click's `default_map`

Click's native precedence is `COMMANDLINE > ENVIRONMENT > DEFAULT_MAP >
DEFAULT`, which is layers 5, 4, 3-2, 1 exactly. Verified on the pinned
`typer 0.27.2`:

| Input                          | Resolved | `get_parameter_source()` |
| ------------------------------ | -------- | ------------------------ |
| nothing                        | 999      | `DEFAULT_MAP`            |
| `DECONVOLVE_N_EPOCHS=50`              | 50       | `ENVIRONMENT`            |
| `DECONVOLVE_N_EPOCHS=50 --n-epochs 7` | 7        | `COMMANDLINE`            |
| key absent from map            | 2        | `DEFAULT`                |

Three properties fall out of this that are worth naming, because they are the
reason this approach was chosen over writing a resolver:

- **`--help` reports the effective default.** Inside a project whose
  `deconvolve.toml` sets `n-epochs = 500`, `deconvolve train --help` prints
  `[default: 500]`. The configuration documents itself, and a sentinel-based
  resolver would have destroyed this.
- **Type coercion is free.** Click casts `default_map` values through each
  parameter's own type: `"999"` to `999`, `"debug"` to `LogLevel.debug`,
  `"runs/x"` to `PosixPath("runs/x")`, `"no"` to `False`.
- **Constraint validation is free.** `n-epochs = -5` from a config file is
  rejected by the existing `min=1` with Click's own error message,
  `-5 is not in the range x>=1`.

Typer 0.27 vendors Click privately as `typer._click` rather than depending on
the published package. This design touches **no private API**: `default_map`
and `get_parameter_source` are both reached through the public
`typer.Context`.

## The uncertainty freeze path

A variance design is a SLURM array of B x S independent `deconvolve uncertainty run`
invocations, all on a shared filesystem, all reading the same working
directory. Layered configuration would let an edit to `deconvolve.toml` at cell 30
change what cells 30-63 measure, producing a design whose cells silently
disagree. The bootstrap-versus-seed decomposition is then meaningless, and
nothing in the output would reveal it.

So `uncertainty run` resolves from a frozen spec instead of from ambient files.

### The gap this exposes

A design directory currently has no configuration file at all. Each cell writes
its own `meta` blob into its own npz at `design.py:264-278`; nothing is written
once, up front, that cells can agree on. `scripts/submit_uncertainty.zsh:65-67`
creates `DESIGN_DIR` on the login node and then submits, so there is a correct
place to write one --- it just does not happen today.

### New command

```
deconvolve uncertainty freeze -d DIR [the same options as run]
```

Resolves the full five-layer stack once and writes `DIR/design.json`: the
resolved values plus their origins. It refuses to overwrite an existing
`design.json` unless `--force`, because overwriting one while an array is in
flight is the precise hazard being guarded against.

`scripts/submit_uncertainty.zsh` calls it between `mkdir -p "${DESIGN_DIR}"`
and `sbatch`, on the login node, once.

### Resolution inside a cell

The callback omits the `uncertainty.run` path from `default_map`, so no
discovered file can reach it. Inside `uncertainty_run_command`, every parameter
whose `ctx.get_parameter_source()` is not `COMMANDLINE` is replaced by the
frozen value:

```
COMMANDLINE  >  design.json  >  code default
```

Environment variables are deliberately excluded --- an exported `DECONVOLVE_*` in a
batch script is as capable of splitting a design as an edited file is.

`deconvolve uncertainty run` with no `design.json` in the directory is an error naming
the `freeze` command. It does not fall back to code defaults, because a silent
fallback would produce exactly the corrupted design this section exists to
prevent. This is a breaking change to `scripts/submit_uncertainty.zsh`, which
is updated in the same change.

## Error handling

Every case below is a hard failure with a non-zero exit. Nothing is a warning.

| Failure                              | Behaviour                                                                     |
| ------------------------------------ | ----------------------------------------------------------------------------- |
| Malformed TOML                       | Report the path and `TOMLDecodeError`'s line and column. Never partial-parse. |
| Unknown key                          | Report path, key, and a `difflib.get_close_matches` suggestion.               |
| Unknown table                        | Same, matched against the real command tree.                                  |
| Key naming a non-layerable option    | Report which option and why it is excluded.                                   |
| Non-integral float for an int option | Reject. Click would otherwise truncate `3.7` to `3` in silence.               |
| Out-of-range value, bad enum member  | Click's existing error. No new code.                                          |
| Unreadable file (permissions)        | Report the path and the OS error.                                             |
| Broken global XDG file               | Hard failure, identical to a broken project file.                             |
| No config file anywhere              | Normal operation. Every layer is optional.                                    |

A global config that silently stopped applying would be worse than one that
stops the run, because the run would still produce plausible numbers.

## `deconvolve config show`

A new `config` sub-app, registered like `baseline` and `uncertainty`, with a
single `show` subcommand. None of its own options are layerable.

```
$ deconvolve config show
Layers (nearest last):
  ~/.config/deconvolve/deconvolve.toml                    found
  /Users/krish/Developer/RANv4/deconvolve.toml     found   (walk stopped: git root)
  [tool.deconvolve] in pyproject.toml              shadowed by deconvolve.toml

train.n-epochs      500        deconvolve.toml
train.lr-g          3e-05      default
log-level           debug      env:DECONVOLVE_LOG_LEVEL

Environment-only (not layered):
  DECONVOLVE_CACHE_DIR     .cache     unset
  DECONVOLVE_TIMING        off        unset
```

`deconvolve config show train` scopes the listing to one command. `--json` emits the
same content for scripting.

The rendered origins come from the same `Resolved` object that supplies
`config.json`'s `_origin` block, so the human-facing answer and the recorded
answer cannot drift apart.

## Provenance in `config.json`

`workflows/train.py:250-262` builds `config_out` from every knob that
distinguishes one run from another. It gains one key:

```json
{
  "n_epochs": 500,
  "_origin": {
    "n_epochs": "deconvolve.toml:/Users/krish/Developer/deconvolve/deconvolve.toml",
    "lr_g": "default",
    "batch_size": "env:DECONVOLVE_BATCH_SIZE"
  }
}
```

`_origin` is leading-underscored to mark it as metadata about the run rather
than a parameter of it. `baselines/_shared.py:parse_run_config` ignores unknown
keys already, so existing readers are unaffected, and a run directory written
before this change simply has no `_origin`.

## Deferred: `cache-dir` and `timing`

These two settings cannot join the stack without a separate refactor, and are
explicitly out of scope.

`CACHE_DIR` is resolved at import of `rantypes/constants.py:35` and then bound
into default arguments at `data/jets.py:62`, `data/download.py:272` and
`data/datasets.py:153`. `COMPILE_CACHE_DIR` derives from it and is read at
`training/engine.py:745`. `DECONVOLVE_TIMING` has the same shape at
`instrumentation/timing.py:122`, where module state is initialized at import.

By the time the Typer callback runs, all of those are already bound. Folding
them in means converting module-level constants into lazy accessors and
unbinding three default arguments --- a change with its own reproducibility
risk, in code paths that decide where datasets are cached.

Both therefore remain environment-only. `ran config show` lists them in a
distinct section so the boundary is visible rather than surprising, and
`docs/claude/caching.md` gains a paragraph recording why.

## Alternatives considered

**Sentinel defaults plus an explicit resolver.** Every layerable option defaults
to `None`; a `resolve()` fills the gaps. Rejected: it moves all ~40 defaults out
of the signatures into a parallel table that must be kept in sync, it destroys
the effective-default display in `--help`, and it adds boilerplate to each of
seven commands --- all to reimplement precedence that Click already implements
and that was measured working on the pinned version.

**`pydantic-settings`.** Rejected: a new runtime dependency on a project that
currently ships nine, whose built-in sources provide neither the upward walk nor
the `[tool.ran]` table. Both would be written as custom sources anyway, so the
dependency buys nothing that approach A does not already have.

**Merging every config file on the path.** Rejected: it permits a repo-wide
default plus per-subdirectory overrides, at the cost of a value having several
possible homes and "where did this come from" having several possible answers.
Nearest-wins was chosen deliberately for legibility.

**Walking past the git root to the filesystem root.** Rejected: a stray
`ran.toml` in `$HOME` would silently capture every run started anywhere beneath
it. The global XDG file already serves the "applies everywhere" case, and does
so at a path that is explicit about it.

## Testing

`config.py` is a pure leaf module, so the bulk of this is fast unit tests that
never import JAX or Keras. Nothing here is marked `slow`; nothing needs a GPU.

**Discovery** (`tmp_path` trees):

- the walk stops at the git root, ignoring a `ran.toml` above it;
- `ran.toml` shadows `pyproject.toml` in the same directory;
- the nearest directory wins and the walk stops, ignoring a farther file;
- a `pyproject.toml` with no `[tool.ran]` table does not stop the walk;
- `.git` as a file (worktree) bounds the walk as `.git` as a directory does;
- no repository, and no config at all, both resolve cleanly.

**Merge and validation**: one test per row of the error table, each asserting
the message names the offending key and the file it came from.

**Precedence, end to end**: Typer's `CliRunner` with patched cwd and
environment, asserting `ParameterSource` values rather than only final numbers,
so a regression reports which layer broke rather than only that something did.

**Freeze**:

- `uncertainty run` without `design.json` exits non-zero naming `freeze`;
- a command-line flag overrides a frozen value;
- editing `ran.toml` between `freeze` and `run` does not change the cell.

That last case is the regression test for the property the freeze path exists
to protect.

**`ran config show`**: golden-ish assertions on the origin strings for a
constructed layer stack, including the environment-only section.

## Documentation

- New `docs/claude/configuration.md`: the layer table, the discovery rule, the
  schema, and the freeze path.
- A row for it in the CLAUDE.md Reference Docs table.
- `docs/claude/caching.md`: a paragraph on why `cache-dir` stayed
  environment-only, extending the existing discussion at lines 36-44.
- `docs/claude/uncertainty.md`: the `freeze` step in the design workflow.
- `docs/claude/cli-and-running.md`: `ran config show` and `ran uncertainty
freeze` in the CLI reference.
