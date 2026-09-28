# Run Directory Legibility and `ran report` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a finished run directory readable by a human — two files at the
root, everything else under `artifacts/` — and add `ran report`, which compiles
a run's configuration, timings, metrics and figures into one LaTeX dossier.

**Architecture:** Six independent changes plus one new module. The layout change
routes every binary and generated file through a single `artifacts_dir()`
helper. The figure changes are confined to `src/ran/plotting.py`. The timings
change makes `timing.write` merge rather than overwrite. `src/ran/report.py` is
new: it reads the JSON a run already writes, substitutes `<<TOKEN>>`
placeholders into a shipped LaTeX template, and shells out to `pdflatex`.

**Tech Stack:** Python 3.13, uv, Typer, matplotlib, siunitx/booktabs LaTeX,
pytest. Lint/format with ruff, types with pyrefly, complexity with complexipy
(max 10). All commands go through `just` or `uv run`.

**Spec:** `docs/superpowers/specs/2026-09-06-run-dir-and-report-design.md`

## Global Constraints

- **Never use `print`.** `tests/test_source_hygiene.py` fails the build on a
  `print` call in production Python. Use `logging.getLogger(__name__)`.
- **`SUBSTRUCTURE_VARIABLES` must not change.** It is the jet column order, the
  cache key, and what `config.json` records. See the Jet Column Order section of
  `CLAUDE.md`. The new `JET_DISPLAY_ORDER` is a *display* permutation applied at
  render time only.
- **No backward compatibility with existing run directories.** Old runs stop
  being readable; there is no migration command and no path fallback.
- **Float32 is the pinned event dtype** (`EVENT_DTYPE`). Nothing in this plan
  changes dtypes. `np.float32` is not JSON-serializable — coerce with `.item()`.
- **Complexity ceiling is 10** (`complexipy`). Split a function rather than
  exceed it.
- **Every task ends green:** `just validate` must pass before the commit step.
- Prefer `rg` over `grep` and `fd` over `find`.
- Python is routed through `uv`: run scripts as `uv run <script.py>`, never
  `python <script.py>`.

---

### Task 1: The `artifacts/` subdirectory

**Files:**
- Modify: `src/ran/rantypes/constants.py` (add `ARTIFACTS_DIR`, `artifacts_dir`)
- Modify: `src/ran/rantypes/__init__.py` (re-export both)
- Modify: `src/ran/workflow.py:201-210`, `:238-240`, `:299`, `:281-288`
- Modify: `src/ran/train.py:150`, `:166`
- Modify: `src/ran/evaluate.py:486`, `:498`
- Modify: `src/ran/baselines/ibu.py:396`, `:417`
- Modify: `src/ran/timing.py:275`
- Modify: `benchmarks/ceiling.py:280,281,415`, `benchmarks/averaging.py:244`,
  `benchmarks/sliced.py:317`, `benchmarks/hparam_collect.py:209,266`
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `ran.rantypes.artifacts_dir(run_dir: Path) -> Path` — returns
  `run_dir / "artifacts"`, creating it (`parents=True, exist_ok=True`).
  Every later task uses it. `config.json` stays at `run_dir / "config.json"`;
  `report.pdf` will land at `run_dir / "report.pdf"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_workflow.py`:

```python
def test_a_completed_run_keeps_only_two_files_at_its_root(
    tmp_path: Path, tiny_splits: DatasetSplits
) -> None:
    """The run root is for humans; everything else lives in `artifacts/`."""
    run_dir: Path = tmp_path / "run"
    _run_tiny(run_dir, tiny_splits)  # existing helper; see test_workflow.py

    at_root: set[str] = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert at_root == {"config.json"}

    produced: set[str] = {p.name for p in (run_dir / "artifacts").iterdir()}
    assert {
        "generator.keras",
        "discriminator.keras",
        "history.npz",
        "params.npz",
    } <= produced
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k test_a_completed_run_keeps_only_two_files_at_its_root`
Expected: FAIL — `generator.keras` is still at the run root, so `at_root` has
five entries.

- [ ] **Step 3: Add the constant and helper**

In `src/ran/rantypes/constants.py`, below `RUN_DIR`:

```python
# A run directory is read by people. `config.json` and `report.pdf` stay at the
# root because they are what a person opens; everything else -- checkpoints,
# arrays, figures, the metrics and timing JSON -- is supporting material and
# lives one level down, flat.
ARTIFACTS_DIR: Final[LiteralString] = "artifacts"


def artifacts_dir(run_dir: Path, /) -> Path:
    """The run's supporting-material subdirectory, created on demand."""
    path: Path = run_dir / ARTIFACTS_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path
```

Add `from pathlib import Path` to the runtime imports if it is only under
`TYPE_CHECKING` — `artifacts_dir` needs it at runtime.

Re-export from `src/ran/rantypes/__init__.py`, in alphabetical position:

```python
from .constants import ARTIFACTS_DIR as ARTIFACTS_DIR
from .constants import artifacts_dir as artifacts_dir
```

- [ ] **Step 4: Route every writer and reader through it**

`src/ran/workflow.py` — in `_write_run_dir`, replace the four write paths:

```python
    artifacts: Path = artifacts_dir(run_dir)
    g.save(artifacts / "generator.keras")
    d.save(artifacts / "discriminator.keras")
    _ = save_params(run_dir, params)
    np.savez(
        file=artifacts / "history.npz",
        **{k: np.array(object=v) for k, v in history.items()},  # pyrefly: ignore[bad-argument-type]  # ty:ignore[invalid-argument-type]
    )
```

`config.json` at line 231 is **unchanged** — it stays at the run root.

In `_load_artifacts` (line 236):

```python
    artifacts: Path = artifacts_dir(run_dir)
    g: RANModel = keras.saving.load_model(artifacts / "generator.keras")
    history: dict[str, list[float]] = {
        k: v.tolist() for k, v in np.load(file=artifacts / "history.npz").items()
    }
```

In `_load_baseline_weights` (line 299): `ibu_path: Path = artifacts_dir(run_dir) / "ibu_weights.npz"`.

In `_draw_figures` (lines 281-288), bind `artifacts = artifacts_dir(run_dir)`
once and change the four figure paths to `artifacts / "<name>.pdf"`.

`src/ran/train.py` — `save_params` and `load_params` join `PARAMS_FILE`
themselves, so this is two lines and every caller is unaffected:

```python
    path: Path = artifacts_dir(run_dir) / PARAMS_FILE     # line 150
    with np.load(file=artifacts_dir(run_dir) / PARAMS_FILE) as f:   # line 166
```

`src/ran/evaluate.py` — line 486 `out_path: Path = artifacts_dir(run_dir) / "metrics.json"`;
line 498 `keras.saving.load_model(artifacts_dir(run_dir) / "generator.keras")`.

`src/ran/baselines/ibu.py` — line 396 `out_path: Path = artifacts_dir(run_dir) / "metrics_ibu.json"`;
line 417 `weights_path: Path = artifacts_dir(run_dir) / "ibu_weights.npz"`.

`src/ran/timing.py` — line 275 writes into `artifacts_dir(run_dir) / "timings.json"`.

`benchmarks/` — `ceiling.py:280` and `:415` and `:281` (the `PARAMS_FILE`
existence probe, which does *not* go through `load_params`), `averaging.py:244`,
`sliced.py:317`, `hparam_collect.py:209` and `:266`. Each becomes
`artifacts_dir(run_dir) / "<name>"`.

- [ ] **Step 5: Repoint the tests that name a moved artifact**

`tests/test_completion_logging.py:69,103,171,172`,
`tests/test_workflow.py:256,387,666,690`, `tests/test_train.py:452,454`,
`tests/test_hparam_collect.py:52,317,320,329`, `tests/test_timing.py:72,195,206,217,226`.
Each path gains `/ "artifacts"`.

- [ ] **Step 6: Run the full suite**

Run: `just test`
Expected: PASS. A failure here is a call site missed in Step 4 — `rg -n
'run_dir / "' src benchmarks tests` finds the stragglers.

- [ ] **Step 7: Update the docs**

`README.md:322-329`, `overview.md:266-274`, and the Project Structure and
Running sections of `CLAUDE.md` get the new tree:

```text
runs/<timestamp>/
├── report.pdf
├── config.json
└── artifacts/   figures, metrics/timings JSON, checkpoints, arrays
```

- [ ] **Step 8: Commit**

```bash
git add -A src benchmarks tests README.md overview.md CLAUDE.md
git commit -m "refactor: move run artifacts into an artifacts/ subdirectory

The run root now holds config.json (and, later, report.pdf) and nothing
else. Everything a person does not read by hand -- checkpoints, arrays,
figures, metrics and timings JSON -- moves one level down.

Breaking: existing run directories are no longer readable. No migration
path, by design.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 2: One-line `variables` in `config.json`

**Files:**
- Modify: `src/ran/workflow.py` (add `_compact_variables`, use it at line 231)
- Test: `tests/test_workflow.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `workflow._compact_variables(text: str) -> str` — private, used only
  by `_write_run_dir`.

- [ ] **Step 1: Write the failing test**

```python
def test_config_json_renders_the_variable_list_on_one_line() -> None:
    """Twelve one-line strings should not cost twelve lines of a config file."""
    dumped: str = json.dumps(
        {"dim": 12, "variables": ["m", "M", "w"], "seed": 3}, indent=2
    )

    compacted: str = _compact_variables(dumped)

    assert '"variables": ["m", "M", "w"]' in compacted
    assert json.loads(compacted) == json.loads(dumped)


def test_compacting_leaves_a_config_without_variables_alone() -> None:
    """The Gaussian path records `gaussian_params` and no `variables` key."""
    dumped: str = json.dumps({"dim": 2, "gaussian_params": {"mu_gen": [0, 0]}}, indent=2)

    assert _compact_variables(dumped) == dumped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k compact`
Expected: FAIL — `ImportError: cannot import name '_compact_variables'`.

- [ ] **Step 3: Implement**

In `src/ran/workflow.py`, near the other module-level helpers:

```python
# `json.dump(indent=2)` has no way to keep one array inline, and a twelve-name
# variable list costs fourteen lines of a config a person is meant to read.
_VARIABLES_ARRAY: Final[re.Pattern[str]] = re.compile(
    pattern=r'("variables": )\[[^\]]*\]', flags=re.DOTALL
)


def _compact_variables(text: str, /) -> str:
    """Re-render the `variables` array of a dumped config on a single line."""

    def _one_line(match: re.Match[str]) -> str:
        names: list[str] = json.loads(s=match.group(0).split(sep=": ", maxsplit=1)[1])
        return match.group(1) + json.dumps(obj=names)

    return _VARIABLES_ARRAY.sub(repl=_one_line, string=text)
```

Add `import re` at the top.

- [ ] **Step 4: Use it at the write site**

Replace `workflow.py:231`:

```python
    _ = (run_dir / "config.json").write_text(
        data=_compact_variables(json.dumps(obj=config_out, indent=2))
    )
```

- [ ] **Step 5: Run the tests**

Run: `just test -k "compact or workflow"`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ran/workflow.py tests/test_workflow.py
git commit -m "feat: render config.json's variable list on one line

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 3: Jet display order

**Files:**
- Modify: `src/ran/rantypes/constants.py`
- Modify: `src/ran/rantypes/__init__.py`
- Test: `tests/test_jets.py`

**Interfaces:**
- Consumes: `SUBSTRUCTURE_VARIABLES` (unchanged).
- Produces:
  - `JET_DISPLAY_ORDER: tuple[LiteralString, ...]` — twelve names.
  - `JET_VARIABLE_GROUPS: tuple[tuple[str, tuple[LiteralString, ...]], ...]` —
    four `(label, members)` pairs.
  - `display_order(variables: Sequence[str], /) -> tuple[int, ...]` — indices
    into `variables`, in presentation order. Tasks 5 and 11 consume this.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_jets.py`:

```python
def test_display_order_permutes_the_twelve_observables() -> None:
    """Physics reading order: mass, angularities, splitting, hadronization."""
    order: tuple[int, ...] = display_order(SUBSTRUCTURE_VARIABLES)
    named: tuple[str, ...] = tuple(SUBSTRUCTURE_VARIABLES[i] for i in order)

    assert named == (
        "m", "sdm",
        "lha", "w", "ang2",
        "zg", "tau21",
        "M", "n_ch", "f_ch", "ptd", "q",
    )


def test_display_order_filters_a_subset() -> None:
    """`--var w --var m` still presents mass before angularity."""
    assert tuple("wm"[i] for i in display_order(("w", "m"))) == ("m", "w")


def test_display_order_is_the_identity_for_a_gaussian_run() -> None:
    """A Gaussian run has `dim_0 ... dim_n` and no physics ordering."""
    assert display_order(("dim_0", "dim_1", "dim_2")) == (0, 1, 2)


def test_the_display_order_and_the_column_order_hold_the_same_names() -> None:
    """Two tuples that must never drift apart, guarded rather than documented."""
    assert set(JET_DISPLAY_ORDER) == set(SUBSTRUCTURE_VARIABLES)
    assert len(JET_DISPLAY_ORDER) == len(SUBSTRUCTURE_VARIABLES)


def test_the_groups_partition_the_display_order() -> None:
    grouped: list[str] = [v for _, members in JET_VARIABLE_GROUPS for v in members]
    assert tuple(grouped) == JET_DISPLAY_ORDER


def test_the_column_order_is_unchanged() -> None:
    """The cache key. See the Jet Column Order section of CLAUDE.md."""
    assert SUBSTRUCTURE_VARIABLES == (
        "m", "M", "w", "tau21", "zg", "sdm",
        "q", "f_ch", "lha", "ang2", "ptd", "n_ch",
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k display_order`
Expected: FAIL — `ImportError: cannot import name 'display_order'`.

- [ ] **Step 3: Implement**

In `src/ran/rantypes/constants.py`, after `JET_OBS`:

```python
# How the observables are *presented*. This is not `SUBSTRUCTURE_VARIABLES`,
# and must never become it: that tuple is the column order, the cache key and
# what `config.json` records, and the Jet Column Order section of `CLAUDE.md`
# documents what happened the last time it was allowed to float. The column
# order carries no physics; this one does, and is applied at render time only.
#
# m -> ln rho -> lambda^1_0.5 -> w -> lambda^1_2 -> z_g -> tau_21
#   -> M -> n_ch -> f_ch -> p_T^D -> q
#
# `M` next to `n_ch` exposes the baseline hadronization ratio (n_ch / M ~ 2/3,
# from pion isospin); `f_ch` bridges particle counting and track-based energy
# reconstruction; `p_T^D` completes the quark/gluon discriminant system with
# `M` and `n_ch`; `q` closes as the valence flavour indicator.
JET_DISPLAY_ORDER: Final[tuple[LiteralString, ...]] = (
    "m", "sdm",
    "lha", "w", "ang2",
    "zg", "tau21",
    "M", "n_ch", "f_ch", "ptd", "q",
)

JET_VARIABLE_GROUPS: Final[tuple[tuple[str, tuple[LiteralString, ...]], ...]] = (
    ("Mass and hard scale (IRC-safe kinematics)", ("m", "sdm")),
    ("Continuous angularities (IRC-safe jet shapes)", ("lha", "w", "ang2")),
    ("Splitting and 2-prong substructure", ("zg", "tau21")),
    (
        "Hadronization, multiplicity and fragmentation (IRC-unsafe)",
        ("M", "n_ch", "f_ch", "ptd", "q"),
    ),
)


def display_order(variables: Sequence[str], /) -> tuple[int, ...]:
    """Indices into `variables`, reordered for presentation.

    Filters `JET_DISPLAY_ORDER` to what this run actually holds, so a `--var`
    subset stays in physics order. A non-jet run (`dim_0`, `dim_1`, ...) has no
    entry in the table and falls through to the identity.
    """
    position: dict[str, int] = {name: i for i, name in enumerate(iterable=variables)}
    ordered: tuple[int, ...] = tuple(
        position[name] for name in JET_DISPLAY_ORDER if name in position
    )
    return ordered if len(ordered) == len(variables) else tuple(range(len(variables)))
```

Import `Sequence` at runtime (it is currently under `TYPE_CHECKING` in some
modules — in `constants.py` add `from collections.abc import Sequence`).

Re-export all three from `src/ran/rantypes/__init__.py`.

- [ ] **Step 4: Run the tests**

Run: `just test -k "display_order or column_order or groups"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ran/rantypes/constants.py src/ran/rantypes/__init__.py tests/test_jets.py
git commit -m "feat: add a physics-motivated jet display order

A render-time permutation, deliberately separate from
SUBSTRUCTURE_VARIABLES, which stays the column order and cache key.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 4: Figure palette, opacity, saving and tick pruning

**Files:**
- Modify: `src/ran/plotting.py:92-215`
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: nothing.
- Produces: module constants `COLOR_NATURE`, `COLOR_MC`, `COLOR_IBU`,
  `COLOR_RAN`, `ALPHA_FILL`, `ALPHA_RAN`, `ALPHA_IBU` in `ran.plotting`.

- [ ] **Step 1: Write the failing test**

```python
def test_ran_is_drawn_more_prominently_than_the_baseline() -> None:
    """RAN's step line was fainter than IBU's. On the same panel."""
    assert plotting.ALPHA_RAN > plotting.ALPHA_IBU > plotting.ALPHA_FILL


def test_ran_has_a_colour_of_its_own() -> None:
    assert plotting.COLOR_RAN not in {
        plotting.COLOR_NATURE, plotting.COLOR_MC, plotting.COLOR_IBU, "black"
    }


def test_saving_a_figure_trims_to_its_contents(monkeypatch, tmp_path: Path) -> None:
    """Without `bbox_inches`, the y-labels are clipped by the page edge."""
    seen: dict[str, object] = {}
    figure = Figure()
    monkeypatch.setattr(
        target=figure, name="savefig", value=lambda **kw: seen.update(kw)
    )

    plotting._save_fig(figure, save_path=tmp_path / "f.pdf")

    assert seen["bbox_inches"] == "tight"


def test_the_main_panel_prunes_its_lowest_tick(tmp_path: Path) -> None:
    """The main axis `0` and the ratio axis `1.5` overprinted each other."""
    figure = Figure()
    figure.canvas = FigureCanvasPdf(figure)
    ax, ax_r = figure.subplots(nrows=2)
    rng = np.random.default_rng(seed=0)

    plotting._hist_ratio_panel(
        ax, ax_r,
        x_nature=rng.normal(size=512).astype(np.single),
        x_mc=rng.normal(size=512).astype(np.single),
        w_ran=np.ones(512, dtype=np.single),
        bins=20, nature_label="Data", mc_label="Sim",
        xlabel="x", title="t",
    )
    figure.canvas.draw()

    assert isinstance(ax.yaxis.get_major_locator(), MaxNLocator)
    assert ax.get_yticks()[0] > ax.get_ylim()[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k "prominently or trims or prunes"`
Expected: FAIL — `AttributeError: module 'ran.plotting' has no attribute 'ALPHA_RAN'`.

- [ ] **Step 3: Add the constants**

In `src/ran/plotting.py`, after the `mpl.rcParams` block:

```python
# One place for the figure's visual hierarchy, rather than seven literals
# scattered through `_hist_ratio_panel`. RAN's step line used to be black at
# alpha 0.35 while IBU's ratio line was at 0.75 -- the baseline drawn twice as
# prominently as the method being showcased, on the same panel.
COLOR_NATURE: Final[str] = "C0"      # Data / Truth
COLOR_MC: Final[str] = "C1"          # Sim / Gen
COLOR_IBU: Final[str] = "green"
COLOR_RAN: Final[str] = "#6A3D9A"    # deep violet; greyscales to a dark mid-tone

ALPHA_FILL: Final[float] = 0.35      # the two filled background histograms
ALPHA_IBU: Final[float] = 0.75
ALPHA_RAN: Final[float] = 0.90
```

- [ ] **Step 4: Use them, and prune the tick**

In `_hist_ratio_panel`, replace every colour and alpha literal:

| Line | Was | Becomes |
| --- | --- | --- |
| 112 | `color="C0"`, `alpha=0.35` | `color=COLOR_NATURE`, `alpha=ALPHA_FILL` |
| 123 | `color="C1"`, `alpha=0.35` | `color=COLOR_MC`, `alpha=ALPHA_FILL` |
| 134 | `color="black"`, `alpha=0.35` | `color=COLOR_RAN`, `alpha=ALPHA_RAN` |
| 161 | `color="C1"`, `alpha=0.35` | `color=COLOR_MC`, `alpha=ALPHA_FILL` |
| 169 | `color="black"`, `alpha=0.35` | `color=COLOR_RAN`, `alpha=ALPHA_RAN` |
| 183 | `color="green"`, `alpha=0.35` | `color=COLOR_IBU`, `alpha=ALPHA_IBU` |
| 200 | `color="green"`, `alpha=0.75` | `color=COLOR_IBU`, `alpha=ALPHA_IBU` |

Then, just before `_ = ax_r.set_xlabel(xlabel)` at the end of the function:

```python
    # The main panel's bottom tick and the ratio panel's top tick land at the
    # same height where the two axes meet and overprint each other. `prune`
    # drops the lowest label only when it sits at the axis edge, which is
    # exactly the collision and nothing else.
    ax.yaxis.set_major_locator(locator=MaxNLocator(prune="lower"))
```

Import at the top: `from matplotlib.ticker import MaxNLocator`.

- [ ] **Step 5: Add `bbox_inches` to the one save path**

```python
def _save_fig(figure: Figure, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Without this the y-labels are clipped by the page edge. `plot_losses`
    # always passed it and never clipped; the other three did not and did.
    figure.savefig(fname=save_path, bbox_inches="tight")
    logger.info("Saved %s", save_path)
```

Then route `plot_losses` (line 435) and `plot_selection` (line 503) through
`_save_fig(figure, save_path=Path(save_path))` so there is one save path.

- [ ] **Step 6: Run the tests**

Run: `just test -k plotting`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/ran/plotting.py tests/test_plotting.py
git commit -m "fix: stop clipping figure labels and drawing RAN fainter than IBU

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 5: The 4x3 panel grid, in display order

**Files:**
- Modify: `src/ran/plotting.py:280-326` (`_plot_level`)
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: `ran.rantypes.display_order` (Task 3).
- Produces: nothing new; `_plot_level` keeps its signature.

- [ ] **Step 1: Write the failing test**

```python
def test_twelve_observables_are_drawn_four_rows_by_three(tmp_path: Path) -> None:
    """A 1x12 column is not a figure anyone reads."""
    save_path: Path = tmp_path / "detector.pdf"
    _plot_twelve_dim_level(save_path)  # helper: 12-column nature/mc/weights

    figure: Figure = _last_drawn_figure()
    hist_axes = [a for a in figure.axes if a.get_ylabel() == "Events"]
    assert len(hist_axes) == 12

    columns: set[float] = {round(a.get_position().x0, 3) for a in hist_axes}
    rows: set[float] = {round(a.get_position().y0, 3) for a in hist_axes}
    assert len(columns) == 3
    assert len(rows) == 4


def test_panels_are_drawn_in_display_order(tmp_path: Path) -> None:
    """Panel 0 is `m`, panel 1 is the soft-drop mass, not the multiplicity."""
    titles: list[str] = _panel_titles_for(SUBSTRUCTURE_VARIABLES, tmp_path)
    assert titles[0].startswith("Jet Mass")
    assert titles[1].startswith("Soft Drop Jet Mass")


def test_a_single_dimension_still_draws_one_panel(tmp_path: Path) -> None:
    """The 1D Gaussian config must be unaffected."""
    figure: Figure = _one_dim_level(tmp_path)
    assert len([a for a in figure.axes if a.get_ylabel() == "Events"]) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k "four_rows or display_order"`
Expected: FAIL — all twelve panels share one `x0`, so `len(columns) == 1`.

- [ ] **Step 3: Implement**

Replace the body of `_plot_level` between `dim = nature.shape[1]` and the loop:

```python
    dim: int = nature.shape[1]
    ncols: int = min(3, dim)
    nrows: int = math.ceil(dim / ncols)
    figure = Figure(figsize=(4.0 * ncols, style.height_per_dim * nrows))
    figure.canvas = FigureCanvasPdf(figure)
    # Absolute margins in inches do not survive a figure whose height now
    # varies with `nrows`; `tight_layout` at the end replaces them.
    outer_grid: GridSpec = figure.add_gridspec(nrows=nrows, ncols=ncols, hspace=0.35)

    names: Sequence[str] = (
        variables if variables is not None else [f"dim_{i}" for i in range(dim)]
    )
    order: tuple[int, ...] = display_order(names)
    for position, i in enumerate(iterable=order):
        inner_grid: GridSpecFromSubplotSpec = outer_grid[
            position // ncols, position % ncols
        ].subgridspec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.0)
        ...  # the existing body, unchanged, using `i`
    figure.tight_layout()
    _save_fig(figure, save_path=Path(save_path))
```

**`_plot_level` needs the variable names, and does not currently receive
them.** `display_order` keys on names like `"sdm"`; `var_info` is a `VarInfo`
`TypedDict` carrying `xlim`, `xlabel`, `symbol`, `mu` and `sigma`, and no name
field. So add a parameter, threaded from the run config:

- `_plot_level(..., variables: tuple[str, ...] | None = None)`
- `plot_detector_level(..., variables: tuple[str, ...] | None = None)`
- `plot_particle_level(..., variables: tuple[str, ...] | None = None)`
- `plot_levels(..., variables: tuple[str, ...] | None = None)`

and in `workflow._draw_figures`, pass `variables=tuple(config["variables"])` for
a jet run and `variables=None` otherwise. `None` makes `names` the `dim_i`
placeholders, for which `display_order` returns the identity, so the Gaussian
configs are untouched.

Import `math` at the top of `plotting.py`, and `display_order` from
`.rantypes`.

- [ ] **Step 4: Run the tests**

Run: `just test -k plotting`
Expected: PASS.

- [ ] **Step 5: Check complexity**

Run: `just complexity`
Expected: PASS. If `_plot_level` now exceeds 10, extract the per-panel body into
a `_draw_panel(figure, cell, i, ...)` helper.

- [ ] **Step 6: Commit**

```bash
git add src/ran/plotting.py src/ran/workflow.py tests/test_plotting.py
git commit -m "feat: lay the level figures out as a grid in physics order

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 6: A fixed, honest y-scale for `losses.pdf`

**Files:**
- Modify: `src/ran/plotting.py:399-437` (`plot_losses`)
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `plotting.LN2: Final[float]` and `plotting.LOSS_YLIM_FRACTION: Final[float]`.

- [ ] **Step 1: Write the failing test**

```python
def test_the_loss_axis_is_fixed_around_log_two(tmp_path: Path) -> None:
    """Autoscaling turned a 0.4% band into an apparent divergence."""
    history = {"train_d": [0.689, 0.688], "train_g": [0.688, 0.687],
               "val_d": [0.690, 0.693]}
    figure: Figure = _drawn_losses(history, tmp_path)
    ax = figure.axes[0]

    low, high = ax.get_ylim()
    assert low == pytest.approx(math.log(2) * (1 - 2**-4))
    assert high == pytest.approx(math.log(2) * (1 + 2**-4))


def test_log_two_is_a_tick_and_not_a_legend_entry(tmp_path: Path) -> None:
    history = {"train_d": [0.689], "train_g": [0.688], "val_d": [0.690]}
    ax = _drawn_losses(history, tmp_path).axes[0]

    assert "log(2)" not in [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(t == pytest.approx(math.log(2)) for t in ax.get_yticks())


def test_the_y_label_has_a_space_in_it(tmp_path: Path) -> None:
    history = {"train_d": [0.689], "train_g": [0.688], "val_d": [0.690]}
    assert _drawn_losses(history, tmp_path).axes[0].get_ylabel() == "Weighted BCE"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k "loss_axis or log_two or y_label"`
Expected: FAIL — the limits are matplotlib's autoscaled ones and `"log(2)"` is
in the legend.

- [ ] **Step 3: Implement**

Add near the other constants:

```python
LN2: Final[float] = math.log(2)
# The equilibrium band. Every series a converged run produces sits within a
# fraction of a percent of `ln 2`, and autoscaling that band to the height of
# the axes makes a 0.4% drift look like a divergence. Fixed limits also make
# two runs' loss plots directly comparable.
LOSS_YLIM_FRACTION: Final[float] = 2.0**-4
```

In `plot_losses`, after the three `ax.plot` calls:

```python
    _ = ax.axhline(y=LN2, color="gray", lw=1)  # no `label`: it is a tick, not a series
    _ = ax.set_ylim(
        bottom=LN2 * (1 - LOSS_YLIM_FRACTION), top=LN2 * (1 + LOSS_YLIM_FRACTION)
    )
    offsets: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    ticks: list[float] = [LN2 * (1 + k * 2.0**-5) for k in offsets]
    _ = ax.set_yticks(ticks=ticks)
    _ = ax.set_yticklabels(
        labels=[r"$\ln 2$" if k == 0.0 else f"{t:.4f}" for k, t in zip(offsets, ticks, strict=True)]
    )

    # The same positions as a percentage deviation, so a reader sees "within
    # 1% of equilibrium" without doing the arithmetic.
    deviation: Axes = ax.twinx()
    _ = deviation.set_ylim(*ax.get_ylim())
    _ = deviation.set_yticks(ticks=ticks)
    _ = deviation.set_yticklabels(labels=[f"{k * 2.0**-5 * 100:+.1f}%" for k in offsets])
    _ = deviation.set_ylabel(ylabel=r"Deviation from $\ln 2$")

    _ = ax.set_ylabel(ylabel="Weighted BCE")
```

Import `math` at the top if Task 5 has not already.

- [ ] **Step 4: Run the tests**

Run: `just test -k plotting`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ran/plotting.py tests/test_plotting.py
git commit -m "fix: pin the loss axis to a fixed band around log 2

The curves span 0.688-0.694 and autoscaling rendered that as a
divergence. A right-hand axis reads the same positions as a percentage
deviation.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 7: Restructure `selection.pdf`

**Files:**
- Modify: `src/ran/plotting.py:439-504` (`plot_selection`)
- Test: `tests/test_plotting.py`

**Interfaces:**
- Consumes: `SELECTION_MMD_LINTHRESH` (existing).
- Produces: `plotting.SELECTION_SMOOTHING_WINDOW: Final[int] = 5`.

- [ ] **Step 1: Write the failing test**

```python
def test_selection_splits_mmd_and_ess_into_two_panels(tmp_path: Path) -> None:
    """Three noisy series on one axis with a twin scale read as seismographs."""
    figure: Figure = _drawn_selection(_noisy_history(), best_epoch=38, path=tmp_path)
    panels = [a for a in figure.axes if a.get_xlabel() or a.get_ylabel()]

    assert any("MMD" in a.get_ylabel() for a in panels)
    assert any("Effective sample size" in a.get_ylabel() for a in panels)
    # The ESS panel is its own axes, not a twin of the MMD one.
    ess = next(a for a in panels if "Effective sample size" in a.get_ylabel())
    mmd = next(a for a in panels if "MMD" in a.get_ylabel())
    assert ess.get_position().y1 <= mmd.get_position().y0 + 1e-6


def test_the_correlation_inset_appears_only_with_truth(tmp_path: Path) -> None:
    """A real measurement has no particle-level curve to scatter against."""
    with_truth: Figure = _drawn_selection(_noisy_history(), 38, tmp_path / "a.pdf")
    history = _noisy_history()
    del history["val_mmd_particle"]
    without: Figure = _drawn_selection(history, 38, tmp_path / "b.pdf")

    assert len(with_truth.axes) == len(without.axes) + 1


def test_the_legend_is_outside_the_axes(tmp_path: Path) -> None:
    """It used to cover the bottom third of the plot."""
    figure: Figure = _drawn_selection(_noisy_history(), 38, tmp_path)
    mmd = next(a for a in figure.axes if "MMD" in a.get_ylabel())
    legend_box = mmd.get_legend().get_window_extent(figure.canvas.get_renderer())
    axes_box = mmd.get_window_extent(figure.canvas.get_renderer())
    assert legend_box.x0 >= axes_box.x1 - 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k selection`
Expected: FAIL — the ESS axis is a `twinx` of the MMD axis, so its position
equals the MMD panel's.

- [ ] **Step 3: Implement**

Rewrite `plot_selection`:

```python
SELECTION_SMOOTHING_WINDOW: Final[int] = 5


def _rolling_median(values: NDArray[np.double], window: int, /) -> NDArray[np.double]:
    """Centred rolling median, edges held at the nearest full window."""
    pad: int = window // 2
    padded: NDArray[np.double] = np.pad(values, pad_width=pad, mode="edge")
    return np.array(
        [np.median(padded[i : i + window]) for i in range(values.size)],
        dtype=np.double,
    )
```

The figure becomes two panels via `figure.add_gridspec(nrows=2, height_ratios=[7, 3], hspace=0.08)`:

- Top: for each of `val_mmd` (`COLOR_NATURE`, solid) and `val_mmd_particle`
  (`COLOR_IBU`, dashed), draw the raw series at `alpha=0.3, lw=1` with no label,
  then `_rolling_median(...)` at `lw=2` carrying the label. Keep
  `ax.set_yscale("symlog", linthresh=SELECTION_MMD_LINTHRESH)`. Shade the floor
  with `ax.axhspan(ymin=0, ymax=SELECTION_MMD_LINTHRESH, color="0.85", zorder=0,
  label="estimator resolution floor")`. Mark the selection with
  `ax.axvline(best_epoch, color="k", ls=":", lw=1, label=f"selected (epoch {best_epoch + 1})")`.
  Legend outside: `ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)`.
- Inset, only when `"val_mmd_particle" in history`:
  `inset = ax.inset_axes([0.62, 0.62, 0.35, 0.35])`, scatter detector against
  particle MMD at `s=8, alpha=0.6, color=COLOR_MC`, overplot the selected epoch
  at `s=40, color="k", marker="x"`, label both axes at `fontsize="x-small"`.
- Bottom: `val_ess` as a percentage of its first value —
  `ess_pct = 100 * np.asarray(history["val_ess"]) / history["val_ess"][0]` —
  plotted in `COLOR_MC`, with `set_ylabel("Effective sample size\n(% of epoch 0)")`,
  `set_xlabel("Epoch")`, and `set_ylim` widened to include 100.

Finish with `figure.tight_layout()` and `_save_fig(figure, save_path=Path(save_path))`.

- [ ] **Step 4: Run the tests**

Run: `just test -k selection`
Expected: PASS.

- [ ] **Step 5: Check complexity**

Run: `just complexity`
Expected: PASS. If `plot_selection` exceeds 10, extract `_mmd_panel(ax, history,
best_epoch)` and `_ess_panel(ax, history)`.

- [ ] **Step 6: Commit**

```bash
git add src/ran/plotting.py tests/test_plotting.py
git commit -m "fix: make selection.pdf legible

Two panels instead of three overlaid series, a rolling median over the
raw traces, the resolution floor shaded, the legend outside the axes,
ESS on a percentage scale, and a detector-vs-particle scatter that shows
the correlation two noisy time series cannot.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 8: Match IBU's metric key order to `evaluate`, and persist outcomes

**Files:**
- Modify: `src/ran/baselines/ibu.py:355-385`, `:415`
- Test: `tests/test_ibu.py`

**Interfaces:**
- Consumes: `artifacts_dir` (Task 1).
- Produces: `artifacts/ibu_outcomes.json` — a list of
  `{"variable_name": str, "status": "completed" | "skipped", "n_bins": int,
  "skip_reason": str | None}`. Task 12 consumes it.

- [ ] **Step 1: Write the failing test**

```python
def test_ibu_and_ran_agree_on_metric_key_order(tmp_path: Path) -> None:
    """Same nominal format, two orders, is how a positional zip goes wrong."""
    run_dir: Path = _run_with_ibu(tmp_path, variables=("m", "w"))

    ran_keys = list(json.loads((run_dir / "artifacts/metrics.json").read_text()))
    ibu_keys = list(json.loads((run_dir / "artifacts/metrics_ibu.json").read_text()))
    assert ibu_keys == ran_keys
    assert ibu_keys == ["detector_m", "detector_w", "particle_m", "particle_w"]


def test_ibu_records_which_variables_it_gave_up_on(tmp_path: Path) -> None:
    """A 0% improvement from a skipped variable must not read as a measurement."""
    run_dir: Path = _run_with_ibu(tmp_path, variables=("m", "w"))

    outcomes = json.loads((run_dir / "artifacts/ibu_outcomes.json").read_text())
    assert {o["variable_name"] for o in outcomes} == {"m", "w"}
    assert all(o["status"] in {"completed", "skipped"} for o in outcomes)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k "key_order or gave_up"`
Expected: FAIL — the keys interleave as `detector_m, particle_m, detector_w,
particle_w`, and `ibu_outcomes.json` does not exist.

- [ ] **Step 3: Reorder the metric dict**

In `_run_and_evaluate` (`ibu.py:355`), accumulate two dicts and merge:

```python
    detector: dict[str, MetricRecord] = {}
    particle: dict[str, MetricRecord] = {}
    for dimension, variable_name in enumerate(iterable=config.variable_names):
        ...
        detector[f"detector_{variable_name}"] = evaluate_dimension(
            reference=test.data[:, dimension],
            comparison=test.mc.x[:, dimension],
            weights=test_weights,
        )
        particle[f"particle_{variable_name}"] = evaluate_dimension(
            reference=test_truth[:, dimension],
            comparison=test.mc.z[:, dimension],
            weights=test_weights,
        )

    # Every detector entry, then every particle entry -- the order
    # `evaluate.evaluate_run` writes. Two files in the same nominal format with
    # different key orders is the shape of bug that surfaces the first time
    # someone zips them positionally.
    metrics: dict[str, MetricRecord] = detector | particle
```

- [ ] **Step 4: Persist the outcomes**

In `evaluate_single`, after the metrics dump:

```python
    # `outcomes` records the variables IBU's purity binning gave up on and
    # returned unchanged. Without it, a report showing `IBU == Sim` and a 0.0%
    # improvement reads as a measurement rather than a refusal.
    _ = (artifacts_dir(run_dir) / "ibu_outcomes.json").write_text(
        data=json.dumps(obj=[asdict(obj=o) for o in result.outcomes], indent=2)
    )
```

Import `asdict` from `dataclasses`. `VariableOutcome` has a `KW_ONLY` marker
field named `_`; verify `asdict` does not emit it, and if it does, build the
dict explicitly with the four named fields.

- [ ] **Step 5: Run the tests**

Run: `just test -k ibu`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ran/baselines/ibu.py tests/test_ibu.py
git commit -m "fix: align IBU's metric key order with evaluate, and record outcomes

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 9: Merge `timings.json` instead of overwriting it

**Files:**
- Modify: `src/ran/timing.py:252-275` (`write`)
- Modify: `src/ran/workflow.py:419` (the `write(written_to)` call)
- Test: `tests/test_timing.py`

**Interfaces:**
- Consumes: `artifacts_dir` (Task 1).
- Produces: `timing.write(run_dir: Path, /, *, pass_name: str) -> None`. Every
  phase record in `timings.json` gains a `"pass": str` field.

- [ ] **Step 1: Write the failing test**

```python
def test_a_reload_pass_does_not_destroy_the_training_numbers(
    tmp_path: Path, timing_on: None
) -> None:
    """`scripts/submit.sh` makes three passes over one directory."""
    with timing.phase("train"):
        pass
    timing.write(tmp_path, pass_name="train")
    timing.reset()

    with timing.phase("plots"):
        pass
    timing.write(tmp_path, pass_name="load")

    payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
    by_name = {p["name"]: p for p in payload["phases"]}
    assert by_name.keys() == {"train", "plots"}
    assert by_name["train"]["pass"] == "train"
    assert by_name["plots"]["pass"] == "load"


def test_a_rerun_phase_replaces_its_earlier_record(tmp_path: Path, timing_on) -> None:
    with timing.phase("plots"):
        pass
    timing.write(tmp_path, pass_name="train")
    first = json.loads((tmp_path / "artifacts/timings.json").read_text())
    timing.reset()

    with timing.phase("plots"):
        pass
    timing.write(tmp_path, pass_name="load")
    second = json.loads((tmp_path / "artifacts/timings.json").read_text())

    assert len(second["phases"]) == 1
    assert second["phases"][0]["pass"] == "load"
    assert second["phases"][0]["seconds"] != first["phases"][0]["seconds"]


def test_the_total_sums_the_merged_top_level_phases(tmp_path: Path, timing_on) -> None:
    with timing.phase("train"):
        pass
    timing.write(tmp_path, pass_name="train")
    timing.reset()
    with timing.phase("plots"):
        pass
    timing.write(tmp_path, pass_name="load")

    payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
    top = [p["seconds"] for p in payload["phases"] if p["depth"] == 0]
    assert payload["total_seconds"] == pytest.approx(sum(top))


def test_a_corrupt_timings_file_is_replaced_rather_than_raised_on(
    tmp_path: Path, timing_on
) -> None:
    """The timing layer must never be what takes a run down."""
    (tmp_path / "artifacts").mkdir()
    _ = (tmp_path / "artifacts/timings.json").write_text("{not json")

    with timing.phase("train"):
        pass
    timing.write(tmp_path, pass_name="train")

    payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
    assert [p["name"] for p in payload["phases"]] == ["train"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k "reload_pass or rerun_phase or corrupt"`
Expected: FAIL — `write()` takes no `pass_name`, so `TypeError`.

- [ ] **Step 3: Implement the merge**

```python
def _existing(path: Path, /) -> dict[str, Any]:
    """What is already on disk, or an empty payload.

    A malformed file is treated as absent rather than raised on: this layer
    exists to describe a run, and must never be what ends one.
    """
    try:
        return cast("dict[str, Any]", json.loads(s=path.read_text()))
    except (OSError, ValueError):
        return {}


def write(run_dir: Path, /, *, pass_name: str) -> None:
    """Merge this pass's phases into `timings.json`.

    `scripts/submit.sh` makes three passes over one run directory -- train,
    baseline, then reload for the figures -- and an overwriting writer meant
    the reload destroyed the training numbers on every pipeline run. Phases
    merge by name: this pass replaces a same-named record and leaves the rest.
    `pass_name` is what makes a merged file legible, saying which invocation
    produced each row.
    """
    if _recorder is None or not _recorder.records:
        return
    path: Path = artifacts_dir(run_dir) / "timings.json"
    previous: dict[str, Any] = _existing(path)
    fresh: list[dict[str, Any]] = [
        {
            "name": p.name,
            "seconds": p.seconds,
            "depth": p.depth,
            "detail": p.detail,
            "failed": p.failed,
            "pass": pass_name,
        }
        for p in _ordered(_recorder.records)
    ]
    replaced: frozenset[str] = frozenset(p["name"] for p in fresh)
    kept: list[dict[str, Any]] = [
        p
        for p in cast("list[dict[str, Any]]", previous.get("phases", []))
        if p["name"] not in replaced
    ]
    phases: list[dict[str, Any]] = kept + fresh
    payload: dict[str, Any] = {
        "total_seconds": sum(p["seconds"] for p in phases if p["depth"] == 0),
        "compile_cache_warm": _recorder.compile_cache_warm,
        "phases": phases,
    }
    _ = path.write_text(data=json.dumps(obj=payload, indent=2))
```

If `_recorder.compile_cache_warm` is `None` on a reload pass, fall back to
`previous.get("compile_cache_warm")`.

- [ ] **Step 4: Pass the name from `workflow.run`**

At `workflow.py:419`:

```python
        if written_to is not None:
            write(written_to, pass_name="load" if load_run is not None else "train")
```

- [ ] **Step 5: Run the tests**

Run: `just test -k timing`
Expected: PASS.

- [ ] **Step 6: Fix the stale comment in `scripts/submit.sh`**

Lines 10 and 31 describe `timings.json` as a single pass's output. Update to
note that the file now accumulates across the three passes.

- [ ] **Step 7: Commit**

```bash
git add src/ran/timing.py src/ran/workflow.py scripts/submit.sh tests/test_timing.py
git commit -m "fix: merge timings.json across passes instead of overwriting it

The reload pass in scripts/submit.sh destroyed the training phases on
every pipeline run. Phases now merge by name and carry the pass that
produced them.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 10: Ship the template, and the value-formatting primitives

**Files:**
- Create: `src/ran/templates/report.tex` (move `response.tex` here)
- Create: `src/ran/report.py`
- Create: `tests/test_report.py`
- Delete: `response.tex`

**Interfaces:**
- Consumes: nothing.
- Produces, all in `ran.report`:
  - `TEMPLATE_TOKEN: Final[re.Pattern[str]]` — `re.compile(r"<<[A-Z_]+>>")`
  - `load_template() -> str`
  - `decimal(value: float, /) -> str` — plain decimal, never exponential
  - `latex_text(value: str, /) -> str` — escapes `\ & % $ # _ { } ~ ^`
  - Tasks 11 and 12 consume all four.

- [ ] **Step 1: Move the template into the package**

```bash
mkdir -p src/ran/templates
git mv response.tex src/ran/templates/report.tex
```

- [ ] **Step 2: Write the failing test**

```python
def test_the_template_ships_with_the_package() -> None:
    """It is package data, not a repo-root file the wheel would drop."""
    assert "<<DETECTOR_TABLE>>" in report.load_template()


def test_every_token_the_generator_fills_is_in_the_template() -> None:
    tokens = set(report.TEMPLATE_TOKEN.findall(report.load_template()))
    assert tokens == {
        "<<RUN_NAME>>", "<<CONFIG_ROWS>>", "<<TIMINGS_ROWS>>",
        "<<DETECTOR_TABLE>>", "<<PARTICLE_TABLE>>", "<<FIGURE_DIR>>",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.2089450359, "0.2089450359"),
        (8.29431403648554e-05, "0.0000829431"),
        (0.0, "0"),
        (-504.2501717, "-504.2501717"),
    ],
)
def test_numbers_are_emitted_as_plain_decimals(value: float, expected: str) -> None:
    """siunitx S columns fix a table-format and cannot absorb an exponent."""
    rendered: str = report.decimal(value)
    assert "e" not in rendered.lower()
    assert float(rendered) == pytest.approx(value, rel=1e-6)


def test_underscores_in_a_value_are_escaped() -> None:
    """`f_ch` inside \\texttt{} was a hard `Missing $ inserted` compile error."""
    assert report.latex_text("f_ch") == r"f\_ch"
    assert report.latex_text("100%") == r"100\%"


def test_the_token_guard_ignores_the_templates_own_documentation() -> None:
    """The header comment documents the contract and contains `<<...>>`."""
    assert report.TEMPLATE_TOKEN.findall("% Replace the six <<...>> tokens") == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `just test -k report`
Expected: FAIL — `ModuleNotFoundError: No module named 'ran.report'`.

- [ ] **Step 4: Implement**

`src/ran/report.py`:

```python
"""Compile a run directory into one LaTeX dossier.

`config.json`, `metrics.json` and `timings.json` are machine interfaces and
stay exactly as they are; this module is a read-only consumer that turns them
into something a person reads. All rounding policy lives in the template, in
siunitx column specifications -- a column formatted string-by-string in Python
cannot align on the decimal marker -- so everything here emits full precision
and lets LaTeX decide how much of it to show.
"""

from __future__ import annotations

import logging
import re
from importlib import resources
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from logging import Logger

logger: Logger = logging.getLogger(name=__name__)

# `<<[A-Z_]+>>` rather than a bare `<<`: the template's own header comment
# documents the replacement contract and legitimately contains `<<...>>`.
TEMPLATE_TOKEN: Final[re.Pattern[str]] = re.compile(pattern=r"<<[A-Z_]+>>")

_LATEX_SPECIALS: Final[dict[str, str]] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}

# Enough places for four significant figures of the smallest metric a real run
# produces (~8.3e-5 -> 0.00008294). The template's D column is `table-format=2.8`.
_DECIMAL_PLACES: Final[int] = 10


def load_template() -> str:
    """The shipped LaTeX template, as text."""
    return (resources.files("ran") / "templates" / "report.tex").read_text(
        encoding="utf-8"
    )


def decimal(value: float, /) -> str:
    """Plain decimal notation, never exponential.

    Every numeric cell lands in a siunitx `S` column, which fixes a
    `table-format` and cannot absorb an exponent -- a value emitted as
    `8.294e-05` misaligns the column silently rather than erroring.
    """
    rendered: str = f"{value:.{_DECIMAL_PLACES}f}".rstrip("0").rstrip(".")
    return rendered or "0"


def latex_text(value: str, /) -> str:
    """Escape the characters that would end a compile or change the meaning."""
    return "".join(_LATEX_SPECIALS.get(character, character) for character in value)
```

- [ ] **Step 5: Make the template reachable from an installed wheel**

`uv_build` includes non-Python files under the module directory, but confirm
rather than assume:

Run: `uv build --wheel && unzip -l dist/*.whl | rg templates`
Expected: `ran/templates/report.tex` is listed. If it is not, add to
`pyproject.toml`:

```toml
[tool.uv.build-backend]
module-name = "ran"
source-include = ["src/ran/templates/*.tex"]
```

- [ ] **Step 6: Run the tests**

Run: `just test -k report`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add -A src/ran/templates src/ran/report.py tests/test_report.py pyproject.toml
git commit -m "feat: ship the report template and its formatting primitives

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 11: Config and timings row bodies

**Files:**
- Modify: `src/ran/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `decimal`, `latex_text` (Task 10); `JET_DISPLAY_ORDER` (Task 3);
  `mmd._SCALES`.
- Produces:
  - `config_rows(config: Mapping[str, Any], timings: Mapping[str, Any] | None, /) -> str`
  - `timing_rows(timings: Mapping[str, Any], /) -> str`

- [ ] **Step 1: Write the failing test**

```python
def test_config_rows_pair_two_entries_per_line() -> None:
    """The config table is four columns: key, value, key, value."""
    rows: str = report.config_rows({"dim": 12, "seed": 3, "n_layers": 2}, None)

    assert r"\ConfigPair{dim}{\Count{12}}{seed}{\Count{3}}" in rows
    # An odd trailing entry pads rather than shifting every later cell.
    assert r"\ConfigPair{n_layers}{\Count{2}}{}{}" in rows


def test_the_variable_list_spans_the_row() -> None:
    rows: str = report.config_rows({"variables": ["m", "f_ch"]}, None)
    assert r"\ConfigWide{variables}{\ConfigVal{m, f_ch}}" in rows


def test_the_mmd_sigmas_collapse_to_a_median_and_a_bracket() -> None:
    """Five bare floats tell a reader nothing; they are median x _SCALES."""
    median: float = 3.10746693611145
    sigmas: list[float] = [median * s for s in (0.5, 2**-0.5, 1.0, 2**0.5, 2.0)]

    rows: str = report.config_rows({"mmd_sigmas_detector": sigmas}, None)

    assert r"\ConfigWide{mmd_sigmas_detector}" in rows
    assert "3.107" in rows
    assert r"\sqrt2" in rows


def test_sigmas_that_are_not_the_bracket_are_printed_in_full() -> None:
    """A future change to _SCALES must not make the report quietly lie."""
    rows: str = report.config_rows({"mmd_sigmas_detector": [1.0, 2.0, 3.0]}, None)
    assert "1" in rows and "2" in rows and "3" in rows
    assert r"\sqrt2" not in rows


def test_compile_cache_warmth_reaches_the_config_table() -> None:
    """It is what makes the `compile` timing interpretable at all."""
    rows: str = report.config_rows({"dim": 1}, {"compile_cache_warm": True})
    assert "compile_cache_warm" in rows


def test_timing_rows_indent_nested_phases_and_blank_their_share() -> None:
    """A nested phase is already inside its parent's share."""
    payload = {
        "total_seconds": 10.0,
        "phases": [
            {"name": "train", "seconds": 8.0, "depth": 0, "detail": None,
             "failed": False, "pass": "train"},
            {"name": "compile", "seconds": 4.0, "depth": 1, "detail": None,
             "failed": False, "pass": "train"},
        ],
    }

    rows: str = report.timing_rows(payload)

    assert r"\TimingSubphase{compile}" in rows
    assert "compile} & 4 &  &" in rows.replace("\\TimingSubphase{", "")
    assert "80" in rows  # train's share
    assert r"\midrule" in rows
    assert "10" in rows  # the total row


def test_the_pass_is_folded_into_the_detail_cell() -> None:
    """The template's timing table has four columns and no pass column."""
    payload = {
        "total_seconds": 1.0,
        "phases": [{"name": "data", "seconds": 1.0, "depth": 0,
                    "detail": "cache hit", "failed": False, "pass": "train"}],
    }
    assert "cache hit -- train" in report.timing_rows(payload)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k "config_rows or timing_rows or sigmas"`
Expected: FAIL — `AttributeError: module 'ran.report' has no attribute 'config_rows'`.

- [ ] **Step 3: Implement**

```python
_WIDE_KEYS: Final[frozenset[str]] = frozenset(
    {"variables", "mmd_sigmas_detector", "mmd_sigmas_particle", "gaussian_params"}
)


def _scalar_cell(value: object, /) -> str:
    if isinstance(value, bool):
        return latex_text(str(value))
    if isinstance(value, int):
        return rf"\Count{{{value}}}"
    if isinstance(value, float):
        return rf"\Raw{{{decimal(value)}}}"
    return rf"\ConfigVal{{{value}}}"


def _sigma_cell(sigmas: Sequence[float], /) -> str:
    """`median x (1/2 .. 2)` when the values really are the bracket.

    The five numbers are not free parameters: they are `mmd.median_bandwidth`
    of the data side, scaled by `mmd._SCALES`. Printed raw they are five opaque
    floats. Verified before collapsing, so a future change to the scale set
    falls back to printing them rather than mislabelling them.
    """
    if len(sigmas) == len(_SCALES):
        median: float = sigmas[_SCALES.index(1.0)]
        if all(
            s == pytest_approx_free(median * scale)
            for s, scale in zip(sigmas, _SCALES, strict=True)
        ):
            return (
                rf"\Raw{{{decimal(median)}}} "
                r"$\times\ (1/2,\ 1/\sqrt2,\ 1,\ \sqrt2,\ 2)$"
            )
    return ", ".join(rf"\Raw{{{decimal(s)}}}" for s in sigmas)
```

where `pytest_approx_free` is a plain tolerance helper defined in the module —
`math.isclose(s, median * scale, rel_tol=1e-6)` inline rather than a named
helper. Write it as:

```python
        if all(
            math.isclose(s, median * scale, rel_tol=1e-6)
            for s, scale in zip(sigmas, _SCALES, strict=True)
        ):
```

Then:

```python
def config_rows(
    config: Mapping[str, Any], timings: Mapping[str, Any] | None, /
) -> str:
    """`<<CONFIG_ROWS>>`: two key/value pairs per line, wide values spanning."""
    entries: dict[str, Any] = dict(config)
    if timings is not None and "compile_cache_warm" in timings:
        entries["compile_cache_warm"] = timings["compile_cache_warm"]

    scalars: list[tuple[str, Any]] = [
        (k, v) for k, v in entries.items() if k not in _WIDE_KEYS
    ]
    lines: list[str] = []
    for i in range(0, len(scalars), 2):
        pair: list[tuple[str, Any]] = scalars[i : i + 2]
        if len(pair) == 1:
            pair.append(("", ""))
        (k1, v1), (k2, v2) = pair
        c1: str = "" if k1 == "" else _scalar_cell(v1)
        c2: str = "" if k2 == "" else _scalar_cell(v2)
        lines.append(rf"\ConfigPair{{{k1}}}{{{c1}}}{{{k2}}}{{{c2}}}")

    if "variables" in entries:
        names: str = ", ".join(entries["variables"])
        lines.append(rf"\ConfigWide{{variables}}{{\ConfigVal{{{names}}}}}")
    for key in ("mmd_sigmas_detector", "mmd_sigmas_particle"):
        if key in entries:
            lines.append(rf"\ConfigWide{{{key}}}{{{_sigma_cell(entries[key])}}}")
    return "\n".join(lines)


def timing_rows(timings: Mapping[str, Any], /) -> str:
    """`<<TIMINGS_ROWS>>`, ending with a rule and a total."""
    total: float = float(timings["total_seconds"])
    lines: list[str] = []
    for phase in timings["phases"]:
        name: str = latex_text(phase["name"])
        if phase["depth"]:
            name = rf"\TimingSubphase{{{name}}}"
        # A nested phase is already inside its parent's share, so its cell is
        # empty rather than a number that would not sum to a hundred.
        share: str = (
            decimal(100.0 * phase["seconds"] / total)
            if phase["depth"] == 0 and total > 0
            else ""
        )
        parts: list[str] = [p for p in (phase.get("detail"), phase.get("pass")) if p]
        detail: str = latex_text(" -- ".join(parts))
        if phase.get("failed"):
            detail = rf"{detail} \textbf{{(raised)}}" if detail else r"\textbf{(raised)}"
        lines.append(f"{name} & {decimal(phase['seconds'])} & {share} & {detail} \\\\")
    lines.append(r"\midrule")
    lines.append(rf"\textbf{{total}} & {decimal(total)} & 100 & \\")
    return "\n".join(lines)
```

Import `math`, `Any`, `Mapping`, `Sequence`, and `from .mmd import _SCALES`.

- [ ] **Step 4: Run the tests**

Run: `just test -k report`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ran/report.py tests/test_report.py
git commit -m "feat: build the report's config and timing row bodies

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 12: Metrics tables, with grouping and IBU outcomes

**Files:**
- Modify: `src/ran/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `decimal` (Task 10); `JET_DISPLAY_ORDER`, `JET_VARIABLE_GROUPS`
  (Task 3); `artifacts/ibu_outcomes.json` (Task 8); `JET_OBS` for symbols.
- Produces:
  `metrics_table(level: str, variables: Sequence[str], ran: Mapping[str, Any],
  ibu: Mapping[str, Any] | None, skipped: frozenset[str], /) -> str`.

- [ ] **Step 1: Write the failing test**

```python
_METRIC_KEYS = ("wasserstein", "jensenshannon", "triangular")


def _entry(before: float, after: float) -> dict[str, float]:
    return {
        f"{m}_{suffix}": value
        for m in _METRIC_KEYS
        for suffix, value in (
            ("before", before),
            ("after", after),
            ("improvement_pct", (1 - after / before) * 100),
        )
    }


def test_rows_are_grouped_and_in_display_order() -> None:
    ran = {f"detector_{v}": _entry(1.0, 0.1) for v in SUBSTRUCTURE_VARIABLES}

    body: str = report.metrics_table(
        "detector", SUBSTRUCTURE_VARIABLES, ran, None, frozenset()
    )

    assert "Mass and hard scale" in body
    assert body.index("Mass and hard scale") < body.index("Continuous angularities")
    assert body.index(r"$\ln\rho$") < body.index(r"$\lambda^{1}_{0.5}$")
    assert body.count(r"\midrule") == 4  # one per group


def test_a_group_with_no_variables_is_omitted() -> None:
    """`--var m --var w` has nothing in the splitting group."""
    ran = {f"detector_{v}": _entry(1.0, 0.1) for v in ("m", "w")}

    body: str = report.metrics_table("detector", ("m", "w"), ran, None, frozenset())

    assert "Splitting" not in body
    assert body.count(r"\midrule") == 2


def test_a_missing_baseline_renders_dashes() -> None:
    """The template fixes sixteen columns, so the group cannot be omitted."""
    ran = {"detector_m": _entry(1.0, 0.1)}

    body: str = report.metrics_table("detector", ("m",), ran, None, frozenset())

    assert body.count(r"\multicolumn{1}{c}{---}") == 6  # 2 IBU cells x 3 metrics


def test_a_skipped_variable_is_daggered_rather_than_shown_as_zero() -> None:
    """IBU returning its input unchanged is a refusal, not a measurement."""
    ran = {"detector_zg": _entry(1.0, 0.1)}
    ibu = {"detector_zg": _entry(1.0, 1.0)}

    body: str = report.metrics_table(
        "detector", ("zg",), ran, ibu, frozenset({"zg"})
    )

    assert r"\dag" in body


def test_a_completed_variable_is_not_daggered() -> None:
    ran = {"detector_m": _entry(1.0, 0.1)}
    ibu = {"detector_m": _entry(1.0, 0.5)}

    body: str = report.metrics_table("detector", ("m",), ran, ibu, frozenset())

    assert r"\dag" not in body


def test_a_gaussian_run_has_rows_but_no_groups() -> None:
    ran = {f"detector_dim_{i}": _entry(1.0, 0.1) for i in range(2)}

    body: str = report.metrics_table(
        "detector", ("dim_0", "dim_1"), ran, None, frozenset()
    )

    assert "Mass and hard scale" not in body
    assert body.count(r"\\") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k metrics_table`
Expected: FAIL — `AttributeError: module 'ran.report' has no attribute 'metrics_table'`.

- [ ] **Step 3: Implement**

```python
_DASH: Final[str] = r"\multicolumn{1}{c}{---}"
_METRICS: Final[tuple[str, ...]] = ("wasserstein", "jensenshannon", "triangular")


def _row(
    variable: str,
    level: str,
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    daggered: bool,
    /,
) -> str:
    """One variable's sixteen cells: label, then Sim/IBU/IBU%/RAN/RAN% x 3."""
    label: str = JET_OBS[variable].symbol if variable in JET_OBS else latex_text(variable)
    if daggered:
        label = rf"{label}$^\dag$"
    ours: Mapping[str, float] = ran[f"{level}_{variable}"]
    theirs: Mapping[str, float] | None = (
        ibu.get(f"{level}_{variable}") if ibu is not None else None
    )

    cells: list[str] = [label]
    for metric in _METRICS:
        cells.append(decimal(ours[f"{metric}_before"]))
        if theirs is None:
            cells.extend((_DASH, _DASH))
        else:
            cells.append(decimal(theirs[f"{metric}_after"]))
            cells.append(decimal(theirs[f"{metric}_improvement_pct"]))
        cells.append(decimal(ours[f"{metric}_after"]))
        cells.append(decimal(ours[f"{metric}_improvement_pct"]))
    return " & ".join(cells) + r" \\"


def metrics_table(
    level: str,
    variables: Sequence[str],
    ran: Mapping[str, Any],
    ibu: Mapping[str, Any] | None,
    skipped: frozenset[str],
    /,
) -> str:
    """`<<DETECTOR_TABLE>>` / `<<PARTICLE_TABLE>>`: the row bodies only.

    The template owns the tabular, the column specification and the header;
    this owns the rules, the group headings and the data rows. `skipped` names
    the variables IBU gave up on, which are marked rather than shown as an
    honest-looking 0.0% improvement.
    """
    present: frozenset[str] = frozenset(variables)
    lines: list[str] = []
    for label, members in JET_VARIABLE_GROUPS:
        in_group: list[str] = [v for v in members if v in present]
        if not in_group:
            continue
        lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{16}}{{@{{}}l}}{{\itshape {label}}} \\")
        lines.extend(_row(v, level, ran, ibu, v in skipped) for v in in_group)

    if not lines:  # a non-jet run: rows, no grouping
        lines.append(r"\midrule")
        lines.extend(_row(v, level, ran, ibu, v in skipped) for v in variables)
    return "\n".join(lines)
```

Add a loader for the skip set:

```python
def skipped_variables(run_dir: Path, ibu: Mapping[str, Any] | None, /) -> frozenset[str]:
    """Variables IBU's purity binning refused, from the recorded outcomes.

    Falls back to the observable signature -- an `after` exactly equal to its
    `before` -- for a `metrics_ibu.json` written before outcomes were recorded.
    """
    path: Path = artifacts_dir(run_dir) / "ibu_outcomes.json"
    try:
        outcomes: list[dict[str, Any]] = json.loads(s=path.read_text())
    except (OSError, ValueError):
        if ibu is None:
            return frozenset()
        return frozenset(
            key.split(sep="_", maxsplit=1)[1]
            for key, entry in ibu.items()
            if entry["wasserstein_after"] == entry["wasserstein_before"]
        )
    return frozenset(o["variable_name"] for o in outcomes if o["status"] == "skipped")
```

- [ ] **Step 4: Run the tests**

Run: `just test -k report`
Expected: PASS.

- [ ] **Step 5: Check complexity**

Run: `just complexity`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/ran/report.py tests/test_report.py
git commit -m "feat: build the report's grouped metrics tables

Rows in physics display order, split into the four observable groups,
with the variables IBU's purity binning refused marked rather than shown
as a 0.0% improvement.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

### Task 13: `ran report` end to end

**Files:**
- Modify: `src/ran/report.py` (`render`, `build_report`)
- Modify: `src/ran/cli.py` (the `report` command)
- Modify: `scripts/submit.sh`
- Test: `tests/test_report.py`, `tests/test_cli.py`, `tests/test_command_targets.py`

**Interfaces:**
- Consumes: everything from Tasks 10-12.
- Produces:
  - `render(run_dir: Path, /) -> str` — the substituted LaTeX source.
  - `build_report(run_dir: Path, /, *, force: bool = False, compile_pdf: bool = True) -> Path`
    — writes `artifacts/report.tex`, compiles to `run_dir / "report.pdf"`,
    returns the path it produced.

- [ ] **Step 1: Write the failing test**

```python
def test_rendering_leaves_no_token_behind(reference_run: Path) -> None:
    assert report.TEMPLATE_TOKEN.findall(report.render(reference_run)) == []


def test_the_figure_paths_are_absolute(reference_run: Path) -> None:
    """`pdflatex` runs in `artifacts/`; a relative path would not resolve."""
    source: str = report.render(reference_run)
    assert str(reference_run.resolve()) in source


def test_a_run_directory_with_an_underscore_resolves(tmp_path: Path) -> None:
    """Sweep arms are named like `hp_x/lrg1e-4_seed03`."""
    run_dir: Path = _reference_run_at(tmp_path / "lrg1e-4_seed03")
    assert report.TEMPLATE_TOKEN.findall(report.render(run_dir)) == []


def test_no_cell_carries_an_exponent(reference_run: Path) -> None:
    """siunitx S columns cannot absorb one."""
    body: str = report.render(reference_run)
    rows = [l for l in body.splitlines() if l.rstrip().endswith(r"\\")]
    assert not any(re.search(r"\d[eE][+-]\d", row) for row in rows)


def test_a_missing_config_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="config.json"):
        _ = report.render(tmp_path)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no TeX installation")
def test_the_report_compiles(reference_run: Path) -> None:
    produced: Path = report.build_report(reference_run)

    assert produced == reference_run / "report.pdf"
    assert produced.stat().st_size > 0
    assert (reference_run / "artifacts/report.tex").exists()
    assert not list(reference_run.glob("*.aux"))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no TeX installation")
def test_a_run_without_a_baseline_still_compiles(reference_run: Path) -> None:
    (reference_run / "artifacts/metrics_ibu.json").unlink()
    assert report.build_report(reference_run, force=True).stat().st_size > 0


def test_no_compile_stops_at_the_source(reference_run: Path) -> None:
    produced: Path = report.build_report(reference_run, compile_pdf=False)
    assert produced == reference_run / "artifacts/report.tex"
```

Add a `reference_run` fixture to `tests/conftest.py` that builds a minimal but
complete run directory: `config.json` with `dataset="jets"` and two variables,
`artifacts/metrics.json`, `artifacts/metrics_ibu.json`,
`artifacts/ibu_outcomes.json`, `artifacts/timings.json`, and four one-page
placeholder PDFs written with matplotlib.

- [ ] **Step 2: Run test to verify it fails**

Run: `just test -k report`
Expected: FAIL — `AttributeError: module 'ran.report' has no attribute 'render'`.

- [ ] **Step 3: Implement `render`**

```python
def _read(path: Path, /) -> dict[str, Any] | None:
    try:
        return cast("dict[str, Any]", json.loads(s=path.read_text()))
    except (OSError, ValueError):
        return None


def render(run_dir: Path, /) -> str:
    """The substituted LaTeX source for one run."""
    config_path: Path = run_dir / "config.json"
    if not config_path.exists():
        msg: str = f"{config_path} does not exist: not a run directory"
        raise FileNotFoundError(msg)
    config: dict[str, Any] = cast("dict[str, Any]", json.loads(s=config_path.read_text()))
    artifacts: Path = artifacts_dir(run_dir)
    ran: dict[str, Any] = _read(artifacts / "metrics.json") or {}
    ibu: dict[str, Any] | None = _read(artifacts / "metrics_ibu.json")
    timings: dict[str, Any] | None = _read(artifacts / "timings.json")
    skipped: frozenset[str] = skipped_variables(run_dir, ibu)

    variables: tuple[str, ...] = tuple(
        config.get("variables") or [f"dim_{i}" for i in range(config["dim"])]
    )
    source: str = load_template()
    for token, value in (
        ("<<RUN_NAME>>", run_dir.name),
        ("<<CONFIG_ROWS>>", config_rows(config, timings)),
        ("<<TIMINGS_ROWS>>", timing_rows(timings) if timings else ""),
        ("<<DETECTOR_TABLE>>", metrics_table("detector", variables, ran, ibu, skipped)),
        ("<<PARTICLE_TABLE>>", metrics_table("particle", variables, ran, ibu, skipped)),
        # Absolute: `pdflatex` runs in `artifacts/`, and a sweep arm's
        # directory name can contain characters a relative path would not
        # survive.
        ("<<FIGURE_DIR>>", str(artifacts.resolve())),
    ):
        source = source.replace(token, value)

    left: list[str] = TEMPLATE_TOKEN.findall(source)
    if left:
        msg = f"template tokens with no value: {', '.join(sorted(set(left)))}"
        raise ValueError(msg)
    return source
```

Guard the metric tables: if `ran` is empty, substitute a single
`\midrule \multicolumn{16}{@{}l}{\itshape metrics.json not found} \\` row rather
than raising, so a run that died before `ran evaluate` still reports.

- [ ] **Step 4: Implement `build_report`**

```python
_LATEX_ARGS: Final[tuple[str, ...]] = (
    "pdflatex", "-interaction=nonstopmode", "-halt-on-error"
)
_AUX_SUFFIXES: Final[tuple[str, ...]] = (".aux", ".log", ".out")


def build_report(
    run_dir: Path, /, *, force: bool = False, compile_pdf: bool = True
) -> Path:
    """Write `artifacts/report.tex` and compile `report.pdf` at the run root."""
    pdf: Path = run_dir / "report.pdf"
    if pdf.exists() and not force and compile_pdf:
        logger.info("%s: report.pdf exists, skipping (use --force)", run_dir.name)
        return pdf

    artifacts: Path = artifacts_dir(run_dir)
    source: Path = artifacts / "report.tex"
    _ = source.write_text(data=render(run_dir), encoding="utf-8")
    if not compile_pdf:
        return source

    if shutil.which(cmd="pdflatex") is None:
        msg: str = (
            "pdflatex is not on PATH. Install a TeX distribution, or pass "
            "--no-compile to emit report.tex alone."
        )
        raise RuntimeError(msg)

    # Twice: the second pass settles \includegraphics box sizes and the page
    # counter. `-output-directory` puts the PDF at the run root while the
    # compile runs beside the figures it includes.
    for _pass in range(2):
        completed = subprocess.run(  # noqa: S603
            [*_LATEX_ARGS, f"-output-directory={run_dir}", source.name],
            cwd=artifacts,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            tail: str = "\n".join(completed.stdout.splitlines()[-40:])
            msg = f"pdflatex failed for {run_dir.name}:\n{tail}"
            raise RuntimeError(msg)

    for suffix in _AUX_SUFFIXES:
        (run_dir / f"report{suffix}").unlink(missing_ok=True)
    logger.info("%s: saved %s", run_dir.name, pdf)
    return pdf
```

Import `json`, `shutil`, `subprocess`, `cast`, `Path`, and `artifacts_dir`.

- [ ] **Step 5: Add the CLI command**

In `src/ran/cli.py`, after `evaluate_command`:

```python
@app.command(name="report")
def report_command(
    run_dir: Annotated[
        Path, typer.Argument(help="Run directory to report on.")
    ],
    force: Annotated[
        bool, typer.Option("--force", help="Rebuild an existing report.pdf.")
    ] = False,
    compile_pdf: Annotated[
        bool,
        typer.Option(
            "--compile/--no-compile", help="Compile the LaTeX, or stop at report.tex."
        ),
    ] = True,
) -> None:
    """Compile a run directory into one PDF dossier."""
    _ = build_report(run_dir, force=force, compile_pdf=compile_pdf)
```

Import `build_report` from `.report`. A positional argument, not a required
option: `ran report runs/2026-09-06T203848Z`.

- [ ] **Step 6: Add the command to the pipeline**

In `scripts/submit.sh`, after the final `ran evaluate`:

```bash
uv run ran report "$RUN_DIR"
```

The report is generated last, so it sees the IBU baseline, the redrawn figures
and the recomputed metrics.

- [ ] **Step 7: Run the full suite**

Run: `just validate`
Expected: PASS, including the `pdflatex` compile test on a machine with TeX.

- [ ] **Step 8: Update the docs**

`README.md` gains a `ran report` section beside `ran evaluate`; `CLAUDE.md`'s
Running section lists `report` among the subcommands and notes that
`scripts/submit.sh` ends with it; `overview.md`'s run-directory tree gains
`report.pdf`.

- [ ] **Step 9: Commit**

```bash
git add -A src/ran/report.py src/ran/cli.py scripts/submit.sh tests README.md CLAUDE.md overview.md
git commit -m "feat: add ran report, a one-PDF dossier for a run directory

Config, timings, both grouped metrics tables and the four figures, from
the JSON a run already writes. The JSON files are unchanged: this is a
read-only consumer.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HD6YqmwBJhsVGxqkNWfTo8"
```

---

## Self-Review

**Spec coverage.** §A → Task 1. §B → Task 2. §C.0 → Task 3. §C.1, C.2, C.6 →
Task 4. §C.3 → Task 5. §C.4 → Task 6. §C.5 → Task 7. §D → Task 8. §E → Task 9.
§F templating and primitives → Task 10; config/timings rows and the
`mmd_sigmas_*` collapse → Task 11; metrics tables, grouping and IBU outcomes →
Task 12; entry point, degradation, document structure and compilation → Task 13.
§G's test matrix is distributed across the tasks that own each behaviour.

**One spec addendum, discovered while planning.** The spec assumed
`IBUResult.outcomes` was on disk. It is not — `evaluate_single` persists only
`result.metrics` (`ibu.py:415`). Task 8 adds `artifacts/ibu_outcomes.json`, and
Task 12's `skipped_variables` keeps the before-equals-after fallback the spec
described, now as the fallback rather than the only path.

**Type consistency.** `artifacts_dir(run_dir: Path) -> Path` is used identically
in Tasks 1, 8, 9, 11, 12, 13. `display_order(variables) -> tuple[int, ...]` is
produced in Task 3 and consumed in Task 5. `decimal` and `latex_text` are
produced in Task 10 and consumed in 11 and 12. `metrics_table` keeps one
five-parameter signature between Task 12 and Task 13. `timing.write` gains
`pass_name` in Task 9 and every call site is updated in the same task.
