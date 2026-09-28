# Unified Typer CLI and Rich Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace six Fire entry points and production built-in `print()` calls with one unified Typer command tree, standard-library logging rendered by `RichHandler`, and Rich-native progress and metric tables.

**Architecture:** `ran.cli` owns a root Typer app and nested baseline and sweep apps; every command lazily imports its domain function so the JAX and TensorFlow Keras backends remain isolated by process. `ran.logging_config` owns root logging setup, while domain modules expose ordinary callable APIs and module-level loggers. `ran.workflow` owns the former primary entry-point workflow, and `ran.leakage` owns the former repository script logic.

**Tech Stack:** Python 3.13, Typer 0.27.1 or newer, Rich via Typer, standard-library `logging`, pytest, Ruff, Pyrefly, Complexipy.

## Global Constraints

- Expose exactly `train`, `evaluate`, `baseline`, `sweep`, and `leakage-check` at the root.
- Expose exactly `omnifold` and `ibu` below `baseline`.
- Expose exactly `ran`, `omnifold`, and `collect` below `sweep`.
- Every Typer application uses `rich_markup_mode="rich"`.
- `ran.cli` must not import Keras, JAX, TensorFlow, `ran.workflow`, `ran.train`, `ran.evaluate`, or either baseline module at module import time.
- `baseline omnifold` and `sweep omnifold` must continue to start TensorFlow in a fresh process; all RAN commands continue to use JAX.
- Global logging defaults to `INFO` and is selected before the command with `--log-level`.
- Domain modules never configure the root logger during import.
- Logging calls use lazy `%`-style argument substitution, not f-strings.
- Download progress and metric tables use Rich presentation; operational events use logging.
- No built-in `print()` call or Fire import remains under `src/` or `scripts/`.
- Existing scientific calculations, saved artifact formats, and recoverable batch failure behavior remain unchanged.
- The user-provided Typer dependency changes in `pyproject.toml` and `uv.lock` are preserved.

---

### Task 1: Shared Rich Logging Configuration

**Files:**
- Create: `src/ran/logging_config.py`
- Create: `tests/test_logging_config.py`

**Interfaces:**
- Produces: `configure_logging(level: str = "INFO") -> None`
- Consumes: `rich.logging.RichHandler`, standard-library root logging

- [ ] **Step 1: Write failing logging configuration tests**

```python
import logging

import pytest
from rich.logging import RichHandler


@pytest.fixture(autouse=True)
def restore_root_logging():
    root = logging.getLogger()
    old_handlers = root.handlers[:]
    old_level = root.level
    yield
    root.handlers[:] = old_handlers
    root.setLevel(old_level)


def test_configure_logging_installs_one_rich_handler_and_level():
    from ran.logging_config import configure_logging

    configure_logging("debug")

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], RichHandler)


def test_configure_logging_is_deterministic_when_called_twice():
    from ran.logging_config import configure_logging

    configure_logging("INFO")
    configure_logging("WARNING")

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], RichHandler)


def test_configure_logging_rejects_unknown_level():
    from ran.logging_config import configure_logging

    with pytest.raises(ValueError, match="Unknown log level"):
        configure_logging("verbose")
```

- [ ] **Step 2: Run the tests and verify the expected import failure**

Run: `uv run pytest -q tests/test_logging_config.py`

Expected: FAIL because `ran.logging_config` does not exist.

- [ ] **Step 3: Implement the minimal shared configuration**

Create `src/ran/logging_config.py` with this behavior:

```python
import logging

from rich.console import Console
from rich.logging import RichHandler


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging with Rich terminal rendering."""
    normalized = level.upper()
    numeric_level = logging.getLevelNamesMapping().get(normalized)
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
```

- [ ] **Step 4: Run the focused tests and confirm they pass**

Run: `uv run pytest -q tests/test_logging_config.py`

Expected: 3 passed.

- [ ] **Step 5: Commit the logging configuration**

```bash
git add src/ran/logging_config.py tests/test_logging_config.py
git commit -m "feat: configure Rich application logging"
```

---

### Task 2: Extract Domain Workflows from CLI Entry Points

**Files:**
- Create: `src/ran/workflow.py`
- Create: `src/ran/leakage.py`
- Create: `tests/test_command_targets.py`
- Modify: `src/ran/evaluate.py`
- Modify: `src/ran/baselines/omnifold.py`
- Modify: `src/ran/baselines/ibu.py`
- Modify: `src/ran/experiments/cubic_sweep.py`
- Delete: `scripts/leakage_check.py`

**Interfaces:**
- Produces: `ran.workflow.run`, with its complete public signature specified in Step 3
- Produces: `ran.leakage.run_leakage_check(poison: bool = False, seed: int = 42, init_seed: int = 0) -> None`
- Produces: `ran.evaluate.evaluate_runs(run_dir: str | Path = "runs", force: bool = False) -> None`
- Produces: `ran.baselines.omnifold.evaluate_runs(run_dir: str | Path = "runs", force: bool = False, niter: int = 3, epochs: int = 50) -> None`
- Produces: `ran.baselines.ibu.evaluate_runs(run_dir: str | Path = "runs", force: bool = False, n_iterations: int = 10, purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD) -> None`
- Preserves: `run_ran`, `run_omnifold`, and `collect` in `ran.experiments.cubic_sweep`

- [ ] **Step 1: Write failing import-boundary tests**

```python
def test_primary_workflow_has_a_domain_entry_point():
    from ran.workflow import run

    assert callable(run)


def test_leakage_check_has_a_package_entry_point():
    from ran.leakage import run_leakage_check

    assert callable(run_leakage_check)


def test_batch_orchestrators_have_domain_names():
    from ran.baselines.ibu import evaluate_runs as evaluate_ibu_runs
    from ran.evaluate import evaluate_runs

    assert callable(evaluate_runs)
    assert callable(evaluate_ibu_runs)
```

Do not import `ran.baselines.omnifold` in this test process because the rest of the test suite uses JAX. OmniFold command registration is exercised without importing its implementation in Task 3.

- [ ] **Step 2: Run the tests and verify missing-module/name failures**

Run: `uv run pytest -q tests/test_command_targets.py`

Expected: FAIL because `ran.workflow`, `ran.leakage`, and the renamed orchestrators do not exist.

- [ ] **Step 3: Move the primary workflow without changing behavior**

Move the imports and body of `src/ran/__main__.py:23-208` into `src/ran/workflow.py` and rename `main` to `run`. Keep this exact public signature:

```python
def run(
    batch_size: int = 1024,
    n_samples: int = 500_000,
    config: str | None = None,
    dataset: str = "gaussian",
    variables: tuple[str, ...] = ("m", "M", "w", "tau21", "zg", "sdm"),
    load_run: str | None = None,
    hidden_units: int = 64,
    n_layers: int = 2,
    patience: int = 5,
    seed: int | None = None,
    data_seed: int = 42,
) -> None:
```

Do not alter calculations, file names, model saving, seed handling, plotting, or metric evaluation in this task.

- [ ] **Step 4: Move the leakage workflow into the package**

Create `src/ran/leakage.py` from `scripts/leakage_check.py`, rename `run` to `run_leakage_check`, give `poison` a default of `False`, remove the Fire import and module entry-point block, and delete `scripts/leakage_check.py`.

- [ ] **Step 5: Rename batch orchestration functions and remove Fire shells**

Apply these exact renames:

```python
# ran.evaluate
def evaluate_runs(run_dir: str | Path = "runs", force: bool = False) -> None:

# ran.baselines.omnifold
def evaluate_runs(
    run_dir: str | Path = "runs",
    force: bool = False,
    niter: int = 3,
    epochs: int = 50,
) -> None:

# ran.baselines.ibu
DEFAULT_PURITY_THRESHOLD = np.sqrt(0.5, dtype=np.double)

def evaluate_runs(
    run_dir: str | Path = "runs",
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> None:
```

Remove Fire imports and `if __name__ == "__main__"` CLI blocks from evaluation, both baselines, and cubic sweep. Leave all prints in place until the logging task so this task is a pure boundary refactor.

- [ ] **Step 6: Run the focused and existing domain tests**

Run: `uv run pytest -q tests/test_command_targets.py tests/test_cubic_sweep.py tests/test_train.py tests/test_datasets.py`

Expected: all selected tests pass.

- [ ] **Step 7: Commit the domain extraction**

```bash
git add src/ran/workflow.py src/ran/leakage.py src/ran/evaluate.py src/ran/baselines/omnifold.py src/ran/baselines/ibu.py src/ran/experiments/cubic_sweep.py tests/test_command_targets.py scripts/leakage_check.py
git commit -m "refactor: extract CLI domain workflows"
```

---

### Task 3: Build the Unified Lazy-Import Typer Command Tree

**Files:**
- Create: `src/ran/cli.py`
- Create: `tests/test_cli.py`
- Replace: `src/ran/__main__.py`

**Interfaces:**
- Produces: `app: typer.Typer`, `baseline_app: typer.Typer`, `sweep_app: typer.Typer`
- Consumes: all domain functions produced by Task 2 through imports inside command functions only
- Consumes: `configure_logging` from Task 1

- [ ] **Step 1: Write failing root and nested help tests**

```python
import subprocess
import sys

from typer.testing import CliRunner

import ran.cli as cli
from ran.cli import app

runner = CliRunner()


def test_root_help_lists_unified_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("train", "evaluate", "baseline", "sweep", "leakage-check"):
        assert command in result.stdout


def test_baseline_help_lists_both_methods():
    result = runner.invoke(app, ["baseline", "--help"])
    assert result.exit_code == 0
    assert "omnifold" in result.stdout
    assert "ibu" in result.stdout


def test_sweep_help_lists_all_actions():
    result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    for command in ("ran", "omnifold", "collect"):
        assert command in result.stdout


def test_importing_cli_does_not_commit_a_keras_backend():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import ran.cli; assert 'keras' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
```

- [ ] **Step 2: Write a failing option-forwarding test for the train command**

Append this test to `tests/test_cli.py`:

```python
from types import ModuleType


def test_train_converts_typer_values_for_the_workflow(monkeypatch, tmp_path):
    calls = []
    configured_levels = []
    fake_workflow = ModuleType("ran.workflow")
    fake_workflow.run = lambda **kwargs: calls.append(kwargs)
    monkeypatch.setitem(sys.modules, "ran.workflow", fake_workflow)
    monkeypatch.setattr(cli, "configure_logging", configured_levels.append)

    result = runner.invoke(
        app,
        [
            "--log-level",
            "warning",
            "train",
            "--dataset",
            "jets",
            "--variable",
            "m",
            "--variable",
            "w",
            "--load-run",
            str(tmp_path),
            "--seed",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert calls[0]["dataset"] == "jets"
    assert calls[0]["variables"] == ("m", "w")
    assert calls[0]["load_run"] == str(tmp_path)
    assert calls[0]["seed"] == 7
    assert configured_levels == ["WARNING"]
```

- [ ] **Step 3: Run CLI tests and verify the missing-app failure**

Run: `uv run pytest -q tests/test_cli.py`

Expected: FAIL because `ran.cli` does not exist.

- [ ] **Step 4: Implement application objects, global logging, and lazy commands**

Create `src/ran/cli.py` with only standard-library, Typer, and logging configuration imports at module scope. Define:

```python
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from ran.logging_config import configure_logging

DEFAULT_VARIABLES = ("m", "M", "w", "tau21", "zg", "sdm")

app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
baseline_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
sweep_app = typer.Typer(rich_markup_mode="rich", no_args_is_help=True)
app.add_typer(baseline_app, name="baseline", help="Run comparison baselines.")
app.add_typer(sweep_app, name="sweep", help="Run cubic-response sweep steps.")


class LogLevel(StrEnum):
    debug = "DEBUG"
    info = "INFO"
    warning = "WARNING"
    error = "ERROR"
    critical = "CRITICAL"


class DatasetName(StrEnum):
    gaussian = "gaussian"
    jets = "jets"


@app.callback()
def configure(
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", case_sensitive=False, help="Application log level."),
    ] = LogLevel.info,
) -> None:
    configure_logging(log_level.value)
```

Add decorated wrappers named `train_command`, `evaluate_command`, `omnifold_command`, `ibu_command`, `sweep_ran_command`, `sweep_omnifold_command`, `sweep_collect_command`, and `leakage_check_command`. Each wrapper imports its target inside the function and forwards every existing option with Typer's hyphenated option names.

The train wrapper must normalize its CLI-only values exactly as follows:

```python
@app.command("train")
def train_command(
    batch_size: Annotated[int, typer.Option(min=1)] = 1024,
    n_samples: Annotated[int, typer.Option(min=1)] = 500_000,
    config: Path | None = None,
    dataset: DatasetName = DatasetName.gaussian,
    variable: Annotated[list[str] | None, typer.Option("--variable")] = None,
    load_run: Annotated[Path | None, typer.Option()] = None,
    hidden_units: Annotated[int, typer.Option(min=1)] = 64,
    n_layers: Annotated[int, typer.Option(min=1)] = 2,
    patience: Annotated[int, typer.Option(min=1)] = 5,
    seed: int | None = None,
    data_seed: int = 42,
) -> None:
    from ran.workflow import run

    run(
        batch_size=batch_size,
        n_samples=n_samples,
        config=str(config) if config is not None else None,
        dataset=dataset.value,
        variables=tuple(variable or DEFAULT_VARIABLES),
        load_run=str(load_run) if load_run is not None else None,
        hidden_units=hidden_units,
        n_layers=n_layers,
        patience=patience,
        seed=seed,
        data_seed=data_seed,
    )
```

Use `Path("runs")` as the default for evaluation and both baselines. Preserve OmniFold defaults `niter=3`, `epochs=50`; preserve IBU defaults `n_iterations=10`, `purity_threshold=sqrt(0.5)` using a CLI float literal `0.7071067811865476`. Keep `s_index` and `sweep_dir` required for the two per-point sweep commands and keep `sweep_dir` required for collect. Define leakage as `--poison/--clean` with clean as the default.

- [ ] **Step 5: Replace the module entry point**

Replace `src/ran/__main__.py` with:

```python
from ran.cli import app


if __name__ == "__main__":
    app()
```

- [ ] **Step 6: Exercise help for every command**

Run:

```bash
uv run -m ran --help
uv run -m ran train --help
uv run -m ran evaluate --help
uv run -m ran baseline omnifold --help
uv run -m ran baseline ibu --help
uv run -m ran sweep ran --help
uv run -m ran sweep omnifold --help
uv run -m ran sweep collect --help
uv run -m ran leakage-check --help
```

Expected: every invocation exits 0 and displays Rich-formatted help without importing Keras for root help.

- [ ] **Step 7: Run CLI and domain tests**

Run: `uv run pytest -q tests/test_cli.py tests/test_command_targets.py`

Expected: all focused tests pass.

- [ ] **Step 8: Commit the unified CLI**

```bash
git add src/ran/cli.py src/ran/__main__.py tests/test_cli.py
git commit -m "feat: add unified Typer command tree"
```

---

### Task 4: Replace Built-In Prints with Semantic Logging

**Files:**
- Modify: `tests/test_cubic_sweep.py`
- Modify: `src/ran/workflow.py`
- Modify: `src/ran/leakage.py`
- Modify: `src/ran/train.py`
- Modify: `src/ran/evaluate.py`
- Modify: `src/ran/baselines/omnifold.py`
- Modify: `src/ran/baselines/ibu.py`
- Modify: `src/ran/data/datasets.py`
- Modify: `src/ran/data/jets.py`
- Modify: `src/ran/data/download.py`
- Modify: `src/ran/plotting.py`
- Modify: `src/ran/experiments/cubic_sweep.py`

**Interfaces:**
- Every reporting module owns `logger = logging.getLogger(__name__)`
- Existing return values and exceptions remain unchanged

- [ ] **Step 1: Convert the cubic-sweep output assertion to a logging assertion**

Replace `capsys` with `caplog`, set the warning level, and assert against records:

```python
def test_collect_skips_points_missing_one_method(tmp_path, caplog):
    from ran.experiments.cubic_sweep import collect

    _write_points(tmp_path, [0], [0.0])
    _write_points(tmp_path, [1], [10.0], omnifold=False)

    with caplog.at_level("WARNING"):
        collect(sweep_dir=tmp_path, n_points=2)

    data = np.load(tmp_path / "results.npz")
    np.testing.assert_array_equal(data["s"], [0.0])
    assert "missing s_index values" in caplog.text
```

- [ ] **Step 2: Run the test and confirm the warning is not logged yet**

Run: `uv run pytest -q tests/test_cubic_sweep.py::test_collect_skips_points_missing_one_method`

Expected: FAIL because `caplog.text` does not contain the printed warning.

- [ ] **Step 3: Add module loggers and convert operational messages**

Use these exact severity mappings:

| Module and event | Level |
|---|---|
| `workflow`: run/baseline weights loaded, run saved | INFO |
| `workflow`: final metric calculation caught failure | `logger.exception` |
| `train`: epoch, early stop, final test losses | INFO |
| `evaluate`: dataset/run loading, skip-existing, run count | INFO |
| `evaluate`: caught per-run failure | WARNING with `exc_info=True` |
| `omnifold` and `ibu`: skip-existing, start, run count, per-variable progress | INFO |
| `omnifold` and `ibu`: caught per-run failure or unusable dimension | WARNING |
| `datasets`, `jets`: cache, generation, and download-start milestones | INFO |
| `plotting`: saved plot path | INFO |
| `cubic_sweep`: point results and written artifacts | INFO |
| `cubic_sweep`: missing points | WARNING |
| `leakage`: run mode and metric comparisons | INFO |

Convert interpolated logging messages to lazy arguments. For example:

```python
logger.info(
    "Epoch %3d/%d  D: %.4f  G: %.4f  | Val D: %.4f  G: %.4f  (patience %d/%d)",
    epoch + 1,
    n_epochs,
    mean_td,
    mean_tg,
    mean_val[0],
    mean_val[1],
    wait,
    patience,
)

try:
    evaluate_run(run_dir, force=load_run is None)
except Exception:
    logger.exception("Metric evaluation failed")
```

Do not convert metric-table or download reporthook prints to logs in this step; Task 5 replaces those with Rich presentation. The source hygiene test is expected to keep failing only for those presentation calls until Task 5.

- [ ] **Step 4: Run Ruff's focused logging rules**

Run: `uv run ruff check --select LOG,T20 src tests`

Expected: only the metric renderer and download reporthook remain as T20 failures; there are no LOG failures in converted modules.

- [ ] **Step 5: Run affected behavior tests**

Run: `uv run pytest -q tests/test_cubic_sweep.py tests/test_train.py tests/test_datasets.py`

Expected: all selected tests pass.

- [ ] **Step 6: Commit semantic logging conversions**

```bash
git add src/ran/workflow.py src/ran/leakage.py src/ran/train.py src/ran/evaluate.py src/ran/baselines/omnifold.py src/ran/baselines/ibu.py src/ran/data/datasets.py src/ran/data/jets.py src/ran/plotting.py src/ran/experiments/cubic_sweep.py tests/test_cubic_sweep.py
git commit -m "refactor: replace operational prints with logging"
```

---

### Task 5: Replace Presentation Prints with Rich Tables and Progress

**Files:**
- Create: `tests/test_console_output.py`
- Create: `tests/test_source_hygiene.py`
- Modify: `src/ran/evaluate.py`
- Modify: `src/ran/baselines/omnifold.py`
- Modify: `src/ran/baselines/ibu.py`
- Modify: `src/ran/data/download.py`

**Interfaces:**
- Produces: `render_metrics(run_name: str, metrics: dict, var_names: list[str], console: Console | None = None) -> None`
- Produces: download reporthook updates a Rich `Progress` task without calling built-in print
- Replaces: `_print_metrics` imports in both baselines with `render_metrics`
- Produces a source-level invariant: no `ast.Name(id="print")` call and no Fire import under `src/` or `scripts/`

- [ ] **Step 1: Write a failing Rich metric-table test**

```python
from io import StringIO

from rich.console import Console


def test_render_metrics_outputs_named_columns_and_values():
    from ran.evaluate import render_metrics

    output = StringIO()
    console = Console(file=output, color_system=None, width=120)
    metrics = {
        "detector_dim_0": {
            "wasserstein_before": 1.0,
            "wasserstein_after": 0.25,
            "wasserstein_improvement_pct": 75.0,
            "jensenshannon_before": 0.2,
            "jensenshannon_after": 0.1,
            "jensenshannon_improvement_pct": 50.0,
            "triangular_before": 8.0,
            "triangular_after": 2.0,
            "triangular_improvement_pct": 75.0,
        }
    }

    render_metrics("sample-run", metrics, ["dim_0"], console=console)

    rendered = output.getvalue()
    assert "sample-run" in rendered
    assert "Wasserstein" in rendered
    assert "Before" in rendered
    assert "After" in rendered
    assert "75.0%" in rendered
```

- [ ] **Step 2: Write a failing download progress test with no network access**

```python
from rich.progress import Progress


def test_download_file_updates_a_rich_progress_task(monkeypatch, tmp_path):
    import ran.data.download as download

    completed = []

    def fake_urlretrieve(url, dest, reporthook):
        reporthook(0, 25, 100)
        reporthook(2, 25, 100)
        reporthook(4, 25, 100)
        completed.append((url, dest))

    monkeypatch.setattr(download.urllib.request, "urlretrieve", fake_urlretrieve)
    progress = Progress(disable=True)
    task_id = progress.add_task("sample.npz", total=None)

    download._download_file(
        "https://example.test/sample.npz",
        tmp_path / "sample.npz",
        progress,
        task_id,
    )

    assert completed == [
        ("https://example.test/sample.npz", tmp_path / "sample.npz")
    ]
    assert progress.tasks[0].completed == 100
    assert progress.tasks[0].total == 100
```

- [ ] **Step 3: Write the failing source hygiene test**

```python
import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]


def test_production_python_uses_neither_builtin_print_nor_fire():
    offenders = []
    for source_root in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "print"
                ):
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: print")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "fire":
                            offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: fire")
                if isinstance(node, ast.ImportFrom) and node.module == "fire":
                    offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}: fire")
    assert offenders == []
```

- [ ] **Step 4: Run the console and hygiene tests and verify failures**

Run: `uv run pytest -q tests/test_console_output.py tests/test_source_hygiene.py`

Expected: FAIL because `render_metrics` and the Rich-aware `_download_file` signature do not exist and the presentation functions still use built-in print.

- [ ] **Step 5: Implement metric tables**

Rename `_print_metrics` to `render_metrics`. Build one `rich.table.Table` per available level with title `f"{run_name} — {level.title()} level"` and columns `Variable`, `Metric`, `Before`, `After`, and `Improvement`. Add three rows per variable using these formats:

```python
(var, "Wasserstein", f"{before:.4f}", f"{after:.4f}", f"{improvement:+.1f}%")
("", "JS div", f"{before:.6f}", f"{after:.6f}", f"{improvement:+.1f}%")
("", "Delta (x1e3)", f"{before:.4f}", f"{after:.4f}", f"{improvement:+.1f}%")
```

Use `active_console = console or Console()` and `active_console.print(table)`. Update evaluation and both baseline callers/imports to use `render_metrics`.

- [ ] **Step 6: Implement Rich download progress**

Change `_download_file` to accept `progress: Progress` and `task_id: TaskID`. Its reporthook sets the task total when `total_size > 0` and updates completed bytes with `min(block_num * block_size, total_size)`.

In `download_jet_data`, create one `Progress` context around both generator loops. Add a task for each missing file, call `_download_file`, and remove or hide the task after completion. Convert section headings, already-downloaded files, extraction milestones, saved cache files, cleanup, and completion to logger events. Do not emit a logger record for every reporthook call.

- [ ] **Step 7: Verify all presentation and source-hygiene tests pass**

Run: `uv run pytest -q tests/test_console_output.py tests/test_source_hygiene.py`

Expected: all tests pass.

- [ ] **Step 8: Verify focused lint rules are clean**

Run: `uv run ruff check --select LOG,T20 src tests`

Expected: exit 0 with no logging or built-in-print violations.

- [ ] **Step 9: Commit Rich presentation**

```bash
git add src/ran/evaluate.py src/ran/baselines/omnifold.py src/ran/baselines/ibu.py src/ran/data/download.py tests/test_console_output.py tests/test_source_hygiene.py
git commit -m "feat: render metrics and downloads with Rich"
```

---

### Task 6: Update Documentation and SLURM Commands

**Files:**
- Modify: `README.md`
- Modify: `scripts/submit.sh`
- Modify: `scripts/submit_sweep.sh`
- Modify: module docstrings in `src/ran/evaluate.py`, `src/ran/baselines/omnifold.py`, `src/ran/baselines/ibu.py`, and `src/ran/experiments/cubic_sweep.py`

**Interfaces:**
- Documents and invokes only the unified `uv run -m ran` command tree
- Preserves separate processes for RAN and OmniFold sweep steps

- [ ] **Step 1: Record stale command references before editing**

Run:

```bash
rg -n "uv run -m ran\.(evaluate|baselines|experiments)|uv run -m ran --(config|dataset|load)|scripts/leakage_check|--[a-z]+_[a-z]+|\bFire\b" README.md scripts src/ran
```

Expected: matches show the old module paths, implicit train command, underscore options, leakage script, and Fire references that need replacement.

- [ ] **Step 2: Update primary usage and the option table**

Use `uv run -m ran train` for every training or reload example. Replace underscore option spellings with Typer's hyphenated spellings. Replace tuple-literal jet variables with:

```bash
uv run -m ran train --dataset jets --variable m --variable w
```

Document the global log level with:

```bash
uv run -m ran --log-level DEBUG train --config params/1d_default.yaml
```

- [ ] **Step 3: Update evaluation, baselines, leakage, and project structure**

Use these canonical command shapes:

```bash
uv run -m ran evaluate --run-dir runs/2026-03-14T061023Z
uv run -m ran baseline omnifold --run-dir runs/2026-03-14T061023Z
uv run -m ran baseline ibu --run-dir runs/2026-03-14T061023Z
uv run -m ran leakage-check --clean
uv run -m ran leakage-check --poison
```

Add `cli.py`, `workflow.py`, `logging_config.py`, and `leakage.py` to the project tree; remove `scripts/leakage_check.py`; replace Fire with Typer and Rich in the dependency list.

Add a cubic sweep subsection documenting the three unified commands:

```bash
uv run -m ran sweep ran --s-index 0 --sweep-dir runs/cubic-sweep
uv run -m ran sweep omnifold --s-index 0 --sweep-dir runs/cubic-sweep
uv run -m ran sweep collect --sweep-dir runs/cubic-sweep
```

- [ ] **Step 4: Update SLURM scripts while preserving process isolation**

In `scripts/submit.sh`, run:

```bash
uv run -m ran train "$@"
uv run -m ran baseline omnifold --run-dir="${LATEST_RUN}"
```

In `scripts/submit_sweep.sh`, update the two commands inside each `srun` process and the final collection command to:

```bash
uv run -m ran sweep ran --s-index='${i}' --sweep-dir='${SWEEP_DIR}' --n-points='${N_POINTS}'
uv run -m ran sweep omnifold --s-index='${i}' --sweep-dir='${SWEEP_DIR}' --n-points='${N_POINTS}'
uv run -m ran sweep collect --sweep-dir="${SWEEP_DIR}" --n-points="${N_POINTS}"
```

- [ ] **Step 5: Confirm no stale CLI references remain**

Run:

```bash
rg -n "uv run -m ran\.(evaluate|baselines|experiments)|uv run -m ran --(config|dataset|load)|scripts/leakage_check|--[a-z]+_[a-z]+|\bFire\b" README.md scripts src/ran
```

Expected: no matches. Mentions of the general word `fire` in unrelated prose are not introduced.

- [ ] **Step 6: Smoke-test every documented command's help path**

Run: `uv run -m ran --help && uv run -m ran train --help && uv run -m ran evaluate --help && uv run -m ran baseline --help && uv run -m ran sweep --help && uv run -m ran leakage-check --help`

Expected: exit 0.

- [ ] **Step 7: Commit documentation and job scripts**

```bash
git add README.md scripts/submit.sh scripts/submit_sweep.sh src/ran/evaluate.py src/ran/baselines/omnifold.py src/ran/baselines/ibu.py src/ran/experiments/cubic_sweep.py
git commit -m "docs: document unified command tree"
```

---

### Task 7: Full Verification and Regression Audit

**Files:**
- Modify only files implicated by failures introduced by Tasks 1-6

**Interfaces:**
- Verifies the approved design end to end
- Does not broaden scope to unrelated pre-existing lint or type issues

- [ ] **Step 1: Synchronize against the committed lock file**

Run: `uv sync --locked`

Expected: exit 0 without modifying `pyproject.toml` or `uv.lock`.

- [ ] **Step 2: Run the complete automated test suite**

Run: `uv run pytest -q`

Expected: all regular tests pass; the explicitly opt-in real OmniFold training test remains skipped unless `RAN_RUN_SLOW=1` is set.

- [ ] **Step 3: Run scoped logging and CLI lint verification**

Run: `uv run ruff check --select LOG,T20 src tests`

Expected: exit 0.

- [ ] **Step 4: Run all repository validation commands**

Run each command independently and record its exit code:

```bash
uv format --check
uv run ruff check
uv run pyrefly check
uv run complexipy
```

Expected for this refactor: no new failures in changed files. The repository already had unrelated lint findings before this work; distinguish those existing findings from regressions rather than silently claiming a globally clean baseline.

- [ ] **Step 5: Re-run CLI and backend-isolation smoke tests**

Run:

```bash
uv run -m ran --help
uv run -m ran baseline omnifold --help
uv run -m ran sweep omnifold --help
uv run python -c "import sys; import ran.cli; assert 'keras' not in sys.modules"
```

Expected: all commands exit 0 and importing `ran.cli` does not import Keras.

- [ ] **Step 6: Audit the final tree and diff**

Run:

```bash
git diff --check
git status --short
rg -n "(^|\s)(from fire import|import fire|print\()" src scripts
```

Expected: no whitespace errors; only intended implementation files are changed; the final search has no matches.

- [ ] **Step 7: Review requirements line by line**

Confirm from fresh command output and source inspection:

- Root and nested command names match the approved tree.
- All three Typer apps use Rich markup mode.
- All command implementations import backend-sensitive modules lazily.
- Root logging uses one RichHandler and honors `--log-level`.
- Operational events have appropriate severity and lazy formatting.
- Metric output and download progress use Rich presentation.
- Fire, the old leakage script, and built-in production prints are gone.
- README and both SLURM scripts use unified commands.
- Scientific return values, artifacts, and tests remain intact.

- [ ] **Step 8: Commit any verification-only corrections**

If verification required corrections, stage only those implicated files and commit them with:

```bash
git commit -m "fix: resolve unified CLI verification findings"
```

If no corrections were required, do not create an empty commit.
