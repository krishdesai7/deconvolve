"""pytest imports conftest before collecting anything, so doing it here makes the
guarantee hold for every file, in any order, one file at a time or all of them.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import TYPE_CHECKING, Any, cast

import deconvolve  # ruff: ignore[unused-import]  -- imported for its backend bootstrap
import pytest
from deconvolve.coretypes import constants

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path


def _default_cache_is_writable() -> bool:
    """Whether the process can write RAN's default (non-`tmp_path`) cache dir.

    A few dataset/workflow tests build a `DeconvolveDataset` without an explicit
    `cache_dir`, so they generate into `constants.CACHE_DIR`. On a locked-down
    filesystem (sandboxed local runs, read-only `/.cache`) that write fails with
    `OSError`; on HPC, where these are meant to run, it succeeds. Probe once.
    """
    path = constants.CACHE_DIR
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".writable-probe-{uuid.uuid4().hex}"
        probe.touch()
        probe.unlink()
    except OSError:
        return False
    return True


_CACHE_WRITABLE = _default_cache_is_writable()


def pytest_configure(config: pytest.Config) -> None:
    # Marked by *cause*, not by stopwatch, so the set stays meaningful as the
    # hardware and the suite move: a test is slow here if it runs a training
    # program, shells out to `pdflatex`, or averages many random draws to
    # measure a statistical property. Those ~37 tests are ~55s of a ~76s
    # suite; the remaining ~500 cost about 20s together, which is what
    # `just test-fast` buys. Nothing is excluded by default -- `just test`,
    # `just validate` and CI all still run everything.
    config.addinivalue_line(
        "markers",
        "slow: trains a network, compiles a document, or averages many draws "
        "(deselect for a quick pass with `-m 'not slow'`, or `just test-fast`)",
    )
    config.addinivalue_line(
        "markers",
        "writes_default_cache: needs a writable default cache dir "
        "(skipped where the filesystem is read-only, e.g. local sandbox runs; "
        "force with DECONVOLVE_RUN_CACHE_TESTS=1)",
    )


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "writes_default_cache" not in item.keywords:
        return
    if _CACHE_WRITABLE or os.environ.get("DECONVOLVE_RUN_CACHE_TESTS"):
        return
    pytest.skip(f"default cache dir {constants.CACHE_DIR} is not writable")


@pytest.fixture(autouse=True)
def _no_global_config_layer(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Point `XDG_CONFIG_HOME` at an empty directory for every test.

    `deconvolve.cli`'s group callback (Task 4 of the layered-config plan) discovers
    `$XDG_CONFIG_HOME/deconvolve/deconvolve.toml` and folds it into
    `ctx.default_map`. Without this fixture, the suite would silently depend on
    whatever the developer (or CI runner) happens to have in their real
    `~/.config/deconvolve/deconvolve.toml` -- a test
    asserting the code default for `n-epochs` would pass or fail depending on
    machine state, not on the code under test. This fixture is autouse and runs
    before any test-specific `monkeypatch.setenv("XDG_CONFIG_HOME", ...)`
    (e.g. the `project` fixture in `tests/test_config_cli.py`), whose own call
    happens later in fixture setup and so still wins for that test.
    """
    monkeypatch.setenv(
        "XDG_CONFIG_HOME", str(tmp_path_factory.mktemp("no-global-config"))
    )


_METRIC_NAMES = ("wasserstein", "jensenshannon", "triangular")
_REFERENCE_VARIABLES = ("m", "M")
_FIGURES = ("detector_level", "particle_level", "losses", "selection")


def _metric_entry(before: float, after: float) -> dict[str, float]:
    return {
        f"{name}_{suffix}": value
        for name in _METRIC_NAMES
        for suffix, value in (
            ("before", before),
            ("after", after),
            ("improvement_pct", (1.0 - after / before) * 100.0),
        )
    }


def _metrics(variables: Sequence[str], before: float, after: float) -> dict[str, Any]:
    return {
        f"{level}_{variable}": _metric_entry(before, after)
        for level in ("detector", "particle")
        for variable in variables
    }


def _write_figures(artifacts: Path) -> None:
    r"""Four small one-page PDFs, so `\ReportGraphic` takes its real branch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in _FIGURES:
        figure = plt.figure(figsize=(4.0, 3.0))
        _ = figure.gca().plot([0.0, 1.0], [0.0, 1.0])
        figure.savefig(artifacts / f"{name}.pdf", format="pdf")
        plt.close(figure)


def _build_reference_run(
    run_dir: Path, /, config: dict[str, Any] | None = None
) -> Path:
    """A minimal but complete run directory: `config.json` plus `artifacts/`."""
    names: object = (config or {}).get("variables") or _REFERENCE_VARIABLES
    variables: tuple[str, ...] = tuple(cast("Sequence[str]", names))
    if config is None:
        config = {
            "batch_size": 1024,
            "n_samples": 4096,
            "dim": len(variables),
            "dataset": "jets",
            "seed": 7,
            "data_seed": 42,
            "n_epochs": 3,
            "lr_g": 3e-05,
            "best_epoch": 2,
            "mmd_sigmas_detector": [1.5 * s for s in (0.5, 2**-0.5, 1.0, 2**0.5, 2.0)],
            "variables": list(variables),
        }

    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    _ = (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    _ = (artifacts / "metrics.json").write_text(
        json.dumps(_metrics(variables, 0.20894503593444824, 8.294314e-05))
    )
    _ = (artifacts / "metrics_ibu.json").write_text(
        json.dumps(_metrics(variables, 0.20894503593444824, 0.05))
    )
    _ = (artifacts / "ibu_outcomes.json").write_text(
        json.dumps(
            [
                {"variable_name": name, "status": "unfolded", "n_bins": 12}
                for name in variables
            ]
        )
    )
    _ = (artifacts / "timings.json").write_text(
        json.dumps(
            {
                "total_seconds": 10.0,
                "compile_cache_warm": True,
                "phases": [
                    {
                        "name": "data",
                        "seconds": 2.0,
                        "depth": 0,
                        "detail": "cache hit",
                        "failed": False,
                        "pass": "train",
                    },
                    {
                        "name": "train",
                        "seconds": 8.0,
                        "depth": 0,
                        "detail": None,
                        "failed": False,
                        "pass": "train",
                    },
                    {
                        "name": "compile",
                        "seconds": 4.0,
                        "depth": 1,
                        "detail": None,
                        "failed": False,
                        "pass": "train",
                    },
                ],
            }
        )
    )
    _write_figures(artifacts)
    return run_dir


@pytest.fixture
def make_reference_run() -> Callable[..., Path]:
    """Build a reference run at a path of the test's choosing.

    A sweep arm's directory name contains an underscore and a Gaussian run
    carries a different config, so the location and the contents both have to
    be the caller's to pick.
    """
    return _build_reference_run


@pytest.fixture
def reference_run(tmp_path: Path) -> Path:
    """A complete jet run directory in the shipped layout."""
    return _build_reference_run(tmp_path / "2026-09-06T203848Z")
