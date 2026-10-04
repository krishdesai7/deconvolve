"""Tests for `deconvolve plot`, the redraw that picks up the baselines.

Nothing here draws: the plotting functions are stubbed, as in
`tests/test_workflow.py`, so what is exercised is the reload -- config parsing,
the dataset rebuild -- and which overlays reach the figures.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from deconvolve.data import parse_gaussian_config
from deconvolve.workflows import plot as plot_workflow
from deconvolve.workflows import train as train_workflow

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any, Final

    from deconvolve.coretypes import GaussianConfig
    from deconvolve.evaluation import BaselineOverlay
    from numpy.typing import NDArray

CONFIG_2D: Final[str] = """
mu_gen: [0.0, 1.0]
mu_true: [0.2, 0.8]
sigma_gen:
  - [1.0, -0.9]
  - [-0.9, 2.25]
sigma_true:
  - [0.81, -0.702]
  - [-0.702, 1.69]
sigma_detector: [0.5, 0.8]
"""


@pytest.fixture
def drawn(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Stub the model load and every figure, recording what they were given."""
    calls: dict[str, list[Any]] = {"levels": [], "losses": [], "loaded": []}

    def fake_load_artifacts(run_dir: Path) -> tuple[Any, dict[str, list[float]]]:
        calls["loaded"].append(run_dir)
        return (lambda z: np.ones(shape=(len(z), 1))), {"train_d": [0.7]}

    def fake_plot_levels(*_args: object, **kwargs: Any) -> None:
        calls["levels"].append(kwargs)

    def fake_plot_losses(*_args: object, save_path: Path, **_kwargs: object) -> None:
        calls["losses"].append(save_path)

    def no_evaluate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("plot must not rescore the run")

    monkeypatch.setattr(plot_workflow, "_load_artifacts", fake_load_artifacts)
    monkeypatch.setattr(train_workflow, "plot_levels", fake_plot_levels)
    monkeypatch.setattr(train_workflow, "plot_losses", fake_plot_losses)
    monkeypatch.setattr(train_workflow, "evaluate_run", no_evaluate)
    return calls


def _write_run(root: Path, name: str = "2026-10-04T022859Z") -> Path:
    _ = (root / "cfg.yaml").write_text(data=CONFIG_2D)
    params: GaussianConfig = parse_gaussian_config(config_path=root / "cfg.yaml")
    run_dir: Path = root / "runs" / name
    (run_dir / "artifacts").mkdir(parents=True)
    config: dict[str, Any] = {
        "batch_size": 64,
        "n_samples": 400,
        "dim": 2,
        "dataset": "gaussian",
        "seed": 0,
        "data_seed": 42,
        "gaussian_params": params.model_dump(),
    }
    _ = (run_dir / "config.json").write_text(data=json.dumps(obj=config))
    return run_dir


def _write_baselines(run_dir: Path, n: int = 10) -> None:
    ones: NDArray[np.single] = np.ones(n, dtype=np.single)
    np.savez(run_dir / "artifacts" / "ibu_weights.npz", weights_0=ones, weights_1=ones)
    np.savez(run_dir / "artifacts" / "omnifold_weights.npz", weights=ones)


@pytest.mark.writes_default_cache
def test_a_fresh_run_redraws_without_overlays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drawn: dict[str, list[Any]]
) -> None:
    monkeypatch.chdir(tmp_path)
    run_dir: Path = _write_run(tmp_path)

    plot_workflow.plot_run(run_dir)

    assert drawn["loaded"] == [run_dir]
    assert len(drawn["levels"]) == 1
    assert drawn["levels"][0]["baselines"] == []
    assert drawn["levels"][0]["detector_path"] == (
        run_dir / "artifacts" / "detector_level.pdf"
    )
    assert drawn["losses"] == [run_dir / "artifacts" / "losses.pdf"]


@pytest.mark.writes_default_cache
def test_both_baselines_reach_the_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drawn: dict[str, list[Any]]
) -> None:
    """The whole point: weights written after training land on the plots."""
    monkeypatch.chdir(tmp_path)
    run_dir: Path = _write_run(tmp_path)
    _write_baselines(run_dir)

    plot_workflow.plot_run(run_dir)

    overlays: list[BaselineOverlay] = drawn["levels"][0]["baselines"]
    assert len(overlays) == 2


@pytest.mark.writes_default_cache
def test_a_directory_of_runs_redraws_each(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drawn: dict[str, list[Any]]
) -> None:
    monkeypatch.chdir(tmp_path)
    first: Path = _write_run(tmp_path, "2026-10-04T000000Z")
    second: Path = _write_run(tmp_path, "2026-10-04T000001Z")

    plot_workflow.plot_runs(tmp_path / "runs")

    assert drawn["loaded"] == [first, second]
