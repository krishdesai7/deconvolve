"""Tests for the --load-run reload path.

This path had two independent breaks that the suite never saw, both found by a
cluster smoke run rather than by CI:

  - `run()` handed `parse_run_config` the raw text of config.json, but that
    function validates an already-decoded object, so every reload raised
    "run config must be a JSON object";
  - the Gaussian params were then read with the wrong key spelling.

Nothing here trains. The reload path does not call `train` at all, and the
model load, the plots and the metrics are stubbed -- what is exercised is the
config parsing and the dataset rebuild, which is exactly where the breaks were.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, override

import numpy as np
import pytest
from deconvolve.coretypes import ZXY, DatasetName, Events, Populations
from deconvolve.coretypes.events import DatasetSplits
from deconvolve.data import DeconvolveDataset, parse_gaussian_config
from deconvolve.training.engine import TrainResult, train
from deconvolve.workflows import train as workflow
from deconvolve.workflows.train import _compact_variables

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any, Final

    from deconvolve.coretypes import DatasetSplits, GaussianConfig
    from deconvolve.evaluation.plotting import BaselineOverlay
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


@dataclass
class Recorded:
    """What the stubbed-out tail of `run()` was asked to do."""

    plots: list[Path | None] = field(default_factory=list)
    evaluated: tuple[Path, bool] | None = None


@pytest.fixture
def stub_heavy(monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """Stub the parts a reload does not need to prove config handling works."""
    recorded = Recorded()

    def fake_load_artifacts(
        run_dir: Path,
    ) -> Callable[
        [NDArray[np.double]], tuple[NDArray[np.double], dict[str, list[float]]]
    ]:
        del run_dir
        return lambda z: np.ones(shape=(len(z), 1)), {"train_d": [0.7]}  # ty: ignore[invalid-return-type]

    def fake_plot(
        *args: tuple[Any, ...], save_path: Path | None = None, **kwargs: dict[str, Any]
    ) -> None:
        del args, kwargs
        recorded.plots.append(save_path)

    def fake_plot_levels(
        *args: tuple[Any, ...],
        detector_path: Path,
        particle_path: Path,
        **kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        recorded.plots.extend((detector_path, particle_path))

    def fake_evaluate(run_dir: Path, force: bool) -> None:
        recorded.evaluated = (run_dir, force)

    monkeypatch.setattr(
        target=workflow, name="_load_artifacts", value=fake_load_artifacts
    )
    monkeypatch.setattr(workflow, "plot_levels", value=fake_plot_levels)
    monkeypatch.setattr(workflow, "plot_losses", value=fake_plot)
    monkeypatch.setattr(target=workflow, name="evaluate_run", value=fake_evaluate)
    return recorded


def _reload(run_dir: Path) -> None:
    """Drive `run()` down the --load-run path.

    `run()` keeps no defaults of its own -- the CLI owns them -- so every
    argument has to be named. The values below are placeholders a reload is
    meant to replace with what config.json recorded, deliberately unlike the
    ones `_write_run` writes so that
    `test_load_run_forwards_recorded_seed_and_size` fails if it stops doing so.
    """
    workflow.run(
        batch_size=8,
        n_samples=32,
        config=None,
        dataset=DatasetName.gaussian,
        variables=(),
        load_run=run_dir,
        hidden_units=4,
        n_layers=1,
        seed=None,
        data_seed=0,
        n_epochs=100,
        n_disc_steps=5,
        lr_g=3e-5,
        lr_d=1e-4,
        lambda_dispersion=0.015,
        log_every=1,
        plots=True,
        run_dir=None,
    )


def _write_run(
    tmp_path: Path, gaussian_params: dict[str, Any], **overrides: dict[str, Any]
) -> Path:
    run_dir = tmp_path / "runs" / "2026-08-06T000000Z"
    run_dir.mkdir(parents=True)
    config = {
        "batch_size": 64,
        "n_samples": 400,
        "dim": 2,
        "dataset": "gaussian",
        "seed": 0,
        "data_seed": 42,
        "gaussian_params": gaussian_params,
        **overrides,
    }
    _ = (run_dir / "config.json").write_text(data=json.dumps(obj=config))
    return run_dir


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
    dumped: str = json.dumps(
        {"dim": 2, "gaussian_params": {"mu_gen": [0, 0]}}, indent=2
    )

    assert _compact_variables(dumped) == dumped


@pytest.mark.writes_default_cache
def test_load_run_reads_a_config_written_by_save_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_heavy: Recorded
) -> None:
    """The round trip that matters: what _save_run writes, run() must read."""
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "cfg.yaml").write_text(data=CONFIG_2D)
    params: GaussianConfig = parse_gaussian_config(config_path=tmp_path / "cfg.yaml")
    run_dir: Path = _write_run(tmp_path, gaussian_params=params.model_dump())

    _reload(run_dir)

    assert len(stub_heavy.plots) == 3
    assert stub_heavy.evaluated == (run_dir, False)


@pytest.mark.writes_default_cache
def test_load_run_reads_master_era_sigma_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stub_heavy: Recorded
) -> None:
    """Runs written before the type refactor stored covariances as sigma_*."""
    monkeypatch.chdir(tmp_path)
    run_dir: Path = _write_run(
        tmp_path,
        gaussian_params={
            "dim": 2,
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [[1.0, -0.9], [-0.9, 2.25]],
            "sigma_true": [[0.81, -0.702], [-0.702, 1.69]],
            "sigma_detector": [[0.25, 0.0], [0.0, 0.64]],
        },
    )

    _reload(run_dir)

    assert len(stub_heavy.plots) == 3


@pytest.mark.writes_default_cache
@pytest.mark.usefixtures("stub_heavy")
def test_load_run_forwards_recorded_seed_and_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A reload must rebuild the split the run trained on, not a fresh one.

    `data_seed` and `n_samples` come from config.json, so the events the plots
    and metrics see are the ones the generator was fitted against.
    """
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "cfg.yaml").write_text(data=CONFIG_2D)
    params = parse_gaussian_config(tmp_path / "cfg.yaml")
    run_dir: Path = _write_run(
        tmp_path,
        gaussian_params=params.model_dump(),
        data_seed=7,  # ty: ignore[invalid-argument-type] -- kwargs
        n_samples=600,  # ty: ignore[invalid-argument-type]
    )

    seen: dict[str, object] = {}
    real: type[DeconvolveDataset] = workflow.DeconvolveDataset

    class Recording(real):
        def __init__(self, *args: tuple[Any, ...], **kwargs: dict[str, Any]) -> None:
            seen["seed"] = kwargs.get("seed")
            super().__init__(*args, **kwargs)  # ty: ignore[invalid-argument-type] -- *args

        @override
        def generate_gaussian_dataset(self, *args, **kwargs) -> DatasetSplits:  # pyrefly: ignore[implicit-any-parameter] -- args, kwargs
            seen["n_samples"] = kwargs.get("n_samples")
            return super().generate_gaussian_dataset(*args, **kwargs)

    monkeypatch.setattr(target=workflow, name="DeconvolveDataset", value=Recording)
    _reload(run_dir)

    assert seen == {"seed": 7, "n_samples": 600}


_GAUSSIAN_CONFIG = """\
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9
sigma_true: 1.0
sigma_detector: 0.5
"""


@pytest.mark.writes_default_cache
def test_particle_curve_is_recorded_when_truth_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The diagnostic curve lives here, not in `train`, because it needs
    `z_true` -- and putting truth-derived Grams in the trace is exactly what
    `Populations` keeping `truth` outside `Events` exists to prevent.
    """
    monkeypatch.chdir(tmp_path)
    config_path: Path = tmp_path / "gaussian.yaml"
    _ = config_path.write_text(data=_GAUSSIAN_CONFIG)

    workflow.run(
        128,
        2000,
        config_path,
        dataset=DatasetName.gaussian,
        variables=(),
        load_run=None,
        hidden_units=8,
        n_layers=1,
        seed=3,
        data_seed=42,
        n_epochs=4,
        n_disc_steps=2,
        lr_g=1e-4,
        lr_d=1e-3,
        lambda_dispersion=0.01,
        log_every=1,
        plots=False,
        run_dir=None,
    )

    run_dir: Path = next((tmp_path / "runs").iterdir())
    history: dict[str, list[float]] = dict(
        np.load(file=run_dir / "artifacts" / "history.npz")
    )
    assert "val_mmd_particle" in history
    assert len(history["val_mmd_particle"]) == 4
    assert np.all(a=np.isfinite(history["val_mmd_particle"]))

    config: Any = json.loads(s=(run_dir / "config.json").read_text())
    assert config["best_epoch"] >= 0
    assert config["mmd_subsample"] == 16384
    assert len(config["mmd_sigmas_detector"]) == 5  # pyrefly: ignore[unknown-argument-type] -- config["mmd_sigmas_*"] is a list
    assert len(config["mmd_sigmas_particle"]) == 5  # pyrefly: ignore[unknown-argument-type]
    # The unsound knobs are gone from the record, not merely unused.
    assert "criterion" not in config
    assert "patience" not in config
    assert "min_delta" not in config


@pytest.mark.writes_default_cache
def test_a_completed_run_keeps_only_two_files_at_its_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run root is for humans; everything else lives in `artifacts/`."""
    monkeypatch.chdir(tmp_path)
    config_path: Path = tmp_path / "gaussian.yaml"
    _ = config_path.write_text(data=_GAUSSIAN_CONFIG)

    workflow.run(
        32,
        200,
        config_path,
        dataset=DatasetName.gaussian,
        variables=(),
        load_run=None,
        hidden_units=4,
        n_layers=1,
        seed=1,
        data_seed=0,
        n_epochs=1,
        n_disc_steps=2,
        lr_g=1e-4,
        lr_d=1e-3,
        lambda_dispersion=0.01,
        log_every=1,
        plots=False,
        run_dir=None,
    )

    run_dir: Path = next((tmp_path / "runs").iterdir())

    at_root: set[str] = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert at_root == {"config.json"}

    produced: set[str] = {p.name for p in (run_dir / "artifacts").iterdir()}
    assert {
        "generator.keras",
        "discriminator.keras",
        "history.npz",
        "params.npz",
    } <= produced


class TestParticleCurve:
    """Direct unit tests of `_particle_curve`, isolated from `train`/`run`.

    A black-box test through `run()` can only exercise the has-truth branch
    -- both real dataset sources (Gaussian, jets) always carry truth. But
    `_particle_curve` itself takes just a `DatasetSplits` and a `TrainResult`,
    so the no-truth branch is directly reachable without either.
    """

    @staticmethod
    def _truthless_splits(n: int = 64) -> DatasetSplits:
        """A `Populations` built with no `truth` argument, the way a real
        measurement's would be -- see `tests/test_datasets.py`.
        """
        rng: np.random.Generator = np.random.default_rng(seed=51)
        z_gen: NDArray[np.single] = rng.normal(size=(n, 1)).astype(dtype=np.single)
        x_sim: NDArray[np.single] = (z_gen + rng.normal(0, 0.4, size=(n, 1))).astype(
            dtype=np.single
        )
        x_data: NDArray[np.single] = rng.normal(size=(n, 1)).astype(dtype=np.single)
        pops: Populations = Populations.create(mc=Events(z_gen, x_sim), data=x_data)
        assert not pops.has_truth
        return DeconvolveDataset(batch_size=32, seed=9).splits_from_data(
            pops.interleave()
        )

    def test_returns_none_without_truth(self) -> None:
        """No truth means no diagnostic -- and no touching `result.g`/
        `result.params` to find that out: a stub with both `None` still
        works, which is only true if the `has_truth` guard runs first.
        """
        splits: DatasetSplits = self._truthless_splits()
        stub = TrainResult(
            g=None,  # ty: ignore[invalid-argument-type]
            d=None,  # ty: ignore[invalid-argument-type]
            history={},
            seed=0,
        )

        assert workflow._particle_curve(splits, result=stub) is None

    @pytest.mark.slow
    def test_returns_a_curve_with_truth(self) -> None:
        """The companion case: with truth present, a real curve comes back.

        Without this, a `_particle_curve` that always returned `None` would
        pass the no-truth test above for the wrong reason.
        """
        rng: np.random.Generator = np.random.default_rng(seed=52)
        n = 384
        z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
        x: NDArray[np.single] = (z + rng.normal(0, 0.4, size=(2 * n, 1))).astype(
            dtype=np.single
        )
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)]
        )
        splits: DatasetSplits = DeconvolveDataset(
            batch_size=64, seed=10
        ).splits_from_data(data=ZXY(Events(z, x), y))
        result: TrainResult = train(
            splits, dim=1, n_epochs=3, hidden_units=8, n_layers=1, seed=5
        )

        curve: tuple[list[float], tuple[float, ...]] | None = workflow._particle_curve(
            splits, result
        )

        assert curve is not None
        values, sigmas = curve
        assert len(values) == 3
        assert len(sigmas) == 5
        assert all(np.isfinite(v) for v in values)


@pytest.mark.slow
def test_run_omits_val_mmd_particle_without_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run()` drops the key rather than writing `None`/`NaN` into history.

    Neither real dataset source (`_prepare_gaussian`, `_prepare_jets`) ever
    produces a truthless split, so this stands `_prepare_gaussian` up on a
    truthless `DatasetSplits` -- the same one `TestParticleCurve` builds
    directly -- to prove `run()`'s own no-truth path end to end.
    """
    monkeypatch.chdir(tmp_path)
    splits: DatasetSplits = TestParticleCurve._truthless_splits(n=200)

    def fake_prepare_gaussian(
        config: GaussianConfig,
        saved_config: GaussianConfig,
        batch_size: int,
        n_samples: int,
        data_seed: int,
    ) -> tuple[DatasetSplits, int, None]:
        del config, saved_config, batch_size, n_samples, data_seed
        return splits, 1, None

    monkeypatch.setattr(
        target=workflow, name="_prepare_gaussian", value=fake_prepare_gaussian
    )

    workflow.run(
        batch_size=32,
        n_samples=200,
        config=None,
        dataset=DatasetName.gaussian,
        variables=(),
        load_run=None,
        hidden_units=8,
        n_layers=1,
        seed=6,
        data_seed=51,
        n_epochs=2,
        n_disc_steps=2,
        lr_g=1e-4,
        lr_d=1e-3,
        lambda_dispersion=0.01,
        log_every=1,
        plots=False,
        run_dir=None,
    )

    run_dir: Path = next((tmp_path / "runs").iterdir())
    history: dict[str, list[float]] = dict(
        np.load(file=run_dir / "artifacts" / "history.npz")
    )
    assert "val_mmd_particle" not in history


class _StubSplits:
    """Just enough of `DatasetSplits` for `_draw_figures` to read `.test`
    without touching it -- `plot_levels` is monkeypatched out in these tests,
    but `_draw_figures` still evaluates `splits.test` as the call argument.
    """

    test: None = None


class TestDrawFiguresSelection:
    """R15: a pre-MMD run's history has no `val_mmd`/`val_ess` at all, and
    `--load-run` must still be able to replot it. `plot_selection` would raise
    `KeyError` on that history and take the other two figures down with it, so
    `_draw_figures` skips it -- rather than erroring -- when `val_mmd` is
    absent.
    """

    @staticmethod
    def _patch_plot_fns(
        monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]]:
        calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = {
            "plot_levels": [],
            "plot_losses": [],
            "plot_selection": [],
        }

        def make(name: str) -> Callable[[tuple[Any, ...], dict[str, Any]], None]:
            return lambda *args, **kwargs: calls[name].append((args, kwargs))

        for name in calls:
            monkeypatch.setattr(workflow, name, value=make(name))
        return calls

    def test_missing_val_mmd_does_not_raise_and_skips_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = (
            self._patch_plot_fns(monkeypatch)
        )
        history: dict[str, list[float]] = {
            "train_d": [0.7],
            "train_g": [0.7],
            "val_d": [0.7],
        }

        workflow._draw_figures(
            tmp_path,
            _StubSplits(),  # ty: ignore[invalid-argument-type]
            None,  # ty: ignore[invalid-argument-type]
            history,
            1,
            None,
            -1,
            None,
            plots=True,
        )

        assert not calls["plot_selection"]
        assert len(calls["plot_levels"]) == 1
        _, level_kwargs = calls["plot_levels"][0]
        assert level_kwargs == {
            "detector_path": tmp_path / "artifacts" / "detector_level.pdf",
            "particle_path": tmp_path / "artifacts" / "particle_level.pdf",
            "var_info": None,
            "baselines": [],
            "variables": None,
        }
        assert calls["plot_losses"]

    def test_val_mmd_present_draws_selection_with_best_epoch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, list[tuple[tuple[Any, ...], dict[str, Any]]]] = (
            self._patch_plot_fns(monkeypatch)
        )
        history: dict[str, list[float]] = {
            "train_d": [0.7],
            "train_g": [0.7],
            "val_d": [0.7],
            "val_mmd": [0.1],
            "val_ess": [10.0],
        }

        workflow._draw_figures(
            tmp_path,
            _StubSplits(),  # ty: ignore[invalid-argument-type]
            None,  # ty: ignore[invalid-argument-type]
            history,
            1,
            None,
            5,
            None,
            plots=True,
        )

        assert len(calls["plot_selection"]) == 1
        args, kwargs = calls["plot_selection"][0]
        assert args[0] is history
        assert args[1] == 5
        assert kwargs["save_path"] == tmp_path / "artifacts" / "selection.pdf"


def _fake_load_artifacts(
    run_dir: Path,
) -> tuple[Callable[[NDArray[np.double]], NDArray[np.double]], dict[str, list[float]]]:
    del run_dir
    return lambda z: np.ones(shape=(len(z), 1)), {"train_d": [0.7]}


def _fake_evaluate_run(run_dir: Path, force: bool) -> None:
    del run_dir, force


def _stub_best_epoch_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_dir: Path
) -> dict[str, int]:
    """Stand up a reload that stubs everything but `_draw_figures`'s
    `best_epoch` argument, and return the dict it lands in.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        target=workflow, name="_load_artifacts", value=_fake_load_artifacts
    )
    monkeypatch.setattr(target=workflow, name="evaluate_run", value=_fake_evaluate_run)
    seen: dict[str, int] = {}

    def fake_draw_figures(
        run_dir: Path,
        splits: DatasetSplits,
        g: None,
        history: dict[str, list[float]],
        dim: int,
        var_info: None,
        best_epoch: int,
        variables: tuple[str, ...] | None,
        /,
        *,
        plots: bool,
    ) -> None:
        del run_dir, splits, g, history, dim, var_info, variables, plots
        seen["best_epoch"] = best_epoch

    monkeypatch.setattr(target=workflow, name="_draw_figures", value=fake_draw_figures)
    _reload(run_dir)
    return seen


@pytest.mark.writes_default_cache
def test_reload_sources_best_epoch_from_recorded_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R15: on `--load-run` there is no `TrainResult`, so `best_epoch` comes
    from `config.json` (`RunConfig.source`), not from a training result.
    """
    _ = (tmp_path / "cfg.yaml").write_text(data=CONFIG_2D)
    params: GaussianConfig = parse_gaussian_config(config_path=tmp_path / "cfg.yaml")
    run_dir: Path = _write_run(
        tmp_path,
        gaussian_params=params.model_dump(),
        best_epoch=7,  # ty: ignore[invalid-argument-type] -- kwargs
    )

    seen: dict[str, int] = _stub_best_epoch_reload(tmp_path, monkeypatch, run_dir)

    assert seen["best_epoch"] == 7


@pytest.mark.writes_default_cache
def test_reload_defaults_best_epoch_for_a_legacy_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A config.json written before this branch has no `best_epoch` key."""
    _ = (tmp_path / "cfg.yaml").write_text(data=CONFIG_2D)
    params: GaussianConfig = parse_gaussian_config(config_path=tmp_path / "cfg.yaml")
    run_dir: Path = _write_run(tmp_path, gaussian_params=params.model_dump())

    seen: dict[str, int] = _stub_best_epoch_reload(tmp_path, monkeypatch, run_dir)

    assert seen["best_epoch"] == -1


class TestNewRunDir:
    """Where a run lands, and what happens when two of them want one place.

    `_save_run` used to build the path inline as a second-resolution UTC
    timestamp and `mkdir(exist_ok=True)`. A packed sweep launches runs of
    identical shape at once, so several finish inside the same second: the
    losers were overwritten with no error and no way to tell afterwards which
    arm had gone missing. Both halves of the fix are here -- an explicit
    directory that refuses to land on an existing run, and a default that stops
    colliding with itself.
    """

    def test_explicit_directory_is_created_and_returned(self, tmp_path: Path) -> None:
        wanted: Path = tmp_path / "hp_lrg" / "lrg1e-4_seed00"

        assert workflow._new_run_dir(explicit=wanted) == wanted
        assert wanted.is_dir()

    def test_explicit_directory_refuses_to_overwrite_a_finished_run(
        self, tmp_path: Path
    ) -> None:
        occupied: Path = tmp_path / "arm" / "seed00"
        occupied.mkdir(parents=True)
        _ = (occupied / "config.json").write_text(data="{}")

        with pytest.raises(expected_exception=FileExistsError, match="seed00"):
            _ = workflow._new_run_dir(explicit=occupied)

    def test_explicit_directory_accepts_a_path_the_launcher_pre_made(
        self, tmp_path: Path
    ) -> None:
        """srun redirects its log into the arm directory, so it may exist."""
        empty: Path = tmp_path / "arm" / "seed00"
        empty.mkdir(parents=True)

        assert workflow._new_run_dir(explicit=empty) == empty

    def test_default_directories_do_not_collide_within_one_second(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        first: Path = workflow._new_run_dir(explicit=None)
        second: Path = workflow._new_run_dir(explicit=None)

        assert first != second
        assert first.is_dir()
        assert second.is_dir()
        assert first.parent.name == second.parent.name == "runs"


def test_run_rejects_an_output_directory_on_the_reload_path(tmp_path: Path) -> None:
    """`--load-run` already names a directory; `--run-dir` would be ignored.

    A flag that silently does nothing in a sweep script is the same class of
    bug as the clobber it was added to prevent.
    """
    with pytest.raises(expected_exception=ValueError, match="--run-dir"):
        workflow.run(
            batch_size=8,
            n_samples=32,
            config=None,
            dataset=DatasetName.gaussian,
            variables=(),
            load_run=tmp_path / "saved",
            hidden_units=4,
            n_layers=1,
            seed=None,
            data_seed=0,
            n_epochs=100,
            n_disc_steps=5,
            lr_g=3e-5,
            lr_d=1e-4,
            lambda_dispersion=0.015,
            log_every=1,
            plots=True,
            run_dir=tmp_path / "elsewhere",
        )


@pytest.mark.writes_default_cache
@pytest.mark.usefixtures("stub_heavy")
def test_timing_writes_a_phase_breakdown_into_the_run_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reload path, end to end, with `DECONVOLVE_TIMING` on.

    `data`/`load`/`plots`/`evaluate` are opened in `run()` itself, so this is
    what says the phases survive a real call rather than only the unit tests in
    `tests/test_timing.py`.
    """
    from deconvolve.instrumentation import timing

    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "cfg.yaml").write_text(data=CONFIG_2D)
    params: GaussianConfig = parse_gaussian_config(config_path=tmp_path / "cfg.yaml")
    run_dir: Path = _write_run(tmp_path, gaussian_params=params.model_dump())

    timing.enable(True)
    try:
        _reload(run_dir)
    finally:
        timing.enable(False)

    payload: dict[str, Any] = json.loads(
        s=(run_dir / "artifacts" / "timings.json").read_text()
    )
    assert {p["name"] for p in payload["phases"]} == {
        "data",
        "load",
        "plots",
        "evaluate",
    }
    assert payload["total_seconds"] > 0.0
    # The reload path never trains, so nothing nested under `train` appears.
    assert all(p["depth"] == 0 for p in payload["phases"])


@pytest.mark.writes_default_cache
@pytest.mark.usefixtures("stub_heavy")
def test_no_timings_file_when_timing_is_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _ = (tmp_path / "cfg.yaml").write_text(data=CONFIG_2D)
    params: GaussianConfig = parse_gaussian_config(config_path=tmp_path / "cfg.yaml")
    run_dir: Path = _write_run(tmp_path, gaussian_params=params.model_dump())

    _reload(run_dir)

    assert not (run_dir / "artifacts" / "timings.json").exists()


class TestBaselineDiscovery:
    """Which baselines reach the figures, and on what evidence.

    Presence of `artifacts/*_weights.npz` is the entire mechanism: no baseline
    runs on the `deconvolve train` path, so a fresh run's figures carry no overlay and
    `deconvolve train --load-run` after a baseline has run is what puts one there.
    That makes "does the file exist" the thing worth testing.
    """

    @staticmethod
    def _artifacts(tmp_path: Path) -> Path:
        artifacts: Path = tmp_path / "artifacts"
        artifacts.mkdir(parents=True, exist_ok=True)
        return artifacts

    def test_no_weights_means_no_overlays(self, tmp_path: Path) -> None:
        assert workflow._load_baseline_weights(run_dir=tmp_path, dim=2) == []

    def test_omnifold_weights_alone_are_picked_up(self, tmp_path: Path) -> None:
        """OmniFold without IBU must work; the two are independent."""
        np.savez(
            file=self._artifacts(tmp_path) / "omnifold_weights.npz",
            weights=np.asarray([1.0, 2.0, 3.0], dtype=np.single),
        )

        overlays: list[BaselineOverlay] = workflow._load_baseline_weights(
            run_dir=tmp_path, dim=2
        )

        assert [o.label for o in overlays] == ["OmniFold"]
        # One vector, repeated: OmniFold reweights events, not observables.
        assert len(overlays[0].weights) == 2
        assert np.array_equal(a1=overlays[0].weights[0], a2=overlays[0].weights[1])

    def test_both_baselines_are_picked_up_in_draw_order(self, tmp_path: Path) -> None:
        artifacts: Path = self._artifacts(tmp_path)
        np.savez(
            file=artifacts / "ibu_weights.npz",
            weights_0=np.asarray(a=[1.0, 1.0], dtype=np.single),
            weights_1=np.asarray(a=[2.0, 2.0], dtype=np.single),
        )
        np.savez(
            file=artifacts / "omnifold_weights.npz",
            weights=np.asarray(a=[3.0, 3.0], dtype=np.single),
        )

        overlays: list[BaselineOverlay] = workflow._load_baseline_weights(
            run_dir=tmp_path, dim=2
        )

        assert [o.label for o in overlays] == ["IBU", "OmniFold"]
        # IBU's per-observable vectors must stay distinct, not be collapsed.
        assert not np.array_equal(a1=overlays[0].weights[0], a2=overlays[0].weights[1])
