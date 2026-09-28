"""The OmniFold baseline's seams, none of which need TensorFlow to test.

What is worth testing here is not OmniFold --- that is a third-party package
under comparison, and running it would pull ~3.5GB of CUDA wheels into CI. What
is worth testing is the quarantine: that the worker stays unimportable, that the
host half talks to it over the contract it claims, and that the one silent
failure mode this baseline has is reported rather than swallowed.

Every test here substitutes a stub worker for the real one, which is why the
`unfold` seam takes a `worker` override at all.
"""

from __future__ import annotations

import json
import logging
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import sys
from typing import TYPE_CHECKING

import numpy as np
import pytest
from deconvolve.baselines import _shared as shared
from deconvolve.baselines import omnifold
from deconvolve.coretypes import ZXY, DatasetSplits, Events
from deconvolve.data import ArrayDataset
from deconvolve.instrumentation import timing

if TYPE_CHECKING:
    from pathlib import Path

    from deconvolve.coretypes import EventArray


def _split(z: list[list[float]], x: list[list[float]], y: list[int]) -> ArrayDataset:
    return ArrayDataset(
        ZXY(
            Events(np.asarray(z, dtype=np.single), np.asarray(x, dtype=np.single)),
            np.asarray(y, dtype=np.ubyte),
        ),
        batch_size=8,
    )


def _splits() -> DatasetSplits:
    """Two-dimensional synthetic splits, balanced so `partition` succeeds.

    Built here rather than generated, for the reason `test_ibu.py` does the
    same: `_load_splits` would reach the dataset cache, and what these tests are
    about is the subprocess seam, not data loading.
    """
    return DatasetSplits(
        train=_split(
            [[0.2, 0.3], [0.4, 0.5], [1.2, 1.3], [1.4, 1.5]],
            [[-1.0, 0.1], [0.5, 0.6], [1.1, 1.2], [3.0, 3.1]],
            [0, 1, 0, 1],
        ),
        val=_split([[0.6, 0.7], [1.6, 1.7]], [[0.7, 0.8], [1.7, 1.8]], [0, 1]),
        test=_split(
            [[0.8, 0.9], [1.8, 1.9], [0.9, 1.0], [1.9, 2.0]],
            [[0.9, 1.0], [2.5, 2.6], [0.8, 0.9], [2.2, 2.3]],
            [0, 0, 1, 1],
        ),
    )


# A stand-in for `_omnifold_worker.py`: same argv contract, same output keys, no
# TensorFlow. It returns a weight vector keyed to the input length so a test can
# tell a real round trip from a coincidence.
STUB_WORKER: str = """
import sys
import numpy as np

with np.load(sys.argv[1], allow_pickle=False) as payload:
    z_target = payload["z_target"]
    weights = np.linspace(0.5, 1.5, len(z_target)).astype(np.single)
    weights = weights / weights.mean()
    np.savez(
        sys.argv[2],
        weights=weights,
        device=np.array("{device}"),
        init_seconds=np.array(1.5),
        unfold_seconds=np.array(30.25),
        reweight_seconds=np.array(0.5),
    )
"""


@pytest.fixture
def stub_worker(tmp_path: Path) -> Path:
    path: Path = tmp_path / "stub_worker.py"
    _ = path.write_text(STUB_WORKER.format(device="/physical_device:GPU:0"))
    return path


class TestQuarantine:
    """The properties that keep TensorFlow out of this interpreter."""

    def test_the_worker_is_not_imported_by_the_package(self) -> None:
        """Importing `deconvolve` must not pull the worker in.

        If it ever is imported, its module-level `KERAS_BACKEND=tensorflow` would
        race the package's `jax` pin and the failure would surface somewhere
        unrelated.
        """
        assert "deconvolve.baselines._omnifold_worker" not in sys.modules

    def test_tensorflow_is_not_importable(self) -> None:
        """The project environment must not contain TensorFlow at all.

        This is the invariant the subprocess exists to preserve, and it is
        cheap enough to assert directly.
        """
        with pytest.raises(ImportError):
            _ = __import__("tensorflow")

    def test_the_worker_declares_a_python_313_environment(self) -> None:
        """The PEP 723 header is the quarantine; assert it is intact.

        A header edited to `>=3.13` would let uv resolve the worker onto 3.14,
        where TensorFlow has no wheels, and the failure would be a resolution
        error far from this file.
        """
        with omnifold.worker_script() as script:
            header: str = script.read_text()
        assert "# /// script" in header
        assert 'requires-python = "==3.13.*"' in header
        assert "omnifold" in header
        assert "tensorflow" in header

    def test_the_packaged_worker_is_reachable(self) -> None:
        """`importlib.resources` must find the worker in an installed package."""
        with omnifold.worker_script() as script:
            assert script.is_file()
            assert script.name == "_omnifold_worker.py"

    def test_the_worker_compiles_under_this_interpreter(self) -> None:
        """Syntax-only check, without importing it.

        It guards the PEP 758 trap: at a `>=3.14` project floor ruff inferred a
        py314 target and formatted `except (A, B)` into the unparenthesized
        form, a SyntaxError on the 3.13 the worker actually runs under. The
        floor is `>=3.12` now so ruff infers py312 and the trap is not armed,
        but `pyproject.toml` still pins `per-file-target-version` for
        `*_worker.py`; this is the assertion that the worker stays compilable
        whichever way the floor moves.
        """
        with omnifold.worker_script() as script:
            _ = compile(script.read_text(), str(script), "exec")


class TestUnfold:
    """The host half's side of the `.npz` contract."""

    def test_it_returns_the_workers_weights(
        self, stub_worker: Path, tmp_path: Path
    ) -> None:
        z_target: EventArray = np.linspace(0.0, 1.0, 32, dtype=np.single).reshape(16, 2)
        weights: EventArray = omnifold.unfold(
            x_data=np.zeros((8, 2), dtype=np.single),
            x_sim=np.zeros((8, 2), dtype=np.single),
            z_gen=np.zeros((8, 2), dtype=np.single),
            z_target=z_target,
            out_dir=tmp_path / "artifacts",
            worker=stub_worker,
        )
        assert weights.shape == (16,)
        # The stub normalizes to mean one, as the real worker does.
        assert float(np.mean(weights)) == pytest.approx(1.0, abs=1e-6)

    def test_it_passes_the_knobs_through(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`niter`, `epochs` and `batch_size` must survive the npz round trip."""
        echo: Path = tmp_path / "echo_worker.py"
        _ = echo.write_text(
            "import sys\n"
            "import numpy as np\n"
            "with np.load(sys.argv[1], allow_pickle=False) as p:\n"
            "    np.savez(\n"
            "        sys.argv[2],\n"
            "        weights=np.ones(len(p['z_target']), dtype=np.single),\n"
            "        device=np.array(str(int(p['niter'])) + '/' +\n"
            "                        str(int(p['epochs'])) + '/' +\n"
            "                        str(int(p['batch_size'])) + ' GPU'),\n"
            "    )\n"
        )
        # The device is reported at INFO, which the logger's effective level
        # discards before any handler runs -- hence `at_level`, not a bare handler.
        with caplog.at_level(logging.INFO, logger=omnifold.logger.name):
            _ = omnifold.unfold(
                x_data=np.zeros((4, 1), dtype=np.single),
                x_sim=np.zeros((4, 1), dtype=np.single),
                z_gen=np.zeros((4, 1), dtype=np.single),
                z_target=np.zeros((4, 1), dtype=np.single),
                out_dir=tmp_path / "artifacts",
                n_iterations=7,
                n_epochs=11,
                batch_size=64,
                worker=echo,
            )

        assert "7/11/64" in caplog.text

    def test_a_failing_worker_raises_with_its_stderr(self, tmp_path: Path) -> None:
        """A worker crash must surface the reason, not just an exit code."""
        broken: Path = tmp_path / "broken_worker.py"
        _ = broken.write_text(
            "import sys\nprint('the CUDA thing went wrong', file=sys.stderr)\n"
            "raise SystemExit(3)\n"
        )
        with pytest.raises(RuntimeError, match="the CUDA thing went wrong"):
            _ = omnifold.unfold(
                x_data=np.zeros((4, 1), dtype=np.single),
                x_sim=np.zeros((4, 1), dtype=np.single),
                z_gen=np.zeros((4, 1), dtype=np.single),
                z_target=np.zeros((4, 1), dtype=np.single),
                out_dir=tmp_path / "artifacts",
                worker=broken,
            )

    def test_a_missing_uv_says_so(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`uv` absent is an install problem and must read as one."""

        def _no_uv(*_args: object, **_kwargs: object) -> object:
            raise FileNotFoundError(2, "No such file or directory: 'uv'")

        monkeypatch.setattr(subprocess, "run", _no_uv)
        with pytest.raises(RuntimeError, match="uv` was not found on PATH"):
            _ = omnifold.unfold(
                x_data=np.zeros((4, 1), dtype=np.single),
                x_sim=np.zeros((4, 1), dtype=np.single),
                z_gen=np.zeros((4, 1), dtype=np.single),
                z_target=np.zeros((4, 1), dtype=np.single),
                out_dir=tmp_path / "artifacts",
            )


class TestCpuFallbackIsReported:
    """The one silent failure this baseline has.

    TensorFlow that cannot load its CUDA libraries runs on the CPU and raises
    nothing, so without this warning a Perlmutter run missing
    `module load cudatoolkit/12.9` would look like a successful, very slow
    baseline. Asserted at the `_warn_if_on_cpu` seam rather than end to end
    because the condition is a string from the worker either way.
    """

    def test_a_cpu_device_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=omnifold.logger.name):
            omnifold._warn_if_on_cpu("/physical_device:CPU:0")
        assert any(r.levelno == logging.WARNING for r in caplog.records)
        assert "cudatoolkit/12.9" in caplog.text

    def test_a_gpu_device_does_not_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO, logger=omnifold.logger.name):
            omnifold._warn_if_on_cpu("/physical_device:GPU:0")
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    def test_an_unknown_device_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        """Absent information is treated as the bad case, not the good one."""
        with caplog.at_level(logging.WARNING, logger=omnifold.logger.name):
            omnifold._warn_if_on_cpu("unknown")
        assert any(r.levelno == logging.WARNING for r in caplog.records)


class TestEvaluateSingle:
    """The artifact layout, which `deconvolve report` reads positionally."""

    def test_it_writes_into_artifacts_and_caches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_dir: Path = tmp_path / "run"
        run_dir.mkdir()
        dim: int = 2
        n: int = 8
        _ = (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "dataset": "gaussian",
                    "dim": dim,
                    "n_samples": n,
                    "batch_size": 8,
                    "data_seed": 42,
                }
            )
        )

        monkeypatch.setattr(shared, "_load_splits", lambda **_kwargs: _splits())

        def _stub_unfold(**kwargs: object) -> EventArray:
            target = kwargs["z_target"]
            assert isinstance(target, np.ndarray)
            return np.ones(target.shape[0], dtype=np.single)

        monkeypatch.setattr(omnifold, "unfold", _stub_unfold)
        metrics = omnifold.evaluate_single(run_dir)

        artifacts: Path = run_dir / "artifacts"
        assert (artifacts / "metrics_omnifold.json").is_file()
        assert (artifacts / "omnifold_weights.npz").is_file()
        # The reload pass in `workflows.train` reads the array back by this name;
        # a positional `np.savez` would store it as `arr_0` instead.
        with np.load(artifacts / "omnifold_weights.npz", allow_pickle=False) as saved:
            assert saved.files == ["weights"]
        assert set(metrics) == {
            f"{level}_dim_{i}" for level in ("detector", "particle") for i in range(dim)
        }

        # A second call must not re-run the unfolding.
        def _explode(**_kwargs: object) -> EventArray:
            raise AssertionError("evaluate_single did not use the cached metrics")

        monkeypatch.setattr(omnifold, "unfold", _explode)
        again = omnifold.evaluate_single(run_dir)
        assert again == metrics


class TestTheWorkerDoesNotImportThisPackage:
    """The worker's `import omnifold` must find the package, not this module.

    `_omnifold_worker.py` sits in the same directory as `omnifold.py`, and a
    script's own directory goes on `sys.path[0]`. So the worker's
    `from omnifold import MLP, DataLoader, MultiFold` resolved to the host half
    instead of the installed package, and died on its `from .. import timing`
    with "attempted relative import with no known parent package" -- an error
    naming neither the collision nor the file that caused it.

    Reproduced here with the same *shape* rather than the same names: a worker
    whose directory contains a module it is about to import.
    """

    @staticmethod
    def _shadowed(tmp_path: Path) -> Path:
        """A worker with a poisoned sibling it must not be able to import."""
        _ = (tmp_path / "omnifold.py").write_text(
            "raise RuntimeError('the sibling was imported')\n"
        )
        worker = tmp_path / "shadow_worker.py"
        _ = worker.write_text(
            "import sys\n"
            "import numpy as np\n"
            "try:\n"
            "    import omnifold\n"
            "except ImportError:\n"
            "    shadowed = 'no'\n"
            "else:\n"
            "    shadowed = 'yes'\n"
            "with np.load(sys.argv[1], allow_pickle=False) as p:\n"
            "    np.savez(\n"
            "        sys.argv[2],\n"
            "        weights=np.ones(len(p['z_target']), dtype=np.single),\n"
            "        device=np.array('GPU shadowed=' + shadowed),\n"
            "    )\n"
        )
        return worker

    def test_the_workers_directory_is_not_on_sys_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger=omnifold.logger.name):
            _ = omnifold.unfold(
                x_data=np.zeros((4, 1), dtype=np.single),
                x_sim=np.zeros((4, 1), dtype=np.single),
                z_gen=np.zeros((4, 1), dtype=np.single),
                z_target=np.zeros((4, 1), dtype=np.single),
                out_dir=tmp_path / "artifacts",
                worker=self._shadowed(tmp_path),
            )

        assert "shadowed=no" in caplog.text, (
            "the worker imported a module from its own directory; "
            "PYTHONSAFEPATH is not reaching it"
        )

    def test_the_env_sets_safepath(self) -> None:
        """The mechanism itself, so a removal is not silent."""
        assert omnifold._worker_env()["PYTHONSAFEPATH"] == "1"


class TestWorkerTimings:
    """The worker's breakdown, folded into this run's timing tree.

    OmniFold has no timing of its own, so the worker measures its three stages
    and wraps `RunStep1`/`RunStep2` for a per-iteration split. Those numbers
    arrive as `.npz` entries with no block here to wrap, which is what
    `timing.record` is for.
    """

    TIMED_WORKER: str = (
        "import sys\n"
        "import numpy as np\n"
        "with np.load(sys.argv[1], allow_pickle=False) as p:\n"
        "    np.savez(\n"
        "        sys.argv[2],\n"
        "        weights=np.ones(len(p['z_target']), dtype=np.single),\n"
        "        device=np.array('/physical_device:GPU:0'),\n"
        "        init_seconds=np.array(12.5),\n"
        "        unfold_seconds=np.array(2400.0),\n"
        "        reweight_seconds=np.array(3.25),\n"
        "        step1_seconds=np.array([400.0, 402.0]),\n"
        "        step2_seconds=np.array([395.0, 401.0]),\n"
        "    )\n"
    )

    def _run(self, tmp_path: Path) -> list[timing.Phase]:
        worker = tmp_path / "timed_worker.py"
        _ = worker.write_text(self.TIMED_WORKER)
        timing.enable(True)
        try:
            with timing.phase("omnifold"):
                _ = omnifold.unfold(
                    x_data=np.zeros((4, 1), dtype=np.single),
                    x_sim=np.zeros((4, 1), dtype=np.single),
                    z_gen=np.zeros((4, 1), dtype=np.single),
                    z_target=np.zeros((4, 1), dtype=np.single),
                    out_dir=tmp_path / "artifacts",
                    worker=worker,
                )
            return list(timing.phases())
        finally:
            timing.enable(False)

    def test_the_three_stages_are_recorded(self, tmp_path: Path) -> None:
        names = [p.name for p in self._run(tmp_path)]

        assert "init" in names
        assert "unfold" in names
        assert "reweight" in names

    def test_the_iteration_split_is_recorded(self, tmp_path: Path) -> None:
        """Step 1 and step 2 are not symmetric, so a single total hides which
        half a long run spent its time in."""
        names = [p.name for p in self._run(tmp_path)]

        assert "iter1_step1" in names
        assert "iter2_step2" in names

    def test_the_iteration_rows_follow_unfold(self, tmp_path: Path) -> None:
        """They are `unfold`'s breakdown and the table is read in order."""
        names = [p.name for p in self._run(tmp_path)]

        assert names.index("unfold") < names.index("iter1_step1")
        assert names.index("iter2_step2") < names.index("reweight")

    def test_the_worker_phases_nest_under_the_open_phase(self, tmp_path: Path) -> None:
        """Depth 1, so `total_seconds` -- which sums depth 0 -- is unaffected."""
        phases = {p.name: p for p in self._run(tmp_path)}

        assert phases["unfold"].depth == 1
        assert phases["iter1_step1"].depth == 1

    def test_a_worker_reporting_no_timings_is_fine(
        self, tmp_path: Path, stub_worker: Path
    ) -> None:
        """A worker whose OmniFold no longer exposes the steps still works.

        The wrapping is guarded, so a rename inside OmniFold costs the
        breakdown and not the baseline.
        """
        timing.enable(True)
        try:
            with timing.phase("omnifold"):
                _ = omnifold.unfold(
                    x_data=np.zeros((4, 1), dtype=np.single),
                    x_sim=np.zeros((4, 1), dtype=np.single),
                    z_gen=np.zeros((4, 1), dtype=np.single),
                    z_target=np.zeros((4, 1), dtype=np.single),
                    out_dir=tmp_path / "artifacts",
                    worker=stub_worker,
                )
            names = [p.name for p in timing.phases()]
        finally:
            timing.enable(False)

        assert not [n for n in names if n.startswith("iter")]
        assert "init" in names  # the stub does report the three totals
