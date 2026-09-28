"""The OmniFold baseline, run in a quarantined subprocess.

OmniFold needs TensorFlow, which this project cannot depend on directly.
`_omnifold_worker.py` carries a PEP 723 header;
`uv run --no-project` provisions Python 3.13 and TensorFlow for it in an
interpreter that cannot import `ran`; the two halves exchange one `.npz` file.

This half does what every other baseline does: read a run's `config.json`,
rebuild its populations, score a weight vector against the same metrics
`deconvolve evaluate` uses, so the comparison is scored by shared code rather than a
vendored copy that could drift.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # ruff: ignore[suspicious-subprocess-import]
import tempfile
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from ..coretypes import artifacts_dir
from ..evaluation import apply_to_runs, render_metrics
from ..instrumentation import timing
from ._shared import evaluate_dimension, load_populations, parse_run_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager
    from logging import Logger
    from typing import Any

    from numpy.typing import NDArray

    from ..coretypes import (
        EventArray,
        MetricRecord,
        RunConfig,
        UnfoldingPopulations,
    )

logger: Logger = logging.getLogger(name=__name__)

DEFAULT_N_ITERATIONS: int = 3
DEFAULT_N_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 512

# Generous because it bounds a full training run on a possibly-cold environment,
# not a single call: uv may be resolving and downloading ~3.5GB of CUDA wheels on
# the first invocation. A bound still exists so a hung worker fails the job
# rather than holding the allocation to its wall clock.
WORKER_TIMEOUT_SECONDS: float = 10_800.0


def worker_script() -> AbstractContextManager[Path]:
    """A real filesystem path to the worker, for the lifetime of the context.

    `uv run` needs a path on disk, which a `Traversable` is not obliged to be;
    `as_file` extracts one when needed (e.g. from a zipimport), which is why
    the caller must treat this as a context manager rather than stash the path.
    """
    # Positional: `as_file` is a `functools.singledispatch` function that
    # dispatches on the type of `args[0]` and rejects `path=`.
    return resources.as_file(
        resources.files(anchor="deconvolve") / "baselines" / "_omnifold_worker.py"
    )


def _worker_env() -> dict[str, str]:
    """The worker's environment, with its own directory off `sys.path`.

    The worker's directory (`deconvolve/baselines/`) also contains this module,
    `omnifold.py`, and a script's own directory goes on `sys.path[0]` by
    default -- so the worker's `from omnifold import MLP, DataLoader,
    MultiFold` would otherwise resolve to this module instead of the
    installed `omnifold` package. `PYTHONSAFEPATH=1` (3.11+) stops the
    interpreter prepending that directory; the worker imports nothing local,
    so it loses nothing.
    """
    return os.environ | {"PYTHONSAFEPATH": "1"}


def _invoke(worker: Path, in_path: Path, out_path: Path) -> None:
    """Run the worker, translating the two failures a caller can act on."""
    # Fixed argv, no shell; the interpolated elements are paths this process
    # created. `--no-project` is essential rather than defensive: without it uv
    # resolves the script against this project and runs it in the project
    # environment -- which is the one environment that must never contain
    # TensorFlow, and whose interpreter is whatever the project is checked out
    # at (3.14 by `.python-version`), not the `==3.13.*` the worker pins. The
    # project floor being `>=3.12` does not help: a floor is not a pin, and it
    # is the resolved environment, not the floor, that the script would land in.
    command: list[str] = [
        "uv",
        "run",
        "--no-project",
        worker.as_posix(),
        in_path.as_posix(),
        out_path.as_posix(),
    ]
    try:
        completed: subprocess.CompletedProcess[str] = subprocess.run(  # ruff: ignore[subprocess-without-shell-equals-true]
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=WORKER_TIMEOUT_SECONDS,
            env=_worker_env(),
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "`uv` was not found on PATH. The OmniFold baseline runs its worker "
            "through `uv run`, which provisions the Python 3.13 and TensorFlow "
            "environment this project cannot contain. Install uv, or run the "
            "baseline from a checkout that has it."
        ) from error

    if completed.returncode != 0:
        raise RuntimeError(
            f"The OmniFold worker exited {completed.returncode}.\n"
            f"stderr tail:\n{completed.stderr[-4000:]}"
        )


def _warn_if_on_cpu(device: str) -> None:
    """Warn, rather than error, when the worker did not get a GPU.

    The weights are still correct; a CPU run of a small configuration is
    legitimate. TensorFlow logs its CUDA troubles and carries on by default,
    so this is the only signal a caller gets that it happened.
    """
    if "GPU" in device:
        logger.info("OmniFold worker deconvolve on %s", device)
        return
    logger.warning(
        "OmniFold worker deconvolve on %s, not a GPU. TensorFlow does not raise when "
        "it cannot load its CUDA libraries, it just runs slowly. On Perlmutter "
        "this is what `module load cudatoolkit/12.9` fixes; see "
        "benchmarks/gpu_coexistence.py.",
        device,
    )


def unfold(
    x_data: EventArray,
    x_sim: EventArray,
    z_gen: EventArray,
    z_target: EventArray,
    out_dir: Path,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    worker: Path | None = None,
) -> EventArray:
    """Per-event weights for `z_target`, mean one, from OmniFold.

    `worker` overrides the packaged script, which is how the tests exercise this
    seam without a TensorFlow environment. Nothing in the package passes it.
    """
    with worker_script() as packaged:
        script: Path = worker if worker is not None else packaged
        with tempfile.TemporaryDirectory() as tmp:
            in_path: Path = Path(tmp) / "in.npz"
            out_path: Path = Path(tmp) / "out.npz"
            np.savez(
                file=in_path,
                x_data=x_data,
                x_sim=x_sim,
                z_gen=z_gen,
                z_target=z_target,
                niter=np.array(object=n_iterations),
                epochs=np.array(object=n_epochs),
                batch_size=np.array(object=batch_size),
                out_dir=np.array(object=out_dir.as_posix()),
            )
            _invoke(script, in_path, out_path)
            with np.load(file=out_path, allow_pickle=False) as handle:
                # Same narrowing `data/jets.py` uses: an NpzFile's `__getitem__`
                # carries no element type, and every reader here needs one.
                result: Mapping[str, NDArray[Any]] = cast(
                    typ="Mapping[str, NDArray[Any]]", val=handle
                )
                weights: EventArray = np.asarray(a=result["weights"])
                device: str = (
                    str(object=result["device"]) if "device" in handle else "unknown"
                )
                _record_worker_timings(result)

    _warn_if_on_cpu(device)
    return weights


# The worker's own phases, in the order it runs them, with what each covers.
_WORKER_PHASES: tuple[tuple[str, str], ...] = (
    ("init_seconds", "DataLoaders and the two MLPs"),
    ("unfold_seconds", "MultiFold.Unfold"),
    ("reweight_seconds", "evaluating weights on z_target"),
)


def _record_worker_timings(result: Mapping[str, NDArray[Any]], /) -> None:
    """Fold the worker's breakdown into this run's timing tree.

    These were measured in another interpreter, so `timing.record` enters
    them directly rather than wrapping a block. Called from inside
    `with timing.phase("omnifold")`, so they nest under it.
    """
    for key, detail in _WORKER_PHASES:
        if key not in result:
            continue
        timing.record(key.removesuffix("_seconds"), float(result[key]), detail=detail)
        if key == "unfold_seconds":
            _record_iteration_timings(result)

    logged: list[str] = [
        f"{key.removesuffix('_seconds')}={float(result[key]):.1f}s"
        for key, _ in _WORKER_PHASES
        if key in result
    ]
    if logged:
        logger.info("OmniFold worker timings: %s", ", ".join(logged))


def _record_iteration_timings(result: Mapping[str, NDArray[Any]], /) -> None:
    """One row per MultiFold iteration per step, siblings of `unfold`.

    `timing`'s tree is one level deep in practice,
    so these render beside `unfold` rather than under it. Absent when the
    worker's wrapping found nothing to wrap -- e.g. after a rename inside
    OmniFold -- in which case the `unfold` total still arrives.
    """
    for key, step, what in (
        ("step1_seconds", 1, "detector-level reweighting"),
        ("step2_seconds", 2, "particle-level reweighting"),
    ):
        if key not in result:
            continue
        for iteration, seconds in enumerate(
            iterable=np.atleast_1d(result[key]), start=1
        ):
            timing.record(f"iter{iteration}_step{step}", float(seconds), detail=what)


def _metrics_for(
    config: RunConfig, data: UnfoldingPopulations, weights: EventArray
) -> dict[str, MetricRecord]:
    """Both levels, scored by the same helpers every other baseline uses."""
    metrics: dict[str, MetricRecord] = {}
    for dimension, name in enumerate(iterable=config.variable_names):
        metrics[f"detector_{name}"] = evaluate_dimension(
            data.test.data[:, dimension], data.test.mc.x[:, dimension], weights
        )
        metrics[f"particle_{name}"] = evaluate_dimension(
            data.test.truth[:, dimension], data.test.mc.z[:, dimension], weights
        )
    return metrics


def evaluate_single(
    run_dir: Path,
    force: bool = False,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, MetricRecord]:
    """Run OmniFold on one run's dataset and save its comparison metrics."""
    out_path: Path = artifacts_dir(run_dir) / "metrics_omnifold.json"
    weights_path: Path = artifacts_dir(run_dir) / "omnifold_weights.npz"

    if out_path.exists() and weights_path.exists() and not force:
        logger.info(
            "%s: metrics_omnifold.json exists, skipping (use --force)", run_dir.name
        )
        return cast(
            typ="dict[str, MetricRecord]", val=json.loads(s=out_path.read_text())
        )

    with timing.phase("parse_config"):
        raw_config: object = json.loads(s=(run_dir / "config.json").read_text())
        config: RunConfig = parse_run_config(raw_config)

    logger.info(
        "%s: running OmniFold (niter=%d, epochs=%d)...",
        run_dir.name,
        n_iterations,
        n_epochs,
    )

    detail: str = f"{config.dataset.value} (n={config.n_samples}, dim={config.dim})"
    with timing.phase("data", detail=detail):
        data: UnfoldingPopulations = load_populations(config)

    # The weights are a function of particle-level MC only, and are evaluated on
    # the test split so nothing the unfolding was fit on is scored.
    with timing.phase("omnifold", detail=f"niter={n_iterations}, epochs={n_epochs}"):
        weights: EventArray = unfold(
            x_data=data.fit.data,
            x_sim=data.fit.mc.x,
            z_gen=data.fit.mc.z,
            z_target=data.test.mc.z,
            out_dir=artifacts_dir(run_dir),
            n_iterations=n_iterations,
            n_epochs=n_epochs,
            batch_size=batch_size,
        )

    with timing.phase("evaluate", detail=f"{len(config.variable_names)} variables"):
        metrics: dict[str, MetricRecord] = _metrics_for(config, data, weights)

    json.dump(obj=metrics, fp=out_path.open(mode="w"), indent=2)
    # Keyword, not positional: np.savez names positional arrays "arr_0", and
    # workflows.train reads this file back as `["weights"]`.
    np.savez(weights_path, weights=weights)
    logger.info(
        "%s: saved OmniFold metrics to %s and weights to %s",
        run_dir.name,
        out_path,
        weights_path,
    )
    render_metrics(f"{run_dir.name} [OmniFold]", metrics, list(config.variable_names))

    # Its own file, not `timings.json`: `timing.write` merges by phase name
    # alone, and this pass's `data`/`evaluate` phases would overwrite the
    # training pass's rows.
    timing.report()
    timing.write(
        run_dir,
        pass_name="omnifold",  # ruff: ignore[hardcoded-password-func-arg]
        filename="timings_omnifold.json",
    )
    return metrics


def evaluate_runs(
    run_dir: Path = Path("runs"),
    force: bool = False,
    n_iterations: int = DEFAULT_N_ITERATIONS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    apply_to_runs(
        run_dir,
        evaluate_one=lambda run_dir: evaluate_single(
            run_dir, force, n_iterations, n_epochs, batch_size
        ),
        description="evaluate with OmniFold",
        log=logger,
    )
