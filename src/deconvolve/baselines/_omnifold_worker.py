# /// script
# requires-python = "==3.13.*"
# dependencies = [
#     "numpy>=2.5.2",
#     "omnifold",
#     "tensorflow[and-cuda]; sys_platform == 'linux' and platform_machine == 'x86_64'",
#     "tensorflow; sys_platform == 'darwin'",
# ]
# ///
"""OmniFold, in the only environment it can run in. Never imported.

This file is inside the package but is not part of it. Nothing imports it, and
nothing can: it runs under Python 3.13 with Keras bound to the TensorFlow
backend, and `ran` binds Keras to JAX. Keras binds its backend once per
interpreter, so those two facts are irreconcilable inside one interpreter --
which is the whole reason OmniFold is a subprocess rather than a module. The
project floor is `>=3.12` and so no longer excludes 3.13 on its own; the
backend binding, and the rule that the project environment never holds
TensorFlow, are what keep this file out of process. The PEP 723 header above
is the quarantine; `uv run --no-project` provisions it, and
`deconvolve.baselines.omnifold` on the other side of an `.npz` file is the only caller.

Three constraints the header encodes, each of which is load-bearing:

* `requires-python = "==3.13.*"` because TensorFlow publishes no 3.14 wheels.
  Pinned rather than bounded so the environment cannot drift onto a version
  TensorFlow does not support.
* The `[and-cuda]` extra only on linux/x86_64, because it does not exist for
  macOS and an unconditional marker makes the script unresolvable on a laptop.
* `omnifold` unpinned, since it is the thing under comparison and pinning it
  here would silently freeze the baseline at whatever version was current.

**On Perlmutter this needs `module load cudatoolkit/12.9`.** Without it TF
cannot reach `libcusolver.so.11`, skips registering every GPU, and runs on the
CPU without raising -- so the reported device is written into the result and the
host warns about it. See `benchmarks/gpu_coexistence.py`.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import ArrayLike, NDArray


os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault(key="TF_CPP_MIN_LOG_LEVEL", value="2")


def _as_2d(array: ArrayLike) -> NDArray[np.single]:
    """OmniFold's `DataLoader` wants `(n, d)`; a 1D observable arrives as `(n,)`."""
    values: np.typing.NDArray[np.single] = np.asarray(a=array, dtype=np.single)
    return values[..., np.newaxis] if values.ndim == 1 else values


def _device_of(model: object) -> str:
    """Where TensorFlow actually placed the work.

    Recorded because the failure that matters here is silent: a TF that cannot
    load its CUDA libraries reports no GPU, runs on the CPU, and returns correct
    weights tens of times slower. Nothing raises, so the only way the caller can
    tell is if this string comes back.
    """
    import tensorflow as tf

    del model
    gpus: list[tf.config.PhysicalDevice] = tf.config.list_physical_devices("GPU")
    return gpus[0].name if gpus else "/physical_device:CPU:0"


def _timed_steps(unfold: object) -> tuple[list[float], list[float]]:
    """Wrap `RunStep1`/`RunStep2` so each iteration reports its own duration.

    MultiFold's two steps are the whole of the cost and they are not
    symmetric --- step 1 reweights at detector level, step 2 at particle level
    --- so a single `unfold_seconds` hides which half a long run spent its time
    in, and whether the per-iteration cost is flat or climbing.

    Wrapping rather than reading: OmniFold exposes no timing of its own. The
    attributes are checked before being replaced, so a future rename degrades
    to an empty breakdown and the totals still arrive, rather than taking the
    baseline down over its own instrumentation.
    """
    step1: list[float] = []
    step2: list[float] = []

    for name, into in (("RunStep1", step1), ("RunStep2", step2)):
        original: Callable | None = getattr(unfold, name, None)
        if not callable(original):
            continue

        def timed(
            iteration: int,
            _original: Callable = original,
            _into: list[float] = into,
        ) -> object:
            started: float = time.perf_counter()
            try:
                return _original(iteration)
            finally:
                _into.append(time.perf_counter() - started)

        setattr(unfold, name, timed)

    return step1, step2


def run(payload: dict[str, NDArray[np.single]], out_path: Path) -> None:
    import keras
    from omnifold import MLP, DataLoader, MultiFold
    from omnifold.net import weighted_binary_crossentropy

    # MultiFold saves and reloads its own checkpoints between iterations, and the
    # loss it compiled with is not in Keras's default custom-object scope.
    keras.saving.get_custom_objects()["weighted_binary_crossentropy"] = (
        weighted_binary_crossentropy
    )

    started: float = time.perf_counter()
    x_data: NDArray[np.single] = _as_2d(array=payload["x_data"])
    x_sim: NDArray[np.single] = _as_2d(array=payload["x_sim"])
    z_gen: NDArray[np.single] = _as_2d(array=payload["z_gen"])
    z_target: NDArray[np.single] = (
        _as_2d(array=payload["z_target"]) if "z_target" in payload else z_gen
    )

    out_dir = Path(str(object=payload["out_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)

    unfold = MultiFold(
        "omnifold_baseline",
        MLP(x_data.shape[1]),
        MLP(x_data.shape[1]),
        DataLoader(reco=x_data),
        DataLoader(reco=x_sim, gen=z_gen),
        log_folder=str(object=out_dir),
        weights_folder=str(object=out_dir / "omnifold_checkpoints"),
        niter=int(payload["niter"]),
        epochs=int(payload["epochs"]),
        batch_size=int(payload["batch_size"]),
        verbose=False,
    )
    init_seconds: float = time.perf_counter() - started

    step1_seconds, step2_seconds = _timed_steps(unfold)
    unfold_started: float = time.perf_counter()
    unfold.Unfold()
    unfold_seconds: float = time.perf_counter() - unfold_started

    # `model2` is the particle-level (step 2) classifier: the one that maps gen
    # features to weights, and the only one whose output is an unfolding. Step 1
    # lives at detector level and would not be applicable to `z_target`.
    reweight_started: float = time.perf_counter()
    weights: NDArray[np.single] = (
        unfold.reweight(z_target, unfold.model2).astype(np.single).ravel()
    )
    # Mean one, matching the normalization `ran`'s own weights carry, so the two
    # are comparable without the caller rescaling either.
    weights = weights / weights.mean()
    reweight_seconds: float = time.perf_counter() - reweight_started

    np.savez(
        file=out_path,
        weights=weights,
        device=np.array(object=_device_of(model=unfold.model2)),
        tf_version=np.array(object=__import__(name="tensorflow").__version__),
        init_seconds=np.array(object=init_seconds),
        unfold_seconds=np.array(object=unfold_seconds),
        reweight_seconds=np.array(object=reweight_seconds),
        step1_seconds=np.asarray(a=step1_seconds, dtype=np.double),
        step2_seconds=np.asarray(a=step2_seconds, dtype=np.double),
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name} <input.npz> <output.npz>")
    with np.load(file=sys.argv[1], allow_pickle=False) as payload:
        run(
            payload={key: payload[key] for key in payload.files},
            out_path=Path(sys.argv[2]),
        )


if __name__ == "__main__":
    main()
