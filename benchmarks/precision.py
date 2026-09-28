# ruff: file-ignore[module-import-not-at-top-of-file]

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from typing import Any, Final

    from deconvolve.coretypes import DatasetSplits, DeconvolveModel, EventArray
    from deconvolve.training.engine import TrainResult
    from numpy.typing import NDArray

    class ModelBuilder(Protocol):
        """The shape of
        `deconvolve.training.models.build_{generator,discriminator}`.

        Spelled out rather than `Callable[..., DeconvolveModel]` so that
        rebinding the module attributes below type-checks instead of needing a
        suppression.
        """

        def __call__(
            self, dim: int = 1, hidden_units: int = 64, n_layers: int = 2
        ) -> DeconvolveModel: ...


if len(sys.argv) < 3:
    raise SystemExit("usage: precision.py <float32|float64> <seed>")
DTYPE: Final[str] = sys.argv[1]
if DTYPE not in {"float32", "float64"}:
    raise SystemExit(f"usage: precision.py <float32|float64> <seed>; got {DTYPE!r}")
SEED: Final[int] = int(sys.argv[2])

# Both must be set before keras or jax load anywhere. This is the whole reason
# the dtype is a process argument rather than a function parameter.
os.environ["KERAS_BACKEND"] = "jax"
os.environ["JAX_ENABLE_X64"] = str(object=int(DTYPE == "float64"))

import deconvolve  # ruff: ignore[unused-import] -- import order is load-bearing; see above
import deconvolve.training.engine as engine_module
import numpy as np
from deconvolve.coretypes import (
    Events,
    Populations,
)
from deconvolve.data import DeconvolveDataset
from scipy.stats import (
    wasserstein_distance,
)

N_SAMPLES: int = 200_000
DIM: int = 6
EPOCHS: int = 40
REPO_ROOT: Path = Path(__file__).resolve().parents[1]


def _builders_at(dtype: str) -> tuple[ModelBuilder, ModelBuilder]:
    """Rebuild the model factories at `dtype` without mutating the source file."""
    source: str = (
        REPO_ROOT / "src" / "deconvolve" / "training" / "models.py"
    ).read_text()
    namespace: dict[str, Any] = {}
    exec(  # ruff: ignore[exec-builtin] -- module's own source, recompiled with one literal changed
        compile(
            source=source.replace('"float32"', f'"{dtype}"'),
            filename="deconvolve/training/models.py",
            mode="exec",
        ),
        namespace,
    )
    return (
        cast(typ="ModelBuilder", val=namespace["build_generator"]),
        cast(typ="ModelBuilder", val=namespace["build_discriminator"]),
    )


def main() -> None:
    scalar: type[np.double | np.single] = np.double if DTYPE == "float64" else np.single
    generator, discriminator = _builders_at(DTYPE)
    # Rebinding the module's builders to the same factories recompiled at
    # another dtype -- the measurement. Left unannotated: pyrefly rejects an
    # annotation on a non-self attribute.
    engine_module.build_generator = generator  # ty: ignore[invalid-assignment]
    engine_module.build_discriminator = discriminator  # ty: ignore[invalid-assignment]

    rng: np.random.Generator = np.random.default_rng(seed=0)
    z_true: NDArray[scalar] = rng.normal(size=(N_SAMPLES, DIM)).astype(dtype=scalar)
    z_gen: NDArray[scalar] = rng.normal(loc=0.5, size=(N_SAMPLES, DIM)).astype(
        dtype=scalar
    )
    # `Populations` is pinned to the package dtype, so the arrays are cast on
    # the way in rather than through a container-level `astype`.
    # This benchmark deliberately runs the pipeline at float64 as well, so the
    # arrays here can be wider than `EventArray`'s pinned float32. The cast is
    # the measurement, not a mistake.
    pops = Populations(
        mc=Events(
            z=cast("EventArray", z_gen),
            x=cast(
                "EventArray",
                (z_gen + 0.5 * rng.normal(size=(N_SAMPLES, DIM))).astype(dtype=scalar),
            ),
        ),
        data=cast(
            "EventArray",
            (z_true + 0.5 * rng.normal(size=(N_SAMPLES, DIM))).astype(dtype=scalar),
        ),
        truth=cast("EventArray", z_true),
    )

    splits: DatasetSplits = DeconvolveDataset(batch_size=1024, seed=0).splits_from_data(
        data=pops.interleave()
    )
    result: TrainResult = engine_module.train(
        splits,
        dim=DIM,
        hidden_units=64,
        n_layers=2,
        n_epochs=EPOCHS,
        seed=SEED,
    )

    raw: NDArray[np.double] = (
        np.asarray(a=result.g(pops.mc.z)).ravel().astype(dtype=np.double)
    )
    weights: NDArray[np.double] = raw * len(raw) / raw.sum()

    # Score in float64 in both arms: only the unfolding dtype is under test,
    # so the measuring stick has to be the same one for each.
    truth: NDArray[np.double] = pops.truth.astype(dtype=np.double)
    gen: NDArray[np.double] = pops.mc.z.astype(dtype=np.double)
    improvements: list[float] = []
    for i in range(DIM):
        before: float = wasserstein_distance(truth[:, i], gen[:, i])
        after: float = wasserstein_distance(truth[:, i], gen[:, i], v_weights=weights)
        improvements.append(100.0 * (before - after) / before)

    _ = sys.stdout.write(
        f"SUMMARY dtype={DTYPE} seed={SEED} "
        f"mean_improvement={np.mean(improvements):.4f} "
        f"min_improvement={np.min(improvements):.4f} "
        f"val_loss={result.history['val_d'][-1]:.10f} "
        f"epochs={len(result.history['val_d'])}\n"
    )


if __name__ == "__main__":
    main()
