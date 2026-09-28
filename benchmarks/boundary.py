"""Where a run's wall clock actually goes, and how much of it jnp could claim.

This existed to answer one question: is it worth porting the scipy metrics in
`deconvolve.evaluation.evaluate` to jnp now that nothing forces the host/device
split any more?
The answer was yes and the port has happened, so what this measures now is the
residue -- the npz write, and the per-dimension divergence reductions that stay
on the host in float64 because they are free there. The number to read is still
a *fraction*, not a duration: a faster GPU shrinks the training term and leaves
the host term alone, so the ratio is what generalises.

Three things this is careful about, each of which an earlier version got wrong:

* `n_epochs` is baked into the trace -- `carry.epoch < n_epochs` in the loop
  condition and `jnp.zeros((n_epochs, ...))` for the history buffer -- and
  `_run` builds a fresh `jax.jit(lambda ...)` per call, so *every* `train()`
  call pays a full XLA compile. Timing one call and calling it "a run" charges
  compile to the training term and flatters it. Two calls at different epoch
  counts separate the two by subtraction.
* The metrics are timed by calling `deconvolve.evaluation.evaluate`'s own
  helpers, on the split `evaluate` actually scores (test, ~20% of the sample)
  and over the same 12 passes it makes: three metrics, two levels, before and
  after. Re-implementing
  that inline is how the earlier version came to measure four passes over five
  times too many rows. Note that `evaluate_run` itself no longer makes those 12
  passes --- it goes through `_metrics_per_dim`, which shares one histogram
  between the two divergences --- so this is an upper bound on what it spends.
* Metric cost is fixed per run while training cost scales with epochs, so the
  fraction is meaningless without the epoch count attached to it.
"""

from __future__ import annotations

import io
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import deconvolve  # ruff: ignore[unused-import]  -- pins KERAS_BACKEND/x64 before keras or jax load
import jax
import numpy as np
from deconvolve.coretypes import EVENT_DTYPE, Events, Populations
from deconvolve.data import DeconvolveDataset
from deconvolve.data.device import TrainSplit

# Private on purpose: the point is to time what `evaluate` runs, not a
# re-implementation of it that can drift.
from deconvolve.evaluation.evaluate import _js_per_dim, _triangular_per_dim, _wd_per_dim
from deconvolve.training.engine import train

if TYPE_CHECKING:
    from collections.abc import Generator

    from deconvolve.coretypes import DatasetSplits, EventArray
    from numpy.typing import NDArray


N_SAMPLES: int = 500_000
DIM: int = 6
EPOCHS: int = 100

results: dict[str, float] = {}


@contextmanager
def phase(name: str, /) -> Generator[None]:
    start: float = time.perf_counter()
    try:
        yield
    finally:
        # Record even on failure, so a crash mid-benchmark still reports the
        # phases that completed rather than losing the whole run.
        results[name] = time.perf_counter() - start


def _sample(rng: np.random.Generator) -> Populations:
    """A fourth data source, narrowing at construction like the other three."""
    z_true: EventArray = rng.normal(size=(N_SAMPLES, DIM)).astype(EVENT_DTYPE)
    z_gen: EventArray = rng.normal(loc=0.5, size=(N_SAMPLES, DIM)).astype(EVENT_DTYPE)

    def smear(a: EventArray, /) -> EventArray:
        return (a + 0.5 * rng.normal(size=a.shape)).astype(EVENT_DTYPE)

    return Populations(
        mc=Events(z=z_gen, x=smear(z_gen)), data=smear(z_true), truth=z_true
    )


def _time_metrics(test: Populations, weights: EventArray) -> None:
    """The 12 passes `evaluate._evaluate_one` makes, timed by metric."""
    levels: list[tuple[EventArray, EventArray]] = [
        (test.data, test.mc.x),
        (test.require_truth(), test.mc.z),
    ]
    for label, fn in (
        ("wasserstein", _wd_per_dim),
        ("jensenshannon", _js_per_dim),
        ("triangular", _triangular_per_dim),
    ):
        with phase(f"  evaluate: {label} (2 levels x before+after)"):
            for ref, comp in levels:
                _: NDArray[np.double] = fn(ref=ref, comp=comp)
                _: NDArray[np.double] = fn(ref=ref, comp=comp, weights=weights)


def main() -> None:
    print(f"backend: {jax.default_backend()}  devices: {jax.devices()}")

    rng: np.random.Generator = np.random.default_rng(seed=0)
    pops: Populations = _sample(rng)
    splits: DatasetSplits = DeconvolveDataset(batch_size=1024, seed=0).splits_from_data(
        data=pops.interleave()
    )
    # `evaluate` scores the test split, not the whole sample.
    test: Populations = splits.test.as_arrays().partition()
    weights: EventArray = np.abs(rng.normal(loc=1.0, size=len(test.mc))).astype(
        EVENT_DTYPE
    )

    # --- device side: scales with the accelerator ---
    with phase("host -> device transfer (once per run)"):
        split: TrainSplit = TrainSplit.from_zxy(splits.train.as_arrays())
        jax.block_until_ready(x=split.z)

    kw: dict[str, int] = {"dim": DIM, "hidden_units": 64, "n_layers": 2, "seed": 0}
    with phase("train(n_epochs=1)   [compile + 1 epoch]"):
        train(splits, n_epochs=1, **kw)  # ty: ignore[invalid-argument-type]
    with phase(f"train(n_epochs={EPOCHS}) [compile + {EPOCHS} epochs]"):
        train(splits, n_epochs=EPOCHS, **kw)  # ty: ignore[invalid-argument-type]

    # --- host side: does NOT scale with the accelerator ---
    _time_metrics(test, weights)

    # Mirrors what `DeconvolveDataset` actually writes. It stopped compressing --
    # incompressible floats, ~20x read tax for a few percent of disk -- so
    # timing `savez_compressed` here would measure a path the pipeline no
    # longer takes, and would overstate the host term this benchmark exists
    # to size.
    with phase("np.savez (cache write, once ever)"):
        np.savez(
            file=io.BytesIO(), z=pops.mc.z, x=pops.mc.x, y=splits.train.as_arrays().y
        )

    width: int = max(len(name) for name in results)
    for name, seconds in results.items():
        print(f"{name:<{width}}  {seconds:8.3f}s")

    t1: float = results["train(n_epochs=1)   [compile + 1 epoch]"]
    tn: float = results[f"train(n_epochs={EPOCHS}) [compile + {EPOCHS} epochs]"]
    per_epoch: float = (tn - t1) / (EPOCHS - 1)
    compile_s: float = t1 - per_epoch
    metrics: float = sum(v for k, v in results.items() if k.startswith("  evaluate:"))

    print(f"\n{'XLA compile (once per run, fixed)':<{width}}  {compile_s:8.3f}s")
    print(f"{'per epoch (steady state)':<{width}}  {per_epoch:8.3f}s")
    print(f"{'scipy metrics (once per run, fixed)':<{width}}  {metrics:8.3f}s")

    print("\nmovable share of a run, by epoch count:")
    for n in (EPOCHS, 500, 1000):
        run: float = compile_s + n * per_epoch + metrics
        print(
            f"  {n:>5} epochs: run {run:7.1f}s   "
            f"scipy {metrics / run:5.1%}   compile {compile_s / run:5.1%}"
        )
    print(
        "\nOnly the scipy column is movable, and only the wasserstein and"
        "jensenshannon rows within it -- the triangular metric shares the"
        "histograms JS already builds. Compile is fixed and jnp cannot touch it."
    )


if __name__ == "__main__":
    main()
