"""How small an MMD^2 difference can the selection criterion actually resolve?

This number decides **admissibility**. An arm whose `mmd_test` sits within the
floor of the best is one the criterion cannot distinguish from a perfect match,
so it is a legitimate setting; an arm outside it is one a truth-free pipeline
would refuse to ship however well it scores against truth. The dispersion sweep
turned that into the primary question -- `--lambda-dispersion 0.1` reaches jet
mass 78.9% against the shipped 32.6%, and is rejected at 30x the floor.

Until this benchmark the floor in use was **an extrapolation**: ~5e-4 measured
at m=8192, scaled by 1/m to `train.MMD_SUBSAMPLE = 16384`, giving 2.5e-4.
Extrapolating a constant that every admissibility call rests on is the kind of
thing that should be measured, and this measures it.

**The null is by construction, not by assumption.** One population is split into
two disjoint halves, so the true MMD^2 is exactly zero and anything the
estimator reports is its own noise. Comparing `x_data` against `x_sim` would
instead measure a real discrepancy plus noise, with no way to separate them.

The estimator is the unbiased U-statistic, so it is **not a distance and takes
both signs**. The floor is therefore two-sided: an arm at -2e-4 and one at
+2e-4 are equally indistinguishable from a perfect match. `sd` is the number
that matters -- how far one run's value wanders -- not `standard_error`, which
only says how well this benchmark has pinned the mean.

```zsh
uv run benchmarks/mmd_floor.py                              # jets, operating point
uv run benchmarks/mmd_floor.py --m 4096 --m 8192 --m 16384  # check the 1/m scaling
uv run benchmarks/mmd_floor.py --repeats 64 --n-samples 800000
uv run benchmarks/mmd_floor.py --side sim                   # split the MC side instead
```
"""

# pyrefly: ignore-errors[unknown-argument-type]
# -- numpy stubs return Any for elementwise ops and boolean indexing
from __future__ import annotations

import os

os.environ.setdefault("KERAS_BACKEND", "jax")  # pyrefly: ignore[unused-call-result]

import argparse
import logging
from typing import TYPE_CHECKING, NamedTuple

import deconvolve  # ruff: ignore[unused-import]  -- pins the backend and the dtype
import jax.numpy as jnp
import numpy as np
from deconvolve.coretypes import SUBSTRUCTURE_VARIABLES, Split
from deconvolve.data import load_jet_dataset
from deconvolve.instrumentation.logging_config import configure_logging
from deconvolve.training.engine import MMD_SUBSAMPLE
from deconvolve.training.mmd import bandwidths, build_cache, weighted_mmd
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

logger: logging.Logger = logging.getLogger("deconvolve.mmd_floor")

#: The value the dispersion analysis has been using, for comparison. It is the
#: 1/m extrapolation of the ~5e-4 measured at m=8192 in `benchmarks/README.md`.
EXTRAPOLATED_FLOOR: float = 2.5e-4

#: Above this share of the pool, repeats stop being independent draws. Measured
#: on 12-dimensional Gaussians the null SD follows the theoretical `m^-1` while
#: `2m/N` stays small (-1.00 and -0.93 at 2m/N <= 0.2) and flattens to `m^-0.74`
#: at 2m/N = 0.4, where successive repeats are largely the same rows. The spread
#: reported there is not the estimator's, so the floor comes out biased.
MAX_POOL_SHARE: float = 0.2


class FloorEstimate(NamedTuple):
    """Repeated null estimates of MMD^2, and what they say about resolution."""

    values: tuple[float, ...]

    @property
    def mean(self) -> float:
        """Should sit at zero: the null is exact, so this is a bias check."""
        return float(np.mean(self.values))

    @property
    def sd(self) -> float:
        """**The floor.** How far a single run's number wanders under the null.

        This is what decides whether two arms differ, so it is the number to
        compare an arm's `mmd_test` against -- not the standard error below.
        """
        return float(np.std(self.values, ddof=1))

    @property
    def standard_error(self) -> float:
        """How well *this benchmark* has pinned the mean. Not the floor."""
        return float(self.sd / np.sqrt(len(self.values)))

    @property
    def sd_error(self) -> float:
        """How well this benchmark has pinned **the floor**.

        `sd / sqrt(2(n-1))` for a normal sample: ~9% at 64 repeats, ~13% at 32.
        It matters because arms sit within a floor or two of the tie threshold,
        where quoting the floor bare invites reading 1.1 as different from 0.9.
        """
        return float(self.sd / np.sqrt(2.0 * (len(self.values) - 1)))


def null_floor(
    x: NDArray[np.single], /, *, m: int, repeats: int, seed: int
) -> FloorEstimate:
    """Estimate MMD^2 between two disjoint halves of one population.

    Both halves are the same distribution by construction, so the true value is
    exactly zero and the spread over `repeats` independent draws is the
    estimator's own resolution. Each repeat redraws the permutation, so the two
    halves and the subsample within them differ every time; without that the
    spread would be zero and the floor would read as zero.
    """
    if x.shape[0] < 2 * m:
        raise ValueError(
            f"need two disjoint subsamples of {m}, but the population has "
            f"{x.shape[0]} rows"
        )
    share: float = 2 * m / x.shape[0]
    if share > MAX_POOL_SHARE:
        logger.warning(
            "2m/N = %.2f exceeds %.2f: repeats overlap heavily and the spread "
            "they show is not the estimator's. Raise --n-samples.",
            share,
            MAX_POOL_SHARE,
        )
    values: list[float] = []
    for repeat in range(repeats):
        order = np.random.default_rng(seed * 10_000 + repeat).permutation(x.shape[0])
        left = jnp.asarray(x[order[:m]])
        right = jnp.asarray(x[order[m : 2 * m]])
        # Bandwidths from the reference side, as `train` takes them from the
        # data side of the comparison it is about to make.
        cache = build_cache(left, right, sigmas=bandwidths(left))
        mmd2, _ess = weighted_mmd(cache, jnp.ones(m, dtype=jnp.float32))
        values.append(float(mmd2))
    return FloorEstimate(values=tuple(values))


def fitted_exponent(sds: Mapping[int, float], /) -> float:
    """The measured scaling of the floor with subsample size.

    Reported rather than assumed. Theory gives `m^-1` for the unbiased
    U-statistic under the null, and the extrapolation this benchmark replaces
    took that on faith to reach the operating point; whether it held for a
    given run is a fact about that run's pool size, not about theory.
    """
    sizes = sorted(sds)
    if len(sizes) < 2:
        return float("nan")
    return float(
        np.polyfit(
            np.log([float(m) for m in sizes]), np.log([sds[m] for m in sizes]), 1
        )[0]
    )


def _report(estimates: dict[int, FloorEstimate], side: str) -> None:
    console = Console()
    table = Table(title=f"MMD² resolution floor — null from the {side} side")
    table.add_column(header="m", justify="right")
    table.add_column(header="mean", justify="right")
    table.add_column(header="SD (the floor)", justify="right")
    table.add_column(header="± on the floor", justify="right")
    table.add_column(header="1/m prediction", justify="right")
    table.add_column(header="bias / SE", justify="right")

    reference_m = min(estimates)
    reference_sd = estimates[reference_m].sd
    exponent = fitted_exponent({m: e.sd for m, e in estimates.items()})
    for m, estimate in sorted(estimates.items()):
        predicted = reference_sd * reference_m / m
        table.add_row(
            str(m),
            f"{estimate.mean:+.3e}",
            f"{estimate.sd:.3e}",
            f"{estimate.sd_error:.1e}  ({100 * estimate.sd_error / estimate.sd:.0f}%)",
            "—" if m == reference_m else f"{predicted:.3e}",
            f"{estimate.mean / estimate.standard_error:+.1f}",
        )
    console.print(table)
    if len(estimates) > 1:
        logger.info(
            "Measured scaling: SD ~ m^%.2f (theory is m^-1 for the unbiased "
            "U-statistic under the null; a shallower exponent means the repeats "
            "overlapped -- check the 2m/N warnings).",
            exponent,
        )

    if MMD_SUBSAMPLE in estimates:
        measured = estimates[MMD_SUBSAMPLE].sd
        logger.info(
            "At the operating point m=%d the floor is %.3e +- %.1e; the "
            "extrapolation in use was %.3e, a factor of %.2f %s.",
            MMD_SUBSAMPLE,
            measured,
            estimates[MMD_SUBSAMPLE].sd_error,
            EXTRAPOLATED_FLOOR,
            max(measured, EXTRAPOLATED_FLOOR) / min(measured, EXTRAPOLATED_FLOOR),
            "too tight" if measured > EXTRAPOLATED_FLOOR else "too loose",
        )
        logger.info(
            "The floor is two-sided: |mmd_test| < %.3e is indistinguishable from "
            "a perfect match, and an arm outside it is one the criterion rejects.",
            measured,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ = parser.add_argument(
        "--m",
        type=int,
        action="append",
        default=None,
        help=f"subsample size; repeatable. Default {MMD_SUBSAMPLE} (operating point)",
    )
    _ = parser.add_argument("--repeats", type=int, default=32)
    _ = parser.add_argument("--n-samples", type=int, default=400_000)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument(
        "--side",
        choices=("data", "sim"),
        default="data",
        help="which population to split; both are valid nulls",
    )
    args = parser.parse_args()
    configure_logging(level="info")

    m_values: list[int] | None = args.m
    n_samples: int = args.n_samples
    repeats: int = args.repeats
    seed: int = args.seed
    side: str = args.side

    sizes: Sequence[int] = sorted(set(m_values or [MMD_SUBSAMPLE]))
    splits, _dim, _std = load_jet_dataset(
        n_samples=n_samples,
        batch_size=1024,
        variables=SUBSTRUCTURE_VARIABLES,
        seed=42,
    )
    # Every split, not just test. The floor is a property of the estimator,
    # the distribution and `m` -- not of which rows a trained model was scored
    # on -- so restricting to test buys nothing and costs a factor of five in
    # pool size. At the operating point m=16384 the test split alone cannot
    # reach 2m/N <= 0.2 without more events than the Zenodo release holds.
    zxy = splits.select(Split.TRAIN | Split.VAL | Split.TEST)
    nature = zxy.y == 1
    x = zxy.x[nature] if side == "data" else zxy.x[~nature]
    logger.info(
        "%s side: %d rows x %d observables, %d repeats",
        side,
        x.shape[0],
        x.shape[1],
        repeats,
    )

    estimates: dict[int, FloorEstimate] = {}
    for m in sizes:
        if x.shape[0] < 2 * m:
            logger.warning(
                "m=%d needs %d rows and only %d are available; skipping "
                "(raise --n-samples)",
                m,
                2 * m,
                x.shape[0],
            )
            continue
        estimates[m] = null_floor(x, m=m, repeats=repeats, seed=seed)
        logger.info("m=%d done: sd %.3e", m, estimates[m].sd)
    if not estimates:
        raise SystemExit("no subsample size fit the data; raise --n-samples")
    _report(estimates, side)


if __name__ == "__main__":
    main()
