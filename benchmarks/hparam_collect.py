"""Compare hyperparameter arms with the pairing and the power they need.

Every negative result in `benchmarks/README.md` under "What was ruled out" was
an n=1 arm against an n=1 baseline at a *different* initialization seed. The
`lr_d` sweep is the clearest case: four arms across a 30x range, each a single
run, each at its own seed. Against the measured per-run spread that test could
not have found anything.

The spread, from the six default-configuration runs in `runs/`:

| response variable                    |   SD | n/arm for 2pp |
| ------------------------------------ | ---: | ------------: |
| particle jet mass                    | 7.16 |           201 |
| mean over all six particle variables | 1.91 |            14 |
| mean excluding jet mass              | 1.49 |             9 |

The choice of response variable is worth a factor of fourteen in compute, so
this scores the **mean over observables** and prints the per-observable values
beside it rather than tuning on any one of them. `--exclude m` drops jet mass
from the aggregate while still reporting it, because §4 of the README measures
a non-universal Herwig/Pythia response that limits mass for reasons no
hyperparameter reaches.

Arms are **paired on the initialization seed**, which costs nothing but buys
less than it looks like it should. Measured on the first `lr_g` sweep, the
between-arm correlation is r = +0.22 and +0.04, so pairing cut the standard
error by 11.5% and 1.3%. Changing the knob decorrelates the trajectories almost
completely: the seed's effect is not an offset that survives it. **Budget arms
at the unpaired n.**

`data_seed` is pinned across a sweep so the arms differ in one thing. The cost
is that the spread reported here is init-only variance *at one batch order*,
which is not the ensemble uncertainty a measurement would quote — on that same
sweep the per-epoch criterion curves of eight seeds correlate at r = +0.88,
because they all saw the identical batch sequence. Vary `data_seed` instead of
`seed` to measure the other half.

Which knob defines an arm is not configured: `varying_keys` reads it off the
saved configs, ignoring `seed` (that is the pairing key) and the outcomes
`config.json` records next to the settings. So this works unchanged for the
`lr_g` sweep and for the weight-decay and output-activation knobs that follow.

Read a trend across levels, not an argmax. At n=8 against SD 1.9 the best of
three arms is usually the luckiest, not the best.

```zsh
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_lrg_2026-08-30T...
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_lrg_... --exclude m
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_lrg_... --baseline lr_g=0.0001
```
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from deconvolve.coretypes import artifacts_dir
from deconvolve.instrumentation.logging_config import configure_logging
from rich.console import Console
from rich.table import Table
from scipy import stats

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray

logger: logging.Logger = logging.getLogger("deconvolve.hparam")

#: The two independent randomness axes of CLAUDE.md's Seeding section:
#: `seed` sets initialization, `data_seed` sets shuffle, split and batch order.
#: Both are nuisance axes, so neither can ever define an arm -- a sweep that
#: replicates over one of them would otherwise put every run in its own arm and
#: leave nothing to pair. Which one indexes the replicates is `--pair-on`.
SEED_KEYS: frozenset[str] = frozenset({"seed", "data_seed"})
PAIRING_KEY = "seed"

#: The MMD^2 estimator's resolution at `train.MMD_SUBSAMPLE = 16384`, measured
#: by `benchmarks/mmd_floor.py` rather than extrapolated. It is the standard
#: deviation of the estimator under an exact null, so it is a noise *scale* and
#: not a threshold -- see `tied_to_best`. Measured at 1.159e-4 +- 1.0e-5 (64
#: repeats, 12 jet observables, 2m/N = 0.08), against the 2.5e-4 that had been
#: extrapolated from m=8192 by 1/m.
MMD_FLOOR: float = 1.159e-4

#: How many floors an arm may sit from the best before the criterion is taken
#: to have separated them. Two, because the floor is one sigma of a zero-mean
#: estimator: at one, a perfectly matching arm is called separated half the
#: time.
TIE_SIGMAS: float = 2.0

#: `config.json` records what a run produced next to what it was told to do.
#: These differ in every run because they are outcomes; treating them as knobs
#: would put every run in an arm of its own and leave nothing to pair.
OUTPUT_KEYS: frozenset[str] = frozenset(
    {
        "best_epoch",
        "mmd_test",
        "mmd_sigmas_detector",
        "mmd_sigmas_particle",
    }
)


@dataclass(frozen=True)
class RunRecord:
    """One finished run: what it was asked to do, and what it achieved."""

    name: str
    config: dict[str, Any]
    particle: dict[str, float]
    #: `val_ess` at the selected epoch, over the MMD subsample. `nan` for a run
    #: saved before the field existed, or one whose history is missing.
    ess: float = float("nan")


@dataclass(frozen=True)
class ArmSummary:
    """One arm's location and, just as importantly, its spread."""

    label: str
    n: int
    mean: float
    sd: float
    standard_error: float
    per_observable: dict[str, float]
    #: Mean effective sample size -- the mechanism variable for the dispersion
    #: penalty, and the thing a coefficient is calibrated against.
    ess: float
    #: Mean `mmd_test`: the detector-level criterion the run was selected on.
    #: Admissibility, not performance -- see MMD_FLOOR.
    mmd_test: float
    per_seed: dict[int, float]
    per_seed_by_observable: dict[str, dict[int, float]]


@dataclass(frozen=True)
class Paired:
    """A within-pair difference against the baseline arm."""

    n: int
    delta: float
    standard_error: float
    t_statistic: float
    p_value: float


def particle_improvements(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Pull the particle-level Wasserstein improvements out of `metrics.json`.

    A run without truth records no `particle_*` entries at all. That is not an
    error -- a real measurement has no answer key -- it just is not something
    this tool can score, so it comes back empty and the caller drops it.
    """
    return {
        key.removeprefix("particle_"): cast(
            "float", value["wasserstein_improvement_pct"]
        )
        for key, value in metrics.items()
        if key.startswith("particle_")
    }


def varying_keys(configs: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    """Name the knobs that actually differ across a set of runs."""
    candidates: set[str] = {
        key
        for config in configs
        for key in config
        if key not in SEED_KEYS and key not in OUTPUT_KEYS
    }
    # Values include lists (`variables`, and a 12-observable arm is a real
    # knob), so compare their JSON rather than requiring them to be hashable.
    return tuple(
        sorted(
            key
            for key in candidates
            if len({json.dumps(config.get(key), sort_keys=True) for config in configs})
            > 1
        )
    )


def arm_label(config: Mapping[str, Any], keys: Sequence[str]) -> str:
    """Name an arm by the knobs that vary, or `baseline` when none do."""
    if not keys:
        return "baseline"
    return " ".join(f"{key}={config.get(key)}" for key in keys)


def load_records(root: Path) -> list[RunRecord]:
    """Read every finished run directly under `root`.

    A run that died before writing `metrics.json` is skipped rather than
    raised on: one node failure in a packed sweep must cost its own point and
    nothing else. An empty directory *is* raised on, because a sweep that
    produced nothing is a failure the caller should hear about rather than an
    empty table.
    """
    records: list[RunRecord] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = artifacts_dir(run_dir) / "metrics.json"
        if not (config_path.exists() and metrics_path.exists()):
            logger.warning("%s: incomplete, skipping", run_dir.name)
            continue
        particle = particle_improvements(json.loads(metrics_path.read_text()))
        if not particle:
            logger.warning("%s: no particle-level metrics, skipping", run_dir.name)
            continue
        config = json.loads(config_path.read_text())
        records.append(
            RunRecord(
                name=run_dir.name,
                config=config,
                particle=particle,
                ess=_selected_ess(run_dir, config),
            )
        )
    if not records:
        raise FileNotFoundError(f"{root}: no runs with particle-level metrics")
    return records


def tied_to_best(mmd_test: float, /, *, best: float, floor: float) -> bool:
    """Can the selection criterion tell this arm from the best one?

    `averaging.py`'s tied-set question, asked across arms instead of epochs. An
    arm the criterion cannot separate from the best is a legitimate setting; one
    it can separate is a setting a truth-free pipeline would refuse however well
    it scores against truth.

    A **difference**, not a magnitude. The floor is the standard deviation of a
    zero-mean estimator, so comparing `|mmd_test|` against it asks whether the
    arm matches perfectly rather than whether it matches as well as anything
    available -- and at the measured floor that rule calls even the shipped
    setting inadmissible, which is how the error was found.

    Signed, because the unbiased estimator takes both signs: `-1.2e-4` and
    `+1.2e-4` are two floors apart, not zero apart.
    """
    if np.isnan(mmd_test) or np.isnan(best):
        return True
    return abs(mmd_test - best) <= TIE_SIGMAS * floor


def _nanmean(values: Sequence[float]) -> float:
    """Mean of what is present, or `nan` when nothing is."""
    if all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def _selected_ess(run_dir: Path, config: Mapping[str, Any]) -> float:
    """`val_ess` at the epoch selection restored, not the last one.

    A missing history is not an error: the field postdates several runs in
    `runs/`, and a run that cannot report ESS can still be scored.
    """
    history_path = artifacts_dir(run_dir) / "history.npz"
    best_epoch = config.get("best_epoch")
    if not history_path.exists() or best_epoch is None:
        return float("nan")
    with np.load(history_path) as history:
        if "val_ess" not in history:
            return float("nan")
        curve: NDArray[np.double] = cast("NDArray[np.double]", history["val_ess"])
        if not 0 <= int(best_epoch) < len(curve):
            return float("nan")
        return float(curve[int(best_epoch)])


def _aggregate(particle: Mapping[str, float], exclude: Iterable[str]) -> float:
    """The scored response: the mean improvement over observables."""
    dropped = set(exclude)
    kept = [v for k, v in particle.items() if k not in dropped]
    if not kept:
        raise ValueError("every observable was excluded")
    return float(np.mean(kept))


def summarize_arm(
    label: str,
    records: Sequence[RunRecord],
    exclude: Iterable[str] = (),
    pair_on: str = PAIRING_KEY,
) -> ArmSummary:
    """Locate one arm and report the seed spread it was measured against.

    The pairing key has to actually index the runs. Pointed at the axis a sweep
    held *fixed*, the dict below collapses every run in the arm onto one key: the
    reported mean becomes a single run's while `n` still counts all of them, and
    the SD comes back `nan`. Presenting one run as an eight-run mean is the
    failure this whole tool exists to prevent, so it raises.
    """
    dropped = tuple(exclude)
    per_seed = {
        int(record.config[pair_on]): _aggregate(record.particle, dropped)
        for record in records
    }
    if len(per_seed) != len(records):
        raise ValueError(
            f"arm {label!r}: {len(records)} runs share {len(per_seed)} distinct "
            f"{pair_on} values, so it cannot index them -- pass --pair-on for the "
            f"axis this sweep replicated over"
        )
    values = np.array(list(per_seed.values()), dtype=np.double)
    # ddof=1: these are a sample of the seed distribution, not the whole of it.
    sd = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    observables = sorted({name for record in records for name in record.particle})
    return ArmSummary(
        label=label,
        n=len(records),
        mean=float(np.mean(values)),
        sd=sd,
        standard_error=sd / np.sqrt(len(values)) if len(values) > 1 else float("nan"),
        # Reported for every observable, including the excluded ones: dropping
        # jet mass from the score is not a reason to stop looking at it.
        ess=_nanmean([r.ess for r in records]),
        mmd_test=_nanmean(
            [float(r.config.get("mmd_test", float("nan"))) for r in records]
        ),
        per_observable={
            name: float(
                np.mean([r.particle[name] for r in records if name in r.particle])
            )
            for name in observables
        },
        per_seed=per_seed,
        per_seed_by_observable={
            name: {
                int(r.config[pair_on]): r.particle[name]
                for r in records
                if name in r.particle
            }
            for name in observables
        },
    )


def observable_deltas(baseline: ArmSummary, arm: ArmSummary) -> dict[str, Paired]:
    """Test each observable on its own, not only the aggregate.

    The aggregate is the right thing to *score* -- it is the low-variance
    response, worth a factor of fourteen in runs against any single observable
    -- but it is the wrong thing to test alone. Averaging twelve observables
    divides a move in one of them by twelve, which can put a real effect inside
    the noise of the mean.

    Read these with the multiplicity in mind: twelve observables give twelve
    chances at a small p-value, so one at p ~ 0.05 is what a null looks like.
    """
    shared = sorted(
        set(baseline.per_seed_by_observable) & set(arm.per_seed_by_observable)
    )
    deltas: dict[str, Paired] = {}
    for name in shared:
        try:
            deltas[name] = paired_delta(
                baseline.per_seed_by_observable[name], arm.per_seed_by_observable[name]
            )
        except ValueError:
            continue
    return deltas


def paired_delta(baseline: Mapping[int, float], arm: Mapping[int, float]) -> Paired:
    """Compare two arms on the seeds they share.

    Pairing is free, and on the first `lr_g` sweep it bought 11.5% and 1.3% of
    standard error (implied r = +0.22 and +0.04). It is kept because it cannot
    hurt, not because it carries the comparison -- plan replicates as though it
    were absent. Seeds only one arm deconvolve are dropped rather than compared across
    pairs, which is what makes a lost run cost one pair instead of the
    comparison.
    """
    shared = sorted(set(baseline) & set(arm))
    if len(shared) < 2:
        raise ValueError(
            f"paired comparison needs at least two shared seeds, found {len(shared)}"
        )
    differences = np.array([arm[s] - baseline[s] for s in shared], dtype=np.double)
    result: Any = stats.ttest_rel(
        [arm[s] for s in shared], [baseline[s] for s in shared]
    )
    t_statistic: float = result.statistic
    p_value: float = result.pvalue
    standard_error: float = np.std(differences, ddof=1) / np.sqrt(len(differences))
    return Paired(
        n=len(shared),
        delta=float(np.mean(differences)),
        standard_error=standard_error,
        t_statistic=t_statistic,
        p_value=p_value,
    )


def _group(records: Sequence[RunRecord]) -> dict[str, list[RunRecord]]:
    keys = varying_keys([r.config for r in records])
    logger.info("Arms defined by: %s", ", ".join(keys) or "(nothing varies)")
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[arm_label(record.config, keys)].append(record)
    return dict(grouped)


def _paired_column(reference: ArmSummary, summary: ArmSummary) -> str:
    if summary.label == reference.label:
        return "(baseline)"
    try:
        paired = paired_delta(reference.per_seed, summary.per_seed)
    except ValueError as err:
        return str(err)
    return (
        f"{paired.delta:+.2f} ± {paired.standard_error:.2f}  "
        f"n={paired.n} t={paired.t_statistic:+.2f} p={paired.p_value:.3f}"
    )


def _arms_table(
    summaries: Sequence[ArmSummary], reference: ArmSummary, scored: str, pair_on: str
) -> Table:
    arms = Table(title=f"Arms — scored on {scored}, paired on {pair_on}")
    arms.add_column(header="Arm")
    arms.add_column(header="n", justify="right")
    arms.add_column(header="Mean", justify="right")
    # The spread is the reason the whole tool exists, so it is a column and not
    # a footnote: a difference smaller than this is not a result.
    arms.add_column(header="SD", justify="right")
    # The oracle sits at 80.1% of the MMD subsample and deconvolve at 73.3%, so ESS is
    # how a dispersion coefficient is calibrated rather than guessed.
    arms.add_column(header="ESS", justify="right")
    # Admissibility before performance: an arm the criterion can separate from
    # the best is one a truth-free pipeline would refuse to ship, whatever it
    # scores against truth.
    arms.add_column(header="Δ MMD (floors)", justify="right")
    arms.add_column(header="Paired vs baseline", justify="right")
    finite = [s.mmd_test for s in summaries if not np.isnan(s.mmd_test)]
    best = min(finite) if finite else float("nan")
    for summary in summaries:
        arms.add_row(
            summary.label,
            str(summary.n),
            f"{summary.mean:.2f}",
            f"{summary.sd:.2f}",
            "—" if np.isnan(summary.ess) else f"{summary.ess:.0f}",
            "—"
            if np.isnan(summary.mmd_test)
            else f"{(summary.mmd_test - best) / MMD_FLOOR:+.1f}"
            + (
                ""
                if tied_to_best(summary.mmd_test, best=best, floor=MMD_FLOOR)
                else " !"
            ),
            _paired_column(reference, summary),
        )
    return arms


def _observable_cell(summary: ArmSummary, name: str, paired: Paired | None) -> str:
    mean = summary.per_observable.get(name, float("nan"))
    if paired is None:
        return f"{mean:.1f}   —"
    return (
        f"{mean:.1f}   {paired.delta:+.1f} ± {paired.standard_error:.1f}"
        f"   {paired.p_value:.2f}"
    )


def _observable_table(
    summaries: Sequence[ArmSummary], reference: ArmSummary, names: Sequence[str]
) -> Table:
    """Observables as rows, not columns.

    Twelve of them do not fit across a terminal, and the comparison a reader
    makes is down an arm rather than across one.

    The aggregate is the scored response, but averaging N observables divides a
    move in one of them by N. Testing each separately is what stops a real
    single-observable effect from being reported as a null.
    """
    others = [s for s in summaries if s.label != reference.label]
    deltas = {s.label: observable_deltas(reference, s) for s in others}

    per_var = Table(title="Per observable (mean over seeds; excluded ones still shown)")
    per_var.add_column(header="Observable")
    per_var.add_column(header=f"{reference.label}\n(baseline)", justify="right")
    for summary in others:
        per_var.add_column(
            header=f"{summary.label}\nmean   Δ ± SE   p", justify="right"
        )
    for name in names:
        per_var.add_row(
            name,
            f"{reference.per_observable.get(name, float('nan')):.1f}",
            *(_observable_cell(s, name, deltas[s.label].get(name)) for s in others),
        )
    return per_var


def _report(
    summaries: Sequence[ArmSummary], baseline: str, scored: str, pair_on: str
) -> None:
    console = Console()
    reference = next(s for s in summaries if s.label == baseline)
    names = sorted(summaries[0].per_observable)

    console.print(_arms_table(summaries, reference, scored, pair_on))
    console.print(_observable_table(summaries, reference, names))
    logger.info(
        "Multiplicity: %d observables give %d chances at a small p-value, so one "
        "at p~0.05 is what a null looks like. Treat these as leads to retest, "
        "not as findings.",
        len(names),
        len(names),
    )

    logger.info(
        "Read a trend across levels, not an argmax: at n=%d against sd %.1f, "
        "the best of several arms is usually the luckiest rather than the best.",
        min(s.n for s in summaries),
        max(s.sd for s in summaries),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ = parser.add_argument("--arm-dir", type=Path, required=True)
    _ = parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="observable to drop from the aggregate but still report",
    )
    _ = parser.add_argument(
        "--baseline",
        default=None,
        help="arm label to compare against; default is the lexically first",
    )
    _ = parser.add_argument(
        "--pair-on",
        default=PAIRING_KEY,
        choices=sorted(SEED_KEYS),
        help="which nuisance axis the sweep replicated over",
    )
    args = parser.parse_args()
    configure_logging(level="info")

    arm_dir: Path = args.arm_dir
    exclude: list[str] = args.exclude
    pair_on: str = args.pair_on
    baseline_label: str | None = args.baseline

    records = load_records(arm_dir)
    logger.info("%s: %d runs", arm_dir.name, len(records))
    grouped = _group(records)
    summaries = [
        summarize_arm(label, grouped[label], exclude, pair_on)
        for label in sorted(grouped)
    ]
    baseline = baseline_label or summaries[0].label
    if all(s.label != baseline for s in summaries):
        raise SystemExit(
            f"no arm labelled {baseline!r}; found {[s.label for s in summaries]}"
        )
    scored = (
        f"mean of {len(summaries[0].per_observable) - len(exclude)} observables"
        + (f", excluding {', '.join(exclude)}" if exclude else "")
    )
    _report(summaries, baseline, scored, pair_on)


if __name__ == "__main__":
    main()
