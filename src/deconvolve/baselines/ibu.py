"""IBU (Iterative Bayesian Unfolding) baseline to compare with RAN.

It is a simple unfolding method that uses a Bayesian approach to unfold the
data. IBU performs 1D per-variable unfolding with purity-based automatic
binning. It builds the response matrix from MC, unfolds data, and converts the
result to per-event weights for evaluation with the same metrics as RAN.

```shell
deconvolve baseline ibu runs/2026-...
deconvolve baseline ibu runs  # all runs
```
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..coretypes import (
    DEFAULT_PURITY_THRESHOLD,
    JOINT_METRICS_KEY,
    IBUResult,
    VariableOutcome,
    artifacts_dir,
)
from ..evaluation import apply_to_runs, render_metrics, warn_if_no_joint
from ..training import EPS
from ._shared import (
    evaluate_dimension,
    evaluate_joint,
    load_populations,
    parse_run_config,
)

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any, Final

    from numpy._typing import _DTypeLikeFloat
    from numpy.typing import NDArray

    from ..coretypes import EventArray, MetricRecord, Populations, RunConfig

logger: Logger = logging.getLogger(name=__name__)


@dataclass(frozen=True, eq=False, slots=True)
class _BinnedReweighting:
    """A per-bin correction, learned from one population and applied to another.

    IBU produces one multiplicative factor per bin of the particle-level axis.
    Which events it is then applied to is a separate choice: here the
    unfolding is fit on train+val and applied to the held-out test split, so
    the sample it scores is genuinely not the sample it learned from.

    That is deliberately not what the unfolding literature usually does.
    Fitting the response and iterating the prior on every event, then quoting
    metrics on a subset of those same events, is conventional for both IBU and
    OmniFold -- and it scores an estimator on data it has already seen. It also
    hands the baseline information RAN is denied: `deconvolve.training.engine`
    reads the test split only to compute a diagnostic that cannot influence the
    returned model, and never to fit or select. A comparison is only a
    comparison if both sides see the same events.
    """

    edges: EventArray
    bin_weights: EventArray

    def weights_for(self, gen: EventArray) -> EventArray:
        """Per-event weights for particle-level values `gen`, mean one."""
        return _normalize_weights(self.bin_weights[_assign_bins(gen, self.edges)])


@dataclass(frozen=True, eq=False, slots=True)
class VariableUnfolding:
    """One variable's reweighting, or `None` where it could not be fit."""

    reweighting: _BinnedReweighting | None
    outcome: VariableOutcome

    def weights_for(self, gen: EventArray) -> EventArray:
        if self.reweighting is None:
            return np.ones(gen.shape[0], dtype=gen.dtype)
        return self.reweighting.weights_for(gen)


def _assign_bins(values: EventArray, edges: EventArray, /) -> NDArray[np.intp]:
    """Assign every value to a saturated bin.

    Underflow enters the first bin; overflow and values equal to the upper
    edge enter the last bin, so every finite input value receives an
    assignment.
    """
    if values.ndim != 1 or not np.all(a=np.isfinite(values)):
        raise ValueError("bin values must be a finite one-dimensional array")
    if edges.ndim != 1 or edges.size < 2 or not np.all(a=np.diff(a=edges) > 0):
        raise ValueError("bin edges must be a strictly increasing 1D array")
    n_bins: int = edges.size - 1
    return (
        np.clip(a=np.digitize(x=values, bins=edges), a_min=1, a_max=n_bins) - 1
    ).astype(dtype=np.intp, copy=False)


def _bin_counts(indices: NDArray[np.intp], n_bins: int, /) -> NDArray[np.intp]:
    """Count saturated assignments while preserving every assigned event.

    Assignments from `_assign_bins` place underflow in the first bin and both
    overflow and the upper edge in the last bin; their counts therefore retain
    one entry for every assigned event.
    """
    if n_bins < 1 or indices.ndim != 1:
        raise ValueError("bin indices must be one-dimensional with n_bins >= 1")
    if np.any(a=(indices < 0) | (indices >= n_bins)):
        raise ValueError("bin index outside configured range")
    return np.bincount(indices, minlength=n_bins)


def _unfolded_to_bin_weights(unfolded: EventArray, prior: EventArray) -> EventArray:
    if (
        unfolded.ndim != 1
        or prior.ndim != 1
        or unfolded.size == 0
        or unfolded.shape != prior.shape
    ):
        raise ValueError(
            "unfolded and prior must have matching nonempty one-dimensional shapes"
        )
    if not np.all(a=np.isfinite(unfolded)) or not np.all(a=np.isfinite(prior)):
        raise ValueError("unfolded and prior must be finite")
    if np.any(a=unfolded < 0) or np.any(a=prior < 0):
        raise ValueError("unfolded and prior must be nonnegative")
    zero_prior_mass: NDArray[np.bool] = (prior == 0) & (unfolded > EPS)
    if np.any(a=zero_prior_mass):
        raise ValueError("unfolded mass in a zero-prior bin")

    weights: EventArray = np.zeros_like(a=unfolded)
    _ = np.divide(unfolded, prior, out=weights, where=prior > 0)
    return weights


def _normalize_weights(weights: EventArray) -> EventArray:
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty one-dimensional vector")
    if not np.all(a=np.isfinite(weights)):
        raise ValueError("weights must be finite")
    if np.any(a=weights < 0):
        raise ValueError("weights must be nonnegative")

    mean: np.single = np.mean(a=weights)
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("weights mean must be finite and strictly positive")

    normalized: EventArray = np.divide(weights, mean)
    if not np.all(a=np.isfinite(normalized)) or np.any(a=normalized < 0):
        raise ValueError("normalized weights must be finite and nonnegative")
    # The division runs in the caller's precision; only the check that it
    # worked is accumulated wider, so a float32 resummation cannot fail it.
    if not np.isclose(a=normalized.mean(dtype=np.double), b=1.0):
        raise ValueError("normalized weights must have mean one")
    return normalized


def _next_pure_edge(
    gen_sorted: EventArray,
    upper_sorted: EventArray,
    lower_by_upper: EventArray,
    lo: np.single,
    gen_max: np.single,
    purity_threshold: float,
    n_candidates: int = 100,
) -> np.single | None:
    """Find the first candidate edge whose bin exceeds the purity threshold.

    Returns:
        The edge, or `None` if no candidate between `lo` and `gen_max` gives
        a bin pure enough.
    """
    if n_candidates <= 0:
        raise ValueError("n_candidates must be positive")

    candidates: EventArray = np.linspace(
        start=lo + 1 / n_candidates,
        stop=gen_max,
        num=n_candidates,
        dtype=gen_sorted.dtype,
    )

    # Denominator = count(lo <= gen < candidate) for every candidate.
    truth_start: np.intp = np.searchsorted(a=gen_sorted, v=lo, side="left")
    truth_stop: NDArray[np.intp] = np.searchsorted(
        a=gen_sorted, v=candidates, side="left"
    )
    n_truth: NDArray[np.intp] = truth_stop - truth_start

    # Because upper_sorted is ordered, the first k elements are precisely
    # those for which max(gen, reco) < candidates[j].
    upper_stop: NDArray[np.intp] = np.searchsorted(
        a=upper_sorted, v=candidates, side="left"
    )

    # Among those elements, count the ones satisfying min(gen, reco) >= lo.
    prefix: NDArray[np.ulong] = np.empty(shape=lower_by_upper.size + 1, dtype=np.ulong)
    prefix[0] = 0
    _ = np.cumsum(
        a=lower_by_upper >= lo,
        dtype=np.ulong,
        out=prefix[1:],
    )
    n_both: NDArray[np.ulong] = prefix[upper_stop]

    purity: NDArray[np.double] = np.zeros(shape=n_candidates, dtype=np.double)
    _ = np.divide(
        n_both,
        n_truth,
        out=purity,
        where=n_truth != 0,
    )

    resolved: NDArray[np.bool] = (n_truth != 0) & (purity > purity_threshold)
    qualifying: NDArray[np.intp] = np.flatnonzero(a=resolved)
    if qualifying.size == 0:
        return None

    return np.single(candidates[qualifying[0]])


def _purity_bins(
    gen: EventArray,
    sim: EventArray,
    purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
    max_bins: int = 50,
) -> EventArray:
    """Determine bin edges where purity exceeds the threshold."""
    if gen.ndim != 1 or sim.ndim != 1:
        raise ValueError("gen and sim must be one-dimensional")
    if gen.shape != sim.shape:
        raise ValueError("gen and sim must have the same shape")
    if gen.size == 0:
        raise ValueError("gen and sim must not be empty")
    if max_bins <= 0:
        raise ValueError("max_bins must be positive")

    gen_sorted: EventArray = np.sort(a=gen)

    lower: EventArray = np.minimum(gen, sim)
    upper: EventArray = np.maximum(gen, sim)

    upper_order: NDArray[np.intp] = np.argsort(a=upper)
    upper_sorted: EventArray = upper[upper_order]
    lower_by_upper: EventArray = lower[upper_order]

    # max_bins bins require at most max_bins + 1 edges.
    edges: EventArray = np.empty(shape=max_bins + 1, dtype=gen.dtype)
    edges[0] = gen.min()
    n_edges = 1

    gen_max: Final[np.single] = gen.max()
    while n_edges <= max_bins and edges[n_edges - 1] < gen_max:
        edge: np.single | None = _next_pure_edge(
            gen_sorted=gen_sorted,
            upper_sorted=upper_sorted,
            lower_by_upper=lower_by_upper,
            lo=edges[n_edges - 1],
            gen_max=gen_max,
            purity_threshold=purity_threshold,
        )
        if edge is None:
            break

        edges[n_edges] = edge
        n_edges += 1

    return edges[:n_edges]


def _build_response(
    gen_bins: NDArray[np.intp],
    sim_bins: NDArray[np.intp],
    n_bins: int,
    dtype: _DTypeLikeFloat,
    /,
) -> EventArray:
    """Build row-normalized response matrix R[t,r] = P(reco=r | truth=t)."""
    response: EventArray = np.zeros(shape=(n_bins, n_bins), dtype=dtype)
    np.add.at(response, (gen_bins, sim_bins), 1)
    row_sums: EventArray = response.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    response /= row_sums
    return response


def _ibu(
    prior: EventArray,
    data_hist: EventArray,
    response: EventArray,
    n_iterations: int,
    strict: bool = False,
) -> EventArray:
    """Iterative Bayesian Unfolding.

    Args:
        prior: Initial truth estimate (MC gen histogram), shape `(n_bins,)`.
        data_hist: Observed reco-level measured histogram, shape `(n_bins,)`.
        response: `R[t, r] = P(sim=r | gen=t)`, shape `(n_bins, n_bins)`.
        n_iterations: Number of unfolding iterations.
        strict: If True, raise an error if the observed data has zero support
            under the response and prior. If False, return zero weights for
            such events.

    Returns:
        Unfolded truth histogram, shape `(n_bins,)`.
    """
    posterior: EventArray = prior.copy()

    for _ in range(n_iterations):
        marginal: EventArray = response.T @ posterior
        unsupported: NDArray[np.bool] = (marginal == 0) & (data_hist != 0)
        if strict and np.any(a=unsupported):
            raise ValueError(
                "Observed data has zero support under the response and prior"
            )
        # `out=` makes the value of the skipped entries zero.
        likelihood: EventArray = np.zeros_like(a=posterior)
        _ = np.divide(
            data_hist,
            marginal,
            out=likelihood,
            where=marginal != 0,
        )
        posterior *= response @ likelihood
    return posterior


def unfold_variable(
    variable_name: str,
    mc_gen: EventArray,
    mc_sim: EventArray,
    observed: EventArray,
    n_iterations: int,
    purity_threshold: float,
) -> VariableUnfolding:
    """Fit one variable's reweighting.

    Takes one column each of a `Populations`' `mc.z`, `mc.x` and `data`; those
    three are what a real measurement has, and `truth` is deliberately not
    among them. Where purity binning yields fewer than two bins there is
    nothing to fit, so the reweighting is `None` and `weights_for` returns
    ones.

    Args:
        variable_name: The variable being unfolded, for logging and the
            outcome record.
        mc_gen: One column of `mc.z`, the generated particle level.
        mc_sim: One column of `mc.x`, row-aligned with `mc_gen`; together they
            give the response.
        observed: The same column of `data`, the measurement. No part of
            `truth` belongs here.
        n_iterations: Number of unfolding iterations.
        purity_threshold: The purity threshold for automatic binning.

    Returns:
        A `VariableUnfolding`, which pairs the reweighting with a
        `VariableOutcome` recording whether the fit happened. Its
        `weights_for(gen)` gives per-event weights for whichever sample is
        being scored.
    """
    dtype: np.dtype[np.single] = mc_gen.dtype
    bins: EventArray = _purity_bins(mc_gen, mc_sim, purity_threshold)
    n_bins: int = bins.size - 1
    if n_bins < 2:
        logger.warning("%s: only %d bin(s), skipping", variable_name, n_bins)
        return VariableUnfolding(
            reweighting=None,
            outcome=VariableOutcome(
                variable_name,
                "skipped",
                n_bins,
                skip_reason="fewer than two purity bins",
            ),
        )

    mc_gen_bins: NDArray[np.intp] = _assign_bins(mc_gen, bins)
    mc_sim_bins: NDArray[np.intp] = _assign_bins(mc_sim, bins)
    observed_bins: NDArray[np.intp] = _assign_bins(observed, bins)

    response: EventArray = _build_response(mc_gen_bins, mc_sim_bins, n_bins, dtype)
    prior: EventArray = _bin_counts(mc_gen_bins, n_bins).astype(dtype)
    data_hist: EventArray = _bin_counts(observed_bins, n_bins).astype(dtype)
    # Accumulated in float64 whatever the unfolding runs in: these are exact
    # integer counts, and float32 stops representing those past 2**24, which
    # would fail the comparison on sample size alone.
    prior_count: np.double = prior.sum(dtype=np.double)
    if prior_count != mc_gen.size:
        raise ValueError(
            "prior/response population count mismatch: "
            f"actual={prior_count}, expected={mc_gen.size}"
        )
    observed_count: np.double = data_hist.sum(dtype=np.double)
    if observed_count != observed.size:
        raise ValueError(
            "observed/data population count mismatch: "
            f"actual={observed_count}, expected={observed.size}"
        )

    unfolded: EventArray = _ibu(prior, data_hist, response, n_iterations)
    logger.info("%s: %d bins, %d iterations", variable_name, n_bins, n_iterations)
    return VariableUnfolding(
        reweighting=_BinnedReweighting(
            edges=bins,
            bin_weights=_unfolded_to_bin_weights(unfolded, prior),
        ),
        outcome=VariableOutcome(variable_name, "completed", n_bins),
    )


def joint_weights(weights: NDArray[np.single]) -> NDArray[np.single]:
    """One weight per event from IBU's `(dim, n)` per-variable weights.

    IBU unfolds each variable on its own, so it never produces a joint
    reweighting; the product of its 1D weights, renormalized, is the joint it
    implicitly assumes -- the observables reweighted as if independent. That is
    exactly the assumption the sliced Wasserstein distance exists to test, so
    it is what IBU's joint score is computed on.

    A variable IBU refused to unfold carries weight one everywhere and drops
    out of the product. The product is taken in float64: twelve factors can
    leave float32's comfortable range before the normalization brings them
    back.
    """
    product: NDArray[np.double] = np.prod(a=weights, axis=0, dtype=np.double)
    return _normalize_weights((product / product.mean()).astype(np.single))


def _run_and_evaluate(
    config: RunConfig,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> IBUResult:
    """Run 1D IBU per variable and evaluate on test set."""
    if type(n_iterations) is not int or n_iterations <= 0:
        raise ValueError("n_iterations must be a positive integer")
    if not np.isfinite(purity_threshold) or not 0 <= purity_threshold <= 1:
        raise ValueError("purity_threshold must be finite and between zero and one")

    fit: Populations
    test: Populations
    fit, test = load_populations(config)
    test_truth: NDArray[np.single] = test.require_truth()
    weights: NDArray[np.single] = np.empty(
        shape=(config.dim, len(test.mc)), dtype=np.single
    )
    detector: dict[str, MetricRecord] = {}
    particle: dict[str, MetricRecord] = {}
    outcomes: list[VariableOutcome] = []

    for dimension, variable_name in enumerate(iterable=config.variable_names):
        unfolding: VariableUnfolding = unfold_variable(
            variable_name=variable_name,
            mc_gen=fit.mc.z[:, dimension],
            mc_sim=fit.mc.x[:, dimension],
            observed=fit.data[:, dimension],
            n_iterations=n_iterations,
            purity_threshold=purity_threshold,
        )
        test_weights: NDArray[np.single] = unfolding.weights_for(
            gen=test.mc.z[:, dimension]
        )
        weights[dimension] = test_weights
        outcomes.append(unfolding.outcome)
        detector[f"detector_{variable_name}"] = evaluate_dimension(
            reference=test.data[:, dimension],
            comparison=test.mc.x[:, dimension],
            weights=test_weights,
        )
        particle[f"particle_{variable_name}"] = evaluate_dimension(
            reference=test_truth[:, dimension],
            comparison=test.mc.z[:, dimension],
            weights=test_weights,
        )

    # Every detector entry, then every particle entry, matching the key order
    # `evaluation.evaluate.evaluate_run` writes to metrics.json.
    metrics: dict[str, MetricRecord] = detector | particle

    return IBUResult(
        metrics=metrics,
        variable_names=config.variable_names,
        weights=weights,
        outcomes=tuple(outcomes),
        joint=evaluate_joint(test, joint_weights(weights)),
    )


def evaluate_single(
    run_dir: Path,
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> dict[str, Any]:
    """Run IBU on a single run's dataset and save comparison metrics.

    It fits on train+val, then scores the held-out test split with the result,
    so the sample scored is genuinely not the sample fitted.

    The cache hit requires both `metrics_ibu.json` and `ibu_outcomes.json` to
    exist -- a directory holding only the former is an incomplete result (an
    older run, or one interrupted between the two writes), and the outcomes
    file is needed to mark variables IBU refused to unfold. Missing either
    file is treated as a cache miss and recomputes both.
    """
    out_path: Path = artifacts_dir(run_dir) / "metrics_ibu.json"
    outcomes_path: Path = artifacts_dir(run_dir) / "ibu_outcomes.json"

    if out_path.exists() and outcomes_path.exists() and not force:
        logger.info("%s: metrics_ibu.json exists, skipping (use --force)", run_dir.name)
        existing: dict[str, Any] = json.loads(s=out_path.read_text())
        warn_if_no_joint(run_dir.name, out_path.name, existing)
        return existing

    raw_config: object = json.loads(s=(run_dir / "config.json").read_text())
    config: RunConfig = parse_run_config(raw_config)
    if out_path.exists() and not outcomes_path.exists():
        logger.info(
            "%s: metrics_ibu.json exists but ibu_outcomes.json is missing, "
            "recomputing both",
            run_dir.name,
        )
    logger.info(
        "%s: running IBU (niter=%d, purity=%.4f)...",
        run_dir.name,
        n_iterations,
        purity_threshold,
    )
    result: IBUResult = _run_and_evaluate(
        config,
        n_iterations=n_iterations,
        purity_threshold=purity_threshold,
    )

    metrics: dict[str, Any] = {**result.metrics, JOINT_METRICS_KEY: result.joint}
    json.dump(obj=metrics, fp=out_path.open(mode="w"), indent=2)

    # `outcomes` records the variables IBU's purity binning gave up on and
    # returned unchanged. Without it, a report showing `IBU == Sim` and a 0.0%
    # improvement reads as a measurement rather than a refusal.
    _ = (artifacts_dir(run_dir) / "ibu_outcomes.json").write_text(
        data=json.dumps(obj=[asdict(obj=o) for o in result.outcomes], indent=2)
    )

    weights_path: Path = artifacts_dir(run_dir) / "ibu_weights.npz"
    np.savez(
        file=weights_path,
        **{
            f"weights_{i}": weights for i, weights in enumerate(iterable=result.weights)
        },  # ty: ignore[invalid-argument-type]
    )
    logger.info(
        "%s: saved IBU metrics to %s and weights to %s",
        run_dir.name,
        out_path,
        weights_path,
    )
    render_metrics(f"{run_dir.name} [IBU]", metrics, list(result.variable_names))
    # The figures were drawn when training finished, before this baseline
    # existed; nothing redraws them on its own.
    logger.info(
        "%s: run `deconvolve plot %s` to add the %s overlay to the figures",
        run_dir.name,
        run_dir,
        "IBU",
    )
    return metrics


def evaluate_runs(
    run_dir: Path = Path("runs"),
    force: bool = False,
    n_iterations: int = 10,
    purity_threshold: np.double = DEFAULT_PURITY_THRESHOLD,
) -> None:
    """Run the IBU baseline on completed RAN runs.

    Args:
        run_dir: A single run, or a directory of runs.
        force: Recompute even if `metrics_ibu.json` exists.
        n_iterations: Number of IBU iterations.
        purity_threshold: Purity threshold for automatic binning.
    """
    apply_to_runs(
        run_dir,
        evaluate_one=lambda run_dir: evaluate_single(
            run_dir,
            force,
            n_iterations,
            purity_threshold,
        ),
        description="evaluate with IBU",
        log=logger,
    )
