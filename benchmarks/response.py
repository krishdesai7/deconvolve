"""Test whether Herwig and Pythia induce the same response on Deconvolve's summaries.

The generator label is S (Herwig=1, Pythia=0).  Two classifiers are fitted:

    D_z(z)       = P(S=1 | z)
    D_zx(z, x)   = P(S=1 | z, x)

If the induced response p(x | z) is generator-independent, x contains no
additional generator information once z is known.  The reported statistic is

    delta = test_BCE(D_z) - test_BCE(D_zx),

which is conditional mutual information I(S; X | Z) for Bayes-optimal,
calibrated classifiers.  Positive delta is evidence of response mismatch.

The optional null closure partitions the paired events from one generator
into two pseudo-domains using labels sampled from a z-dependent propensity.
Their z distributions differ, but their response is identical by construction.
These smaller within-generator fits diagnose finite-sample classifier bias; they
are not a calibrated significance test for the larger observed comparison.

`--z-var` and `--x-var` set the two column sets independently, which is what
lets the conditioning set differ from the tested one.  If the observable-level
response is non-universal only because it marginalizes over the rest of the
event, then conditioning on more of the particle-level event should shrink the
statistic:

    I(S; X_m | Z_m)        --x-var m --z-var m
    I(S; X_m | Z_1..6)     --x-var m
    I(S; X_1..6 | Z_1..6)  (no flags)

A drop from the first to the second says the other five observables carry the
missing information.  Flat says they do not, and the answer is not in these six.

Both sides come from a single `load_jet_dataset` call over the union of the two
sets, because standardization is fitted per sample: loading each side
separately would put the two halves of a pair in different units.

Run from the repository root:

    uv run benchmarks/response.py --n-samples 200000
    uv run benchmarks/response.py --n-samples 200000 --null-repeats 5
    uv run benchmarks/response.py --n-samples 200000 --x-var m
"""

# pyrefly: ignore-errors[unknown-argument-type]
# -- argparse Namespace and numpy elementwise/boolean-index results are Any
from __future__ import annotations

import os

os.environ.setdefault(key="KERAS_BACKEND", value="jax")  # pyrefly: ignore[unused-call-result]

import argparse
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import keras
import numpy as np
from deconvolve.coretypes import SUBSTRUCTURE_VARIABLES, Split
from deconvolve.data import load_jet_dataset
from deconvolve.instrumentation.logging_config import configure_logging
from scipy.special import expit

try:
    from benchmarks.ceiling import _fit_classifier, _labelled
except ModuleNotFoundError:  # Direct execution puts benchmarks/ on sys.path.
    from ceiling import _fit_classifier, _labelled  # pyrefly: ignore[missing-import]

if TYPE_CHECKING:
    from deconvolve.coretypes import (
        DatasetSplits,
        DeconvolveModel,
        EventArray,
        Populations,
    )
    from numpy.typing import NDArray


LOG2: float = math.log(2.0)
_P_CLIP: float = 1e-7
logger: logging.Logger = logging.getLogger(name="deconvolve.response")


@dataclass(frozen=True)
class Pairs:
    """Row-aligned particle- and detector-level events from one generator."""

    z: NDArray[np.floating]
    x: NDArray[np.floating]

    def __post_init__(self) -> None:
        if self.z.ndim != 2 or self.x.ndim != 2:
            raise ValueError("z and x must both have shape (events, features)")
        if len(self.z) != len(self.x):
            raise ValueError("z and x must be row-aligned")


@dataclass(frozen=True)
class Domains:
    """Positive and negative domain samples for a binary classifier."""

    positive: Pairs
    negative: Pairs


@dataclass(frozen=True)
class ResponseStatistic:
    """Held-out response-mismatch score and its test-event sampling error."""

    bce_z: float
    bce_zx: float
    delta_nats: float
    delta_bits: float
    standard_error: float


@dataclass(frozen=True)
class PseudoDomainRule:
    """A fixed z-dependent propensity used to construct exact-null domains."""

    projection: NDArray[np.floating]
    center: float
    scale: float
    strength: float

    @classmethod
    def fit(
        cls, z: NDArray[np.floating], /, *, seed: int, strength: float
    ) -> PseudoDomainRule:
        """Choose a random z direction and fix its scale from training events."""
        if z.ndim != 2 or len(z) < 2:
            raise ValueError("z must contain at least two row-wise events")
        if strength <= 0:
            raise ValueError("strength must be positive")
        projection: NDArray[np.floating] = np.random.default_rng(seed).normal(
            size=z.shape[1]
        )
        pivot = int(np.argmax(np.abs(projection)))
        if projection[pivot] < 0:
            projection = -projection
        score: NDArray[np.double] = np.asarray(a=z @ projection, dtype=np.double)
        scale = float(score.std())
        if not np.isfinite(scale) or scale == 0:
            raise ValueError("pseudo-domain projection has zero variance")
        return cls(
            projection=projection,
            center=float(score.mean()),
            scale=scale,
            strength=strength,
        )

    def partition(self, pairs: Pairs, /, *, seed: int) -> Domains:
        """Sample pseudo-labels from P(S=1|z) without looking at x."""
        score: NDArray[np.double] = (
            pairs.z @ self.projection - self.center
        ) / self.scale
        probability: NDArray[np.double] = expit(self.strength * score)
        positive: NDArray[np.bool_] = (
            np.random.default_rng(seed).random(len(pairs.z)) < probability
        )
        if positive.all() or not positive.any():
            raise ValueError("pseudo-domain assignment produced an empty class")
        return Domains(
            positive=Pairs(z=pairs.z[positive], x=pairs.x[positive]),
            negative=Pairs(z=pairs.z[~positive], x=pairs.x[~positive]),
        )


def response_statistic(
    labels: NDArray[np.floating],
    p_z: NDArray[np.floating],
    p_zx: NDArray[np.floating],
) -> ResponseStatistic:
    """Calculate the paired held-out BCE gain from adding detector information."""
    labels = np.asarray(labels, dtype=np.double).ravel()
    p_z = np.asarray(p_z, dtype=np.double).ravel()
    p_zx = np.asarray(p_zx, dtype=np.double).ravel()
    if not (len(labels) == len(p_z) == len(p_zx)) or len(labels) < 2:
        raise ValueError("labels and predictions must have the same length >= 2")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    if min(np.count_nonzero(labels == label) for label in (0.0, 1.0)) < 2:
        raise ValueError("labels must contain at least two events from each class")
    if any(
        not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
        for probability in (p_z, p_zx)
    ):
        raise ValueError("probabilities must be finite and lie in [0, 1]")

    def losses(probability: NDArray[np.floating]) -> NDArray[np.floating]:
        probability = np.clip(a=probability, a_min=_P_CLIP, a_max=1.0 - _P_CLIP)
        return cast(
            typ="NDArray[np.floating]",
            val=-(
                labels * np.log(probability) + (1.0 - labels) * np.log1p(-probability)
            ),
        )

    loss_z: NDArray[np.floating] = losses(p_z)
    loss_zx: NDArray[np.floating] = losses(p_zx)
    improvement: NDArray[np.floating] = loss_z - loss_zx
    delta = float(improvement.mean())
    by_class: tuple[NDArray[np.floating], NDArray[np.floating]] = tuple(
        improvement[labels == label] for label in (0.0, 1.0)
    )
    standard_error: float = 0.5 * math.sqrt(
        sum(float(values.var(ddof=1)) / len(values) for values in by_class)
    )
    return ResponseStatistic(
        bce_z=float(loss_z.mean()),
        bce_zx=float(loss_zx.mean()),
        delta_nats=delta,
        delta_bits=delta / LOG2,
        standard_error=standard_error,
    )


def _pairs(
    pop: Populations,
    /,
    *,
    nature: bool,
    z_cols: NDArray[np.intp],
    x_cols: NDArray[np.intp],
) -> Pairs:
    """One generator's paired events, restricted to the two column sets.

    The sets are allowed to differ, which is the whole point: `I(S; X | Z)`
    with `X` a single observable and `Z` the full vector asks whether
    conditioning on the rest of the particle-level event constrains the
    latent `u` that the observable-level response marginalizes over.
    """
    if nature:
        return Pairs(z=pop.require_truth()[:, z_cols], x=pop.data[:, x_cols])
    return Pairs(z=pop.mc.z[:, z_cols], x=pop.mc.x[:, x_cols])


def _balanced(domains: Domains, /) -> Domains:
    """Use an exact 50:50 class prior without sample-weight conventions."""
    n = min(len(domains.positive.z), len(domains.negative.z))
    if n < 2:
        raise ValueError("each domain must contain at least two events")
    return Domains(
        positive=Pairs(domains.positive.z[:n], domains.positive.x[:n]),
        negative=Pairs(domains.negative.z[:n], domains.negative.x[:n]),
    )


def _inputs(pairs: Pairs, /, *, include_x: bool) -> NDArray[np.floating]:
    return np.concatenate((pairs.z, pairs.x), axis=1) if include_x else pairs.z


def _fit(
    train: Domains,
    val: Domains,
    /,
    *,
    include_x: bool,
    label: str,
    fit_kwargs: dict[str, Any],
) -> DeconvolveModel:
    train = _balanced(train)
    val = _balanced(val)
    return _fit_classifier(
        _inputs(train.positive, include_x=include_x),
        _inputs(train.negative, include_x=include_x),
        _inputs(val.positive, include_x=include_x),
        _inputs(val.negative, include_x=include_x),
        label=label,
        **fit_kwargs,
    ).model


def _probability(model: DeconvolveModel, inputs: NDArray[np.floating]) -> EventArray:
    # `DeconvolveModel` is a narrow protocol; the classifier built here is a full
    # `keras.Model` and `.predict` is what the batched forward pass needs.
    predicted = cast("keras.Model", model).predict(inputs, batch_size=8192, verbose=0)
    return cast("EventArray", np.asarray(predicted).ravel().astype(np.double))


def _measure(
    train: Domains,
    val: Domains,
    test: Domains,
    /,
    *,
    seed: int,
    label: str,
    fit_kwargs: dict[str, Any],
) -> ResponseStatistic:
    """Fit both nested predictors and score their BCE gain on the same events."""
    keras.utils.set_random_seed(seed)
    d_z = _fit(
        train,
        val,
        include_x=False,
        label=f"{label} D_z(z)",
        fit_kwargs=fit_kwargs,
    )
    d_zx = _fit(
        train,
        val,
        include_x=True,
        label=f"{label} D_zx(z,x)",
        fit_kwargs=fit_kwargs,
    )

    test = _balanced(test)
    z, labels = _labelled(test.positive.z, test.negative.z)
    zx, labels_zx = _labelled(
        _inputs(test.positive, include_x=True),
        _inputs(test.negative, include_x=True),
    )
    if not np.array_equal(labels, labels_zx):
        raise RuntimeError("internal error: test labels are not row-aligned")
    return response_statistic(labels, _probability(d_z, z), _probability(d_zx, zx))


def _report(statistic: ResponseStatistic, /, *, prefix: str = "") -> None:
    logger.info("%sBCE D_z       %.6f", prefix, statistic.bce_z)
    logger.info("%sBCE D_zx      %.6f", prefix, statistic.bce_zx)
    logger.info(
        "%sdelta         %+.6f nats/event  %+.6f bits/event",
        prefix,
        statistic.delta_nats,
        statistic.delta_bits,
    )
    logger.info("%stest-event SE %.6f nats/event", prefix, statistic.standard_error)


def _null_statistics(
    train_sources: tuple[tuple[str, Pairs], ...],
    val_sources: tuple[Pairs, ...],
    test_sources: tuple[Pairs, ...],
    /,
    *,
    repeats: int,
    strength: float,
    seed: int,
    fit_kwargs: dict[str, Any],
) -> dict[str, list[float]]:
    null: dict[str, list[float]] = {name: [] for name, _pairs in train_sources}
    for source_index, ((name, train), val, test) in enumerate(
        zip(train_sources, val_sources, test_sources, strict=True)
    ):
        for repeat in range(repeats):
            rule_seed = seed + 10_000 * source_index + repeat
            rule = PseudoDomainRule.fit(train.z, seed=rule_seed, strength=strength)
            statistic = _measure(
                rule.partition(train, seed=rule_seed + 1),
                rule.partition(val, seed=rule_seed + 2),
                rule.partition(test, seed=rule_seed + 3),
                seed=rule_seed + 4,
                label=f"null {name} {repeat + 1}/{repeats}",
                fit_kwargs=fit_kwargs,
            )
            null[name].append(statistic.delta_nats)
            logger.info(
                "null %-6s %d/%d: delta %+.6f nats/event",
                name,
                repeat + 1,
                repeats,
                statistic.delta_nats,
            )
    return null


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    _ = parser.add_argument("--n-samples", type=int, default=200_000)
    _ = parser.add_argument(
        "--var",
        action="append",
        dest="variables",
        choices=SUBSTRUCTURE_VARIABLES,
        help="default column set for both sides (all six if omitted)",
    )
    _ = parser.add_argument(
        "--z-var",
        action="append",
        dest="z_variables",
        choices=SUBSTRUCTURE_VARIABLES,
        help="particle-level columns to condition on; defaults to --var",
    )
    _ = parser.add_argument(
        "--x-var",
        action="append",
        dest="x_variables",
        choices=SUBSTRUCTURE_VARIABLES,
        help="detector-level columns to test; defaults to --var",
    )
    _ = parser.add_argument("--hidden-units", type=int, default=128)
    _ = parser.add_argument("--n-layers", type=int, default=3)
    _ = parser.add_argument("--epochs", type=int, default=200)
    _ = parser.add_argument("--patience", type=int, default=10)
    _ = parser.add_argument("--batch-size", type=int, default=1024)
    _ = parser.add_argument("--lr", type=float, default=1e-3)
    _ = parser.add_argument("--data-seed", type=int, default=42)
    _ = parser.add_argument("--seed", type=int, default=0)
    _ = parser.add_argument(
        "--null-repeats",
        type=int,
        default=0,
        help="pseudo-domain repeats per generator (0 disables calibration)",
    )
    _ = parser.add_argument(
        "--null-strength",
        type=float,
        default=1.5,
        help="strength of z-dependent covariate shift in pseudo-domains",
    )
    args = parser.parse_args()
    if args.null_repeats < 0:
        parser.error("--null-repeats must be non-negative")
    if args.null_repeats and (
        not math.isfinite(args.null_strength) or args.null_strength <= 0
    ):
        parser.error("--null-strength must be positive and finite")
    return args


class Columns(NamedTuple):
    """The two column sets and their indices into one shared load."""

    z_variables: tuple[str, ...]
    x_variables: tuple[str, ...]
    variables: tuple[str, ...]
    z_cols: NDArray[np.intp]
    x_cols: NDArray[np.intp]


def _resolve_columns(args: argparse.Namespace, /) -> Columns:
    """Work out which columns each side sees, from --var/--z-var/--x-var.

    One load of the union, then two column views of it. Loading each side
    separately would standardize each against its own sample, and the two
    halves of a pair have to live in the same units.
    """

    def canonical(names: list[str] | None) -> tuple[str, ...]:
        """Canonical order, as `cli._canonical_variables` fixes it.

        These names index columns, so a permuted list is a different dataset.
        """
        chosen = set(names or args.variables or SUBSTRUCTURE_VARIABLES)
        return tuple(v for v in SUBSTRUCTURE_VARIABLES if v in chosen)

    z_variables = canonical(args.z_variables)
    x_variables = canonical(args.x_variables)
    variables = canonical(list(set(z_variables) | set(x_variables)))
    return Columns(
        z_variables=z_variables,
        x_variables=x_variables,
        variables=variables,
        z_cols=np.array([variables.index(v) for v in z_variables], dtype=np.intp),
        x_cols=np.array([variables.index(v) for v in x_variables], dtype=np.intp),
    )


def _domains(pop: Populations, cols: Columns, /) -> Domains:
    """Herwig as the positive domain, Pythia as the negative one."""
    return Domains(
        _pairs(pop, nature=True, z_cols=cols.z_cols, x_cols=cols.x_cols),
        _pairs(pop, nature=False, z_cols=cols.z_cols, x_cols=cols.x_cols),
    )


def main() -> None:
    args = _parse_args()
    configure_logging(level="info")
    cols = _resolve_columns(args)
    z_variables, x_variables = cols.z_variables, cols.x_variables
    variables, z_cols, x_cols = cols.variables, cols.z_cols, cols.x_cols

    splits: DatasetSplits
    splits, _dim, _std = load_jet_dataset(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        variables=variables,
        seed=args.data_seed,
    )
    train_pop = splits.select(Split.TRAIN).partition()
    val_pop = splits.select(Split.VAL).partition()
    test_pop = splits.select(Split.TEST).partition()

    fit_kwargs = {
        "hidden_units": args.hidden_units,
        "n_layers": args.n_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
    }
    observed = _measure(
        _domains(train_pop, cols),
        _domains(val_pop, cols),
        _domains(test_pop, cols),
        seed=args.seed,
        label="observed",
        fit_kwargs=fit_kwargs,
    )

    logger.info("")
    logger.info(
        "RESPONSE MISMATCH: Herwig versus Pythia -- I(S; X | Z)",
    )
    logger.info(
        "  X = %s   (detector level, %d column%s)",
        ", ".join(x_variables),
        len(x_variables),
        "" if len(x_variables) == 1 else "s",
    )
    logger.info(
        "  Z = %s   (particle level, %d column%s)",
        ", ".join(z_variables),
        len(z_variables),
        "" if len(z_variables) == 1 else "s",
    )
    _report(observed, prefix="  ")

    if args.null_repeats:
        null_by_source = _null_statistics(
            (
                (
                    "Herwig",
                    _pairs(train_pop, nature=True, z_cols=z_cols, x_cols=x_cols),
                ),
                (
                    "Pythia",
                    _pairs(train_pop, nature=False, z_cols=z_cols, x_cols=x_cols),
                ),
            ),
            (
                _pairs(val_pop, nature=True, z_cols=z_cols, x_cols=x_cols),
                _pairs(val_pop, nature=False, z_cols=z_cols, x_cols=x_cols),
            ),
            (
                _pairs(test_pop, nature=True, z_cols=z_cols, x_cols=x_cols),
                _pairs(test_pop, nature=False, z_cols=z_cols, x_cols=x_cols),
            ),
            repeats=args.null_repeats,
            strength=args.null_strength,
            seed=args.seed + 1_000_000,
            fit_kwargs=fit_kwargs,
        )
        logger.info("")
        logger.info("WITHIN-GENERATOR EXACT-NULL CLOSURES")
        logger.info(
            "  These use half the observed per-class sample size; compare as a"
            " classifier-bias diagnostic, not as a significance calibration."
        )
        for name, values in null_by_source.items():
            null = np.asarray(values, dtype=np.double)
            null_sd = float(null.std(ddof=1)) if len(null) > 1 else float("nan")
            logger.info(
                "  %-7s mean %+.6f  SD %.6f nats/event  (%d statistics, %d fits)",
                name,
                float(null.mean()),
                null_sd,
                len(null),
                2 * len(null),
            )


if __name__ == "__main__":
    main()
