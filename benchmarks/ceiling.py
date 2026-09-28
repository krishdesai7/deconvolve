from __future__ import annotations

import json
import logging
from itertools import starmap
from pathlib import (
    Path,  # ruff: ignore[typing-only-standard-library-import] -- needed by typer
)
from typing import TYPE_CHECKING, Annotated, NamedTuple, cast

import jax.numpy as jnp
import keras
import numpy as np
import typer
from deconvolve.coretypes import SUBSTRUCTURE_VARIABLES, Split, artifacts_dir
from deconvolve.data import load_jet_dataset
from deconvolve.evaluation.evaluate import _improvement, _wd_per_dim
from deconvolve.instrumentation.logging_config import configure_logging
from deconvolve.training.engine import (
    MMD_SUBSAMPLE,
    PARAMS_FILE,
    load_params,
    normalize_weights,
    weighted_bce,
)
from deconvolve.training.mmd import (
    bandwidths,
    build_cache,
    subsample_indices,
    weighted_mmd,
)
from deconvolve.training.models import build_discriminator, build_generator
from jax import Array

if TYPE_CHECKING:
    from logging import Logger
    from typing import Any, Final

    from deconvolve.coretypes import (
        DatasetSplits,
        DeconvolveModel,
        EventArray,
        Populations,
    )
    from deconvolve.training.mmd import MMDCache
    from numpy.typing import NDArray

LOG2: Final[np.double] = np.log(2.0)

# Resolution limit of a float32 sigmoid.
_P_CLIP: Final[float] = 1e-7

logger: Logger = logging.getLogger(name="deconvolve.ceiling")


class Fit(NamedTuple):
    model: keras.Model
    val_bce: float
    epochs_run: int


def _labelled(
    positive: EventArray, negative: EventArray, /
) -> tuple[EventArray, EventArray]:
    """Stack two populations into one classifier training set."""
    return (
        np.concatenate([positive, negative], axis=0),
        np.concatenate(
            [np.ones(shape=len(positive)), np.zeros(shape=len(negative))],
            dtype=np.single,
        ),
    )


def _fit_classifier(
    train_pos: EventArray,
    train_neg: EventArray,
    val_pos: EventArray,
    val_neg: EventArray,
    /,
    *,
    label: str,
    hidden_units: int,
    n_layers: int,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    train_w: EventArray | None = None,
    val_w: EventArray | None = None,
) -> Fit:
    """Converge a plain binary classifier and report its held-out BCE floor.

    `build_discriminator` is reused rather than reimplemented so that the number
    this returns is comparable to deconvolve's `val_d` -- same depth, same width, same
    activations, same sigmoid output, and therefore the same Keras epsilon
    clipping in the loss. Only the training regime differs, which is the point:
    no adversary, no per-event weights, and a fixed target.
    """
    x_train, y_train = _labelled(train_pos, train_neg)
    x_val, y_val = _labelled(val_pos, val_neg)
    # Keras reduces a weighted loss with `sum_over_batch_size` -- it divides by
    # the row count, not by the weight sum -- which is exactly what
    # `deconvolve.training.engine.weighted_bce` does. The two are the same
    # quantity, so a sample-weighted fit here early-stops on the same number C
    # then scores.
    fit_w: dict[str, NDArray[np.single]] | None = (
        None if train_w is None else {"sample_weight": train_w}
    )
    val_data: tuple[NDArray[np.single], ...] = (
        (x_val, y_val) if val_w is None else (x_val, y_val, val_w)
    )

    model: keras.Model = cast(
        typ=keras.Model,
        val=build_discriminator(
            dim=x_train.shape[1], hidden_units=hidden_units, n_layers=n_layers
        ),
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr), loss="binary_crossentropy"
    )
    history: keras.callbacks.History = model.fit(
        x_train,
        y_train,
        validation_data=val_data,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0,
        **(fit_w or {}),
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                min_delta=0.0,
                restore_best_weights=True,
            )
        ],
    )
    curve: list[float] = history.history["val_loss"]
    fit = Fit(model=model, val_bce=min(curve), epochs_run=len(curve))
    logger.info(
        "%-28s val BCE %.6f  (log2 - BCE = %.6f)  [%d epochs]",
        label,
        fit.val_bce,
        LOG2 - fit.val_bce,
        fit.epochs_run,
    )
    return fit


def _likelihood_ratio(model: keras.Model, z: EventArray, /) -> NDArray[np.double]:
    """`p / (1 - p)` from a calibrated classifier, normalized to preserve count.

    The normalization matches `train.normalize_weights` so the weights entering
    the metrics below are on the same footing as the ones Deconvolve produces.
    """
    p: NDArray[np.double] = (
        np.asarray(a=model.predict(z, batch_size=8192, verbose=0))
        .ravel()
        .astype(np.double)
    )
    # A float32 sigmoid saturates to exactly 1.0 above a logit of ~17, and the
    # ratio there is a division by zero. Clip at float32 resolution rather than
    # float64's, which is below what the model can actually resolve, and report
    # how many events landed on the bound -- a ratio pinned at 1e7 is a weight
    # that will dominate every sum it enters, and that is a finding, not a
    # detail to smooth over.
    saturated: int = int(np.count_nonzero(a=(p <= _P_CLIP) | (p >= 1.0 - _P_CLIP)))
    if saturated:
        logger.info(
            "  %d / %d events (%.3f%%) hit the sigmoid clip at p=%g",
            saturated,
            len(p),
            100.0 * saturated / len(p),
            _P_CLIP,
        )
    raw: NDArray[np.double] = np.clip(a=p, a_min=_P_CLIP, a_max=1.0 - _P_CLIP)
    raw = raw / (1.0 - raw)
    return raw * (len(raw) / raw.sum())


def _report_level(
    name: str,
    reference: EventArray,
    comparison: EventArray,
    weights: EventArray,
    variables: tuple[str, ...],
    /,
) -> None:
    before: NDArray[np.double] = _wd_per_dim(ref=reference, comp=comparison)
    after: NDArray[np.double] = _wd_per_dim(
        ref=reference, comp=comparison, weights=weights
    )
    logger.info("  %s level", name)
    for i, var in enumerate(iterable=variables):
        logger.info(
            "    %-6s W %.5f -> %.5f  (%+.1f%%)",
            var,
            before[i],
            after[i],
            _improvement(before[i], after[i]),
        )
    logger.info(
        "    %-6s %+.1f%%",
        "mean",
        float(np.mean(a=list(starmap(_improvement, zip(before, after, strict=True))))),
    )


def _weight_stats(w: EventArray, /) -> None:
    """Effective sample size is what a degenerate reweighting destroys first."""
    ess: np.double = w.sum() ** 2 / np.square(w).sum()
    logger.info(
        "  weights: min %.3g  max %.3g  ESS %.0f / %d (%.1f%%)",
        w.min(),
        w.max(),
        ess,
        len(w),
        100.0 * ess / len(w),
    )


def diagnostic_a(
    train_pop: Populations,
    val_pop: Populations,
    /,
    **kwargs: dict[str, Any],
) -> Fit:
    """How much detector-level signal is there for `d` to find?

    Deliberately fitted on train and scored on val, the same two splits deconvolve's
    `val_d` is built from, so the two numbers are directly comparable.
    """
    logger.info(msg="")
    logger.info(msg="A. Detector-level separability of x_sim from x_data")
    logger.info(msg="   The BCE floor a converged, unweighted classifier reaches.")
    return _fit_classifier(
        train_pop.data,
        train_pop.mc.x,
        val_pop.data,
        val_pop.mc.x,
        label="d*(x): x_data vs x_sim",
        **kwargs,  # ty: ignore[invalid-argument-type]
    )


def diagnostic_b(
    train_pop: Populations,
    val_pop: Populations,
    test_pop: Populations,
    variables: tuple[str, ...],
    /,
    **kwargs: dict[str, Any],
) -> EventArray:
    """Does the weight function that fixes particle level also fix detector level?"""
    logger.info(msg="")
    logger.info(msg="B. The particle-level likelihood ratio, applied at both levels")
    logger.info(msg="   Fitted on train, early-stopped on val, scored on test.")
    # Three splits, not two: a likelihood ratio early-stopped on the same events
    # it is then scored against reports a particle-level match it will not
    # reproduce anywhere else, which is precisely the failure this diagnostic
    # exists to rule out.
    fit: Fit = _fit_classifier(
        train_pop.require_truth(),
        train_pop.mc.z,
        val_pop.require_truth(),
        val_pop.mc.z,
        label="w*(z): z_true vs z_gen",
        **kwargs,  # ty: ignore[invalid-argument-type]
    )
    w: EventArray = _likelihood_ratio(fit.model, test_pop.mc.z).astype(dtype=np.single)
    _weight_stats(w)
    # Particle level is the level w* was fitted to separate, so a large
    # improvement here is a check that the fit worked, not a result.
    _report_level("particle", test_pop.require_truth(), test_pop.mc.z, w, variables)
    # Detector level is the result: the same per-event weights, unmodified.
    _report_level("detector", test_pop.data, test_pop.mc.x, w, variables)
    return w


def _generator_at(
    run_dir: Path, config: dict[str, Any], epoch: int | None, /
) -> DeconvolveModel:
    """The run's generator, either as saved or rebuilt at an arbitrary epoch.

    `generator.keras` holds `best_epoch` alone. Auditing any *other* epoch is
    the reason `train.save_params` writes the whole stack, and it is what
    decides whether a set of epochs the criterion cannot separate is genuinely
    equivalent at detector level or merely unresolved.
    """
    if epoch is None:
        return keras.saving.load_model(artifacts_dir(run_dir) / "generator.keras")  # pyrefly: ignore[no-any-return-implicit]
    if not (artifacts_dir(run_dir) / PARAMS_FILE).exists():
        typer.echo(
            message=f"{run_dir} predates per-epoch parameter saving, so only its"
            f"{run_dir} predates per-epoch parameter saving, so only its "
            f"selected epoch ({config['best_epoch']}) can be audited. "
            "Re-run training to get params.npz, or drop --epoch.",
        )
        raise typer.Exit(code=-1)
    g: DeconvolveModel = build_generator(
        dim=int(config["dim"]),
        hidden_units=int(config["hidden_units"]),
        n_layers=int(config["n_layers"]),
    )
    for var, stacked in zip(
        g.trainable_variables, load_params(run_dir).g_trainable, strict=True
    ):
        var.assign(value=cast(typ=Array, val=np.asarray(a=stacked[epoch])))
    return g


def _run_weights(
    run_dir: Path,
    pop: Populations,
    /,
    *,
    config: dict[str, Any],
    epoch: int | None = None,
) -> tuple[EventArray, EventArray, EventArray]:
    """`(x, y, w)` for one split, weighted by a saved run's generator.

    Rows are `[x_data ; x_sim]`, so `y` is 1 on nature and 0 on MC and the
    weights come back through `deconvolve.training.engine.normalize_weights` -- the same
    normalization the training loop applies, rather than a re-derivation of it.
    """
    g: keras.Model = cast(typ=keras.Model, val=_generator_at(run_dir, config, epoch))
    raw_mc: EventArray = (
        np.asarray(a=g.predict(pop.mc.z, batch_size=8192, verbose=0))
        .ravel()
        .astype(np.single)
    )
    x, y = _labelled(pop.data, pop.mc.x)
    raw: EventArray = np.concatenate(
        [np.ones(shape=len(pop.data), dtype=np.single), raw_mc]
    )
    w: EventArray = np.asarray(
        a=normalize_weights(raw, y, np.ones_like(a=y)), dtype=np.single
    )
    return x, y, w


def _check_run_matches(
    run_dir: Path, variables: tuple[str, ...], n_samples: int, data_seed: int, /
) -> dict[str, Any]:
    """Refuse to score a generator against a sample it was not trained on.

    Column order is the sharp edge here: `variables` indexes columns, so a
    permuted list is a different dataset that a saved model will happily
    consume and report nonsense about. `n_samples` matters too, because
    standardization is fitted on the sample.
    """
    config: dict[str, Any] = json.loads((run_dir / "config.json").read_text())
    mismatches: list[str] = [
        f"{key}: run has {have!r}, this sample is {want!r}"
        for key, have, want in (
            ("variables", tuple(config.get("variables", ())), variables),
            ("n_samples", config.get("n_samples"), n_samples),
            ("data_seed", config.get("data_seed"), data_seed),
        )
        if have != want
    ]
    if mismatches:
        raise SystemExit(
            f"{run_dir} was trained on a different sample:\n  "
            + "\n  ".join(mismatches)
            + "\nRe-run with matching --var/--n-samples/--data-seed."
        )
    return config


def diagnostic_c(
    run_dir: Path,
    train_pop: Populations,
    val_pop: Populations,
    variables: tuple[str, ...],
    n_samples: int,
    data_seed: int,
    floor: float,
    /,
    *,
    epoch: int | None = None,
    **kwargs: dict[str, Any],
) -> None:
    """Did `g` really match detector level, or was deconvolve's `d` just too weak?"""
    logger.info(msg="")
    logger.info(msg="C. A fresh discriminator against a finished run's weights")
    logger.info(
        "   %s, epoch %s", run_dir, "best (as saved)" if epoch is None else epoch
    )
    config: dict[str, Any] = _check_run_matches(
        run_dir, variables, n_samples, data_seed
    )

    _, _, w_tr = _run_weights(run_dir, train_pop, config=config, epoch=epoch)
    x_va, y_va, w_va = _run_weights(run_dir, val_pop, config=config, epoch=epoch)
    _weight_stats(w_va[y_va == 0])

    # Split back into the two populations `_fit_classifier` expects. The
    # weights stay row-aligned with the `[nature ; mc]` stacking `_labelled`
    # rebuilds, so this is a regrouping, not a reordering.
    fit: Fit = _fit_classifier(
        train_pop.data,
        train_pop.mc.x,
        val_pop.data,
        val_pop.mc.x,
        label="fresh d(x): x_data vs w*x_sim",
        train_w=w_tr,
        val_w=w_va,
        **kwargs,  # ty: ignore[invalid-argument-type]
    )
    d_out: EventArray = (
        np.asarray(a=fit.model.predict(x_va, batch_size=8192, verbose=0))
        .ravel()
        .astype(np.single)
    )
    # Scored with the training loop's own function, so this is `val_d` as
    # `history.npz` defines it and not merely something close to it.
    bce: Array = weighted_bce(d_out, y_va, w_va, np.ones_like(a=y_va))
    # `val_d` at the *selected* epoch, not the run's minimum over epochs. Those
    # are different generators: the saved model is the one from `best_epoch`,
    # and the minimum of `val_d` is usually some other epoch entirely, whose
    # weights this fresh `d` never saw. Comparing against the minimum reports a
    # difference between two generators as if it were a difference between two
    # discriminators, and can come out negative.
    curve: NDArray[np.double] = np.asarray(
        a=np.load(file=artifacts_dir(run_dir) / "history.npz")["val_d"],  # pyrefly: ignore[unknown-argument-type]
        dtype=np.double,
    )
    best_epoch: int = int(config["best_epoch"]) if epoch is None else epoch
    deconvolve_val_d: float = float(curve[best_epoch])
    logger.info(
        "  fresh d, scored as val_d   %.6f  (log2 - BCE = %+.6f)", bce, LOG2 - bce
    )
    logger.info(
        "  deconvolve's own d at epoch %-3d   %.6f  (log2 - BCE = %+.6f)",
        best_epoch,
        deconvolve_val_d,
        LOG2 - deconvolve_val_d,
    )
    logger.info(
        "  unweighted floor from A    %.6f  (log2 - BCE = %+.6f)", floor, LOG2 - floor
    )
    present: float = LOG2 - floor
    found = float(LOG2 - bce)
    missed: float = found - (LOG2 - deconvolve_val_d)
    logger.info(
        "  Of the %.6f nats of detector-level mismatch present before "
        "reweighting, g removed %.1f%%, leaving %.6f that a converged d can "
        "still find. deconvolve's own d found %.6f less than that.",
        present,
        100.0 * (1.0 - found / present),
        found,
        missed,
    )
    logger.info(
        "  A converged classifier is the most sensitive two-sample test "
        "available, so %.6f nats bounds what *any* detector-level criterion "
        "has left to discriminate on -- %.1f%% of the original signal.",
        found,
        100.0 * found / present,
    )
    logger.info(
        "  best_epoch %s, mmd_test %.3e",
        config.get("best_epoch"),
        config.get("mmd_test", float("nan")),
    )


def _mmd2(
    reference: EventArray,
    comparison: EventArray,
    w: EventArray,
    i_nature: NDArray[np.intp],
    i_mc: NDArray[np.intp],
    /,
) -> tuple[float, float]:
    """Detector- or particle-level MMD^2 on the subsample, as selection sees it.

    Two index arrays, not one. The nature and MC sides of a `Populations` are
    independent samples and are *not* the same length -- a 1M-event jet run
    splits 199945 / 200055 on test -- so a single draw sized to one side runs
    off the end of the other. `train._detector_arrays` draws them separately
    for the same reason; this mirrors it.
    """
    ref, comp = reference[i_nature], comparison[i_mc]
    sigmas: tuple[float, ...] = bandwidths(jnp.asarray(ref))
    cache: MMDCache = build_cache(ref, comp, sigmas=sigmas)
    mmd2, ess = weighted_mmd(cache, jnp.asarray(a=w[i_mc]))
    return float(mmd2), float(ess)


def diagnostic_d(
    run_dir: Path | None,
    test_pop: Populations,
    oracle_w: EventArray,
    seed: int,
    /,
) -> None:
    """Does the selection criterion prefer the oracle, or prefer deconvolve?

    A and C establish that detector level is nearly saturated after
    reweighting. That leaves one question the resolution of the estimator
    cannot answer: among weight functions that all match at detector level,
    does the criterion *rank* the truth-optimal one first?

    Scoring `w*` -- known-correct at particle level by construction -- on the
    same MMD the training loop selects with settles it. If `w*` scores worse
    than a finished run's own weights, the criterion is not noisy but pointed
    the wrong way, and no subsample size fixes that.
    """
    logger.info(msg="")
    logger.info("D. The selection criterion, scored on the oracle's weights")
    n_mc: int = len(test_pop.mc)
    n_nature: int = len(test_pop.data)
    m: int = min(MMD_SUBSAMPLE, n_mc, n_nature)
    # The same helper the training loop draws its subsample with, rather than a
    # second sampler that could drift from it. One draw per side, reused across
    # both levels: `truth` is the particle-level view of the same events as
    # `data`, and `mc.z` of the same events as `mc.x`, so a shared pair of
    # indices compares the same physical events at both levels.
    i_nature: NDArray[np.intp] = subsample_indices(seed, n_nature, m)
    i_mc: NDArray[np.intp] = subsample_indices(seed + 1, n_mc, m)
    ones: EventArray = np.ones(shape=n_mc, dtype=np.single)

    rows: list[tuple[str, EventArray]] = [
        ("unweighted", ones),
        ("oracle w*(z)", oracle_w),
    ]
    if run_dir is not None:
        config: dict[str, Any] = json.loads(s=(run_dir / "config.json").read_text())
        _, y, w = _run_weights(run_dir, test_pop, config=config)
        rows.append(
            (f"deconvolve {run_dir.name}", np.asarray(a=w[y == 0], dtype=np.single))
        )

    logger.info(
        "   m = %d of %d nature / %d mc test events, median-heuristic bandwidths",
        m,
        n_nature,
        n_mc,
    )
    logger.info(
        "   %-22s %14s %14s %10s", "weights", "detector MMD2", "particle MMD2", "ESS%"
    )
    for label, w in rows:
        d_mmd, ess = _mmd2(test_pop.data, test_pop.mc.x, w, i_nature, i_mc)
        p_mmd, _ = _mmd2(test_pop.require_truth(), test_pop.mc.z, w, i_nature, i_mc)
        logger.info(
            "   %-22s %14.3e %14.3e %9.1f%%", label, d_mmd, p_mmd, 100.0 * ess / m
        )
    logger.info(
        msg="   If the oracle's detector MMD2 is not the smallest, a detector-level"
    )
    logger.info(
        msg="   criterion cannot rank the right answer first at any subsample size."
    )


def main(
    n_samples: Annotated[
        int, typer.Option(help="the number of events to sample")
    ] = 1_000_000,
    var: Annotated[list[str] | None, typer.Option(help="the variables to use")] = None,
    hidden_units: Annotated[int, typer.Option(help="the number of hidden units")] = 128,
    n_layers: Annotated[int, typer.Option(help="the number of layers")] = 3,
    epochs: Annotated[int, typer.Option(help="the number of epochs")] = 200,
    patience: Annotated[int, typer.Option(help="the patience for early stopping")] = 10,
    batch_size: Annotated[int, typer.Option(help="the batch size")] = 1024,
    lr: Annotated[float, typer.Option(help="the learning rate")] = 1e-3,
    data_seed: Annotated[int, typer.Option(help="the data seed")] = 42,
    seed: Annotated[int, typer.Option(help="the random seed")] = 0,
    run_dir: Annotated[
        Path | None, typer.Option(help="a finished run to audit with diagnostic C")
    ] = None,
    epoch: Annotated[
        int | None,
        typer.Option(
            help="audit this epoch instead of the selected one (needs params.npz)"
        ),
    ] = None,
) -> None:

    configure_logging(level="info")
    keras.utils.set_random_seed(seed)

    # Canonical order, exactly as `cli._canonical_variables` fixes it: these
    # names index columns, so a permutation is a different dataset.
    chosen: set[str] = set(var or SUBSTRUCTURE_VARIABLES)
    variables: tuple[str, ...] = tuple(v for v in SUBSTRUCTURE_VARIABLES if v in chosen)

    splits: DatasetSplits
    splits, _dim, _std = load_jet_dataset(
        n_samples=n_samples,
        batch_size=batch_size,
        variables=variables,
        seed=data_seed,
    )
    train_pop: Populations = splits.select(Split.TRAIN).partition()
    val_pop: Populations = splits.select(Split.VAL).partition()
    test_pop: Populations = splits.select(Split.TEST).partition()

    logger.info(
        "%d events over %s, %d train / %d val / %d test (MC side)",
        n_samples,
        ", ".join(variables),
        len(train_pop.mc),
        len(val_pop.mc),
        len(test_pop.mc),
    )
    logger.info(
        "classifier: %d x %d, Adam lr=%g, batch %d, early stop patience %d",
        n_layers,
        hidden_units,
        lr,
        batch_size,
        patience,
    )
    logger.info("log 2 = %.6f", LOG2)

    fit_kwargs: dict[str, Any] = {
        "hidden_units": hidden_units,
        "n_layers": n_layers,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "patience": patience,
    }
    a: Fit = diagnostic_a(train_pop, val_pop, **fit_kwargs)
    oracle_w: EventArray = diagnostic_b(
        train_pop, val_pop, test_pop, variables, **fit_kwargs
    )
    if run_dir is not None:
        diagnostic_c(
            run_dir,
            train_pop,
            val_pop,
            variables,
            n_samples,
            data_seed,
            a.val_bce,
            epoch=epoch,
            **fit_kwargs,
        )
    diagnostic_d(run_dir, test_pop, oracle_w, data_seed)

    logger.info(msg="")
    logger.info(msg="SUMMARY")
    logger.info(
        "  detector-level BCE floor      %.6f   (log2 - floor = %.6f)",
        a.val_bce,
        LOG2 - a.val_bce,
    )
    logger.info(
        msg="\tCompare against `val_d` in a run's history.npz."
        "deconvolve's `d` scoring far"
    )
    logger.info(
        msg="  above this floor means `d` is the bottleneck, not `g`; scoring at it"
    )
    logger.info(msg="  means the residual is real and `g` is doing the work.")


if __name__ == "__main__":
    typer.run(function=main)
