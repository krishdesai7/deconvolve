"""Adversarial training loop for RAN, as a single fused XLA program.

Keras 3 with the JAX backend. The min-max game needs two optimizers driven at
different cadences against a shared loss, which does not fit `Model.fit`, so
this module implements a hand-rolled loop. It follows the standard Keras 3 +
JAX pattern: model state lives in JAX pytrees (`TrainState`) for the duration
of training, updates go through `stateless_call`/`stateless_apply`, and each
step is a single jitted function. Values are written back into the Keras
models at the end so the returned objects are ordinary, saveable
`keras.Model`s.

The training loop is not a Python loop over batches: the dataset is moved to
device once (`deconvolve.data.device`), one epoch is a `lax.scan` over grouped
batch indices, and the epoch loop is a `lax.scan` with a fixed trip count, so a
whole run compiles to one program and the batch gathers fuse into the first
`Dense`. Every epoch's parameters are retained so a checkpoint can be selected
on the host once the scan is done, by detector-level MMD against a validation
subsample.

The loss math is plain `jnp`. `stateless_call`/`stateless_apply` are the only
Keras calls inside the trace. `lax.scan` and `jax.random` are both native JAX.

`train(fused=False)` runs the very same epoch function from an ordinary Python
`while`. It is still one XLA program per epoch, but it keeps breakpoints,
readable tracebacks and host-side logging, which can be helpful for debugging
when a run goes wrong.
"""

from __future__ import annotations

import logging
import operator
import os
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
import keras
import numpy as np
from beartype import beartype
from jax import lax
from jaxtyping import Array, Float, Int, jaxtyped

# `COMPILE_CACHE_DIR` is a runtime value; `Variables` only annotates, but it
# annotates `@jaxtyped(beartype)` and beartype resolves at decoration time
from ..coretypes import COMPILE_CACHE_DIR, Variables, artifacts_dir
from ..data import DeviceSplits, EvalSplit, gather, train_indices
from ..instrumentation import is_enabled, phase
from .mmd import bandwidths, build_cache, mmd_curve, subsample_indices, weighted_mmd
from .models import build_discriminator, build_generator

if TYPE_CHECKING:
    from collections.abc import Callable
    from logging import Logger
    from pathlib import Path
    from typing import Any, Final

    from jax._src.pjit import JitWrapped
    from jax.stages import Compiled
    from jaxtyping import PRNGKeyArray
    from numpy.typing import NDArray

    from ..coretypes import (
        ZXY,
        DatasetSplits,
        DeconvolveModel,
        DiscGradFn,
        EvalStep,
        EventArray,
        GenGradFn,
        KerasVariable,
        StatelessOptimizer,
        TrainStep,
    )
    from ..data import TrainSplit
    from . import MMDCache


logger: Logger = logging.getLogger(name=__name__)

if keras.backend.backend() != "jax":
    # Importing `keras` before `ran` wins the race for the backend, and the
    # jitted steps below fail deep inside a trace.
    raise RuntimeError(
        "deconvolve.training.engine requires the JAX backend, got "
        f"{keras.backend.backend()!r}. "
        "Import `ran` (or any deconvolve.* module) before `keras`, or set "
        "KERAS_BACKEND=jax in the environment."
    )

EPS: Final[float] = keras.config.epsilon()
_HISTORY_KEYS: Final[tuple[str, str, str]] = ("train_d", "train_g", "val_d")
"""The per-epoch history columns the scan records.

Every column is the weighted BCE on the same scale (`_make_pass` negates
`g_loss` back before recording it, so `train_g` is the BCE at the generator's
batch rather than the objective g descends). Validation measures the BCE in
exactly one place: `eval_step` runs once per epoch and both networks are
scored by that number.
"""

# Fixed subsample size for the detector-level MMD comparison selection reads.
# The unbiased estimator has a resolution floor around 5e-4 in MMD^2, measured
# at m=8192 (not this m); the floor scales approximately as 1/m, so 16384 is
# chosen to sit below that figure. Measuring the floor at this operating point
# -- rather than extrapolating to it -- is still outstanding work.
MMD_SUBSAMPLE: Final[int] = 16384


class EpochParams(NamedTuple):
    """Every epoch's model parameters, stacked on a leading epoch axis.

    Both networks, not just `g`: nothing currently reads
    `discriminator.keras`, but an artifact directory whose two models come
    from different epochs is a trap. The non-trainable lists are empty for
    these Dense-only architectures, and are carried anyway so the day someone
    adds BatchNorm this stays correct rather than silently wrong.
    """

    g_trainable: Variables
    g_non_trainable: Variables
    d_trainable: Variables
    d_non_trainable: Variables


class TrainResult(NamedTuple):
    """What a training run returns, unpacked as `(g, d, history, seed, ...)`.

    The fields after `seed` are defaulted so `TrainResult` stays constructible
    from a stub.

    Attributes:
        history: `train_d`, `train_g`, `val_d` (the three `lax.scan` columns)
            plus `val_mmd` and `val_ess` (host additions computed from the
            retained per-epoch parameters, once the scan is done).
        seed: The weight-initialization seed actually used.
        best_epoch: Which epoch's parameters were restored: the argmin of
            `history["val_mmd"]`. Two criteria can select checkpoints tens of
            epochs apart on one run, so a sweep that does not record this
            cannot tell an effect of the hyperparameter from an effect of
            where selection happened to land.
        params: Every epoch's parameters, stacked on a leading epoch axis --
            what makes host-side selection possible at all.
        mmd_test: The honest number: the weighted MMD at `best_epoch`,
            recomputed on a held-out test subsample rather than read off the
            validation curve selection minimized.
        sigmas: The RBF bandwidths `bandwidths()` chose from the validation
            subsample, reused for the test-side cache so both numbers share
            one kernel.
    """

    g: DeconvolveModel
    d: DeconvolveModel
    history: dict[str, list[float]]
    seed: int
    best_epoch: int = -1
    params: EpochParams = EpochParams(
        g_trainable=[], g_non_trainable=[], d_trainable=[], d_non_trainable=[]
    )
    mmd_test: float = float("nan")
    sigmas: tuple[float, ...] = ()


class TrainState(NamedTuple):
    """All mutable training state, as a JAX pytree.

    Held outside the `keras.Model`s so jitted steps stay pure and no
    host/device sync happens between steps.
    """

    g_trainable: Variables
    g_non_trainable: Variables
    d_trainable: Variables
    d_non_trainable: Variables
    opt_g: Variables
    opt_d: Variables


PARAMS_FILE: Final[str] = "params.npz"


def save_params(run_dir: Path, params: EpochParams, /) -> Path:
    """Write every epoch's parameters beside the run's other artifacts.

    Without this, `EpochParams` dies with the process that produced it and any
    question about an epoch other than the selected one costs a full retrain.
    A different selection criterion then becomes a re-read rather than a
    rerun, which is why `scan` emits the stack.

    ~27 MB for 100 epochs of both networks at 3x128, uncompressed for the same
    reason the dataset caches are: these are incompressible floats.
    """
    flat: dict[str, NDArray[Any]] = {
        f"{field}:{i}": np.asarray(a)
        for field, arrays in params._asdict().items()
        for i, a in enumerate(iterable=arrays)
    }
    path: Path = artifacts_dir(run_dir) / PARAMS_FILE
    # Same unpack-into-savez suppression `workflows.train._save_run` carries: a
    # str-keyed dict could in principle hold "allow_pickle", which is declared
    # bool. These keys are all `field:index`, so it cannot.
    np.savez(file=path, **flat)  # ty:ignore[invalid-argument-type]
    return path


def load_params(run_dir: Path, /) -> EpochParams:
    """Read back what `save_params` wrote, in field order.

    Keys carry their list index rather than relying on npz ordering, because
    the lists are positional -- Keras hands variables back to
    `stateless_call` in the order it gave them out, and a permuted list is a
    silently wrong model rather than an error.
    """
    with np.load(file=artifacts_dir(run_dir) / PARAMS_FILE) as f:
        keys: list[str] = list(f.keys())  # pyrefly: ignore[unknown-argument-type]
        return EpochParams(
            **{
                field: [
                    jnp.asarray(a=f[k])  # pyrefly: ignore[unknown-argument-type]
                    for k in sorted(
                        (k for k in keys if k.split(sep=":")[0] == field),
                        key=lambda k: int(k.split(sep=":")[1]),
                    )
                ]
                for field in EpochParams._fields
            }
        )


class RunCarry(NamedTuple):
    """What crosses an epoch boundary in the `lax.scan` over epochs.

    Everything else -- the per-epoch `(train_d, train_g, val_d)` row and the
    full `EpochParams` -- is a `scan` output, not carried state.

    Attributes:
        key: The random key, split once per epoch.
    """

    state: TrainState
    key: PRNGKeyArray


@jaxtyped(typechecker=beartype)
def normalize_weights(
    raw_w: Float[Array | NDArray[np.single], " n"],
    y: Float[Array | NDArray[np.single], " n"],
    mask: Float[Array | NDArray[np.single], " n"],
    /,
) -> Float[Array, " n"]:
    """Per-batch weights: fixed at 1 for nature, renormalized to count for MC.

    `raw_w` is the raw generator output for every event in the batch, already
    squeezed to one dimension. Data events (y=1) are pinned to weight 1; MC
    events (y=0) are rescaled so their weights sum to the MC event count,
    preserving the per-class normalization.

    The y=1 entries of `raw_w` are multiplied by (1 - y) = 0 in both the sum
    and the result, so `g`'s output on data rows, which are `z_true`, never
    reaches the loss or its gradient.

    `mask` is 1 for a real event and 0 for a padding row, and it enters every
    sum so a padded eval batch gives exactly the value the unpadded one would.
    On the training path the mask is all ones and this is the plain form.
    """
    one: Float[Array, " n"] = jnp.ones_like(a=y)
    n_mc: Float[Array, ""] = jnp.sum(a=mask * (one - y))
    # `np.double`, alone among the annotations here: the numpy stubs promote
    # `ndarray * <unknown>` to float64, so a `np.single` union member does not
    # typecheck. At runtime `n_mc` is a JAX array and takes the operation, so
    # this is always `Array` -- the double is a stub artefact, not a dtype.
    w_mc_norm: Float[Array | NDArray[np.double], " n"] = (
        raw_w * n_mc / (jnp.sum(a=mask * raw_w * (one - y)) + EPS)
    )
    return jnp.multiply(mask, y + (one - y) * w_mc_norm)


@jaxtyped(typechecker=beartype)
def weight_dispersion(
    w: Float[Array | NDArray[np.single], " n"],
    y: Float[Array | NDArray[np.single], " n"],
    mask: Float[Array | NDArray[np.single], " n"],
    /,
) -> Float[Array, ""]:
    """How far the generator has travelled from `w = 1`, as one number.

    The variance of the normalised MC weights. It is the natural regulariser
    here because it has a **target** rather than being a free dial:
    `benchmarks/README.md` §2 measures the oracle's ESS at 80.1% against RAN's
    73.3%, so RAN's weights are more dispersed than the truth's. For weights of
    mean 1 the identity is `ESS/n = 1 / (1 + Var(w))`, putting the oracle at
    0.249 and RAN at 0.364 — a coefficient can be tuned to close that gap.

    Nature's rows are excluded, not merely down-weighted: `normalize_weights`
    pins them to exactly 1 and no gradient reaches `g` through them, so
    including them would scale the penalty by the class balance and measure
    nothing. Padded rows go out through `mask`, so a padded batch reports what
    an unpadded one would.

    Variance rather than `E[w log w]` (the KL from nominal, the other canonical
    choice): the two agree to leading order in the dispersion, this one maps
    onto the stated ESS target in closed form, and it has no logarithm to
    protect at small `w`.
    """
    one: Float[Array, " n"] = jnp.ones_like(a=y)
    mc: Float[Array | NDArray[np.single], " n"] = mask * (one - y)
    n_mc: Float[Array, ""] = jnp.sum(a=mc)
    mean_w: Float[Array, ""] = jnp.sum(a=mc * w) / (n_mc + EPS)
    return jnp.sum(a=mc * jnp.square(w - mean_w)) / (n_mc + EPS)


@jaxtyped(typechecker=beartype)
def bce_sums(
    d_out: Float[Array | NDArray[np.single], " n"],
    y: Float[Array | NDArray[np.single], " n"],
    w: Float[Array | NDArray[np.single], " n"],
    mask: Float[Array | NDArray[np.single], " n"],
    /,
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Masked weighted BCE, unnormalized, paired with the count it divides by.

    So a scan can accumulate across batches and divide once with
    `jnp.sum(...) / n` rather than a mean.
    """
    one: Float[Array, " n"] = jnp.ones_like(a=d_out)
    terms: Float[Array | NDArray[np.single], " n"] = w * y * jnp.log(
        d_out + EPS
    ) + w * (one - y) * jnp.log(one - d_out + EPS)
    return -jnp.sum(a=mask * terms), jnp.sum(a=mask)


@jaxtyped(typechecker=beartype)
def weighted_bce(
    d_out: Float[Array | NDArray[np.single], " n"],
    y: Float[Array | NDArray[np.single], " n"],
    w: Float[Array | NDArray[np.single], " n"],
    mask: Float[Array | NDArray[np.single], " n"],
    /,
) -> Float[Array, ""]:
    """Weighted binary cross-entropy over one batch: `bce_sums`, divided out.

    Returns:
        The scalar loss; the shape annotation catches a dropped reduction.
    """
    total, count = bce_sums(d_out, y, w, mask)
    return total / count


def _make_steps(
    g: DeconvolveModel,
    d: DeconvolveModel,
    opt_g: StatelessOptimizer,
    opt_d: StatelessOptimizer,
    # No default: `train` always passes it, and a second copy of the shipped
    # value here is a second place for it to drift.
    lambda_dispersion: float,
) -> tuple[TrainStep, TrainStep, EvalStep]:
    """Build the jitted disc/gen/eval steps, closing over the models.

    The models are captured rather than passed so jit sees only array
    arguments; each returned function is traced once per input shape.
    """

    @jaxtyped(typechecker=beartype)
    def _weights(
        g_trainable: Variables,
        g_non_trainable: Variables,
        z: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
        training: bool,
    ) -> tuple[Float[Array, " n"], Variables]:
        raw_w, g_non_trainable = g.stateless_call(
            trainable_variables=g_trainable,
            non_trainable_variables=g_non_trainable,
            inputs=z,
            training=training,
        )
        return normalize_weights(
            jnp.squeeze(a=raw_w, axis=-1), y, mask
        ), g_non_trainable

    @jaxtyped(typechecker=beartype)
    def _disc_loss(
        d_trainable: Variables,
        d_non_trainable: Variables,
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        w: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], Variables]:
        d_out, d_non_trainable = d.stateless_call(
            trainable_variables=d_trainable,
            non_trainable_variables=d_non_trainable,
            inputs=x,
            training=True,
        )
        loss: Float[Array, ""] = weighted_bce(jnp.squeeze(a=d_out, axis=-1), y, w, mask)
        return loss, d_non_trainable

    @jaxtyped(typechecker=beartype)
    def _gen_loss(
        g_trainable: Variables,
        g_non_trainable: Variables,
        d_trainable: Variables,
        d_non_trainable: Variables,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], tuple[Variables, Float[Array, ""]]]:
        w, g_non_trainable = _weights(
            g_trainable, g_non_trainable, z, y, mask, training=True
        )
        d_out, _ = d.stateless_call(
            trainable_variables=d_trainable,
            non_trainable_variables=d_non_trainable,
            inputs=x,
            training=False,
        )
        # g maximizes the BCE that d minimizes, so its loss is the negation.
        adversarial: Float[Array, ""] = -weighted_bce(
            jnp.squeeze(a=d_out, axis=-1), y, w, mask
        )
        # The penalty steers the gradient but is kept out of the reported
        # number: CLAUDE.md's Training Loop documents the history's three
        # columns as one weighted BCE on one scale, and `losses.pdf` would
        # otherwise plot a penalised curve against two unpenalised ones.
        penalised: Float[Array, ""] = adversarial + lambda_dispersion * (
            weight_dispersion(w, y, mask)
        )
        return penalised, (g_non_trainable, adversarial)

    disc_grad_fn: DiscGradFn = jax.value_and_grad(fun=_disc_loss, has_aux=True)
    gen_grad_fn: GenGradFn = jax.value_and_grad(fun=_gen_loss, has_aux=True)

    @jaxtyped(typechecker=beartype)
    def disc_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[TrainState, Float[Array, ""]]:
        """One discriminator update; g is frozen."""
        # Computed outside differentiated function, so the weights are constants
        w, _ = _weights(
            state.g_trainable, state.g_non_trainable, z, y, mask, training=False
        )
        (loss, d_non_trainable), grads = disc_grad_fn(
            state.d_trainable, state.d_non_trainable, x, y, w, mask
        )
        d_trainable, opt_d_vars = opt_d.stateless_apply(
            optimizer_variables=state.opt_d,
            grads=grads,
            trainable_variables=state.d_trainable,
        )
        return (
            state._replace(
                d_trainable=d_trainable,
                d_non_trainable=d_non_trainable,
                opt_d=opt_d_vars,
            ),
            loss,
        )

    @jaxtyped(typechecker=beartype)
    def gen_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[TrainState, Float[Array, ""]]:
        """One generator update; d is frozen (it enters only as a constant)."""
        (_penalised, (g_non_trainable, adversarial)), grads = gen_grad_fn(
            state.g_trainable,
            state.g_non_trainable,
            state.d_trainable,
            state.d_non_trainable,
            z,
            x,
            y,
            mask,
        )
        g_trainable, opt_g_vars = opt_g.stateless_apply(
            optimizer_variables=state.opt_g,
            grads=grads,
            trainable_variables=state.g_trainable,
        )
        return (
            state._replace(
                g_trainable=g_trainable,
                g_non_trainable=g_non_trainable,
                opt_g=opt_g_vars,
            ),
            # The unpenalised adversarial loss, so what `scan` records stays on
            # the same scale as `train_d` and `val_d`.
            adversarial,
        )

    @jaxtyped(typechecker=beartype)
    def eval_step(
        state: TrainState,
        z: Float[Array, " n d"],
        x: Float[Array, " n d"],
        y: Float[Array, " n"],
        mask: Float[Array, " n"],
    ) -> tuple[Float[Array, ""], Float[Array, ""]]:
        """Weighted BCE with no updates, both models in inference mode."""
        w, _ = _weights(
            state.g_trainable, state.g_non_trainable, z, y, mask, training=False
        )
        d_out, _ = d.stateless_call(
            trainable_variables=state.d_trainable,
            non_trainable_variables=state.d_non_trainable,
            inputs=x,
            training=False,
        )
        return bce_sums(jnp.squeeze(a=d_out, axis=-1), y, w, mask)

    return disc_step, gen_step, eval_step


def _make_pass(
    train: TrainSplit,
    disc_step: TrainStep,
    gen_step: TrainStep,
    *,
    batch_size: int,
    n_disc_steps: int,
) -> Callable[
    [TrainState, PRNGKeyArray], tuple[TrainState, Float[Array, ""], Float[Array, ""]]
]:
    """Build the scanned pass over one epoch's grouped batch indices.

    `d` updates every batch and `g` once per group of `n_disc_steps`, the usual
    adversarial cadence, giving the discriminator a head start each round. The
    generator loss is negated back to d's sign convention so the two curves
    stay directly comparable in the history.
    """

    def _disc_body(
        state: TrainState, idx: Int[Array, " b"]
    ) -> tuple[TrainState, Float[Array, ""]]:
        z, x, y = gather(train, idx)
        return disc_step(state, z, x, y, jnp.ones_like(a=y))

    def _group(
        state: TrainState, group_idx: Int[Array, "s b"]
    ) -> tuple[TrainState, tuple[Float[Array, " s"], Float[Array, ""]]]:
        state, d_losses = lax.scan(f=_disc_body, init=state, xs=group_idx)
        # The generator updates once per group, on the group's first batch.
        z, x, y = gather(train, group_idx[0])
        state, g_loss = gen_step(state, z, x, y, jnp.ones_like(a=y))
        return state, (d_losses, -g_loss)

    def one_pass(
        state: TrainState, key: PRNGKeyArray
    ) -> tuple[TrainState, Float[Array, ""], Float[Array, ""]]:
        idx: Int[Array, " b"] = train_indices(key, train.n, batch_size, n_disc_steps)
        state, (d_losses, g_losses) = lax.scan(f=_group, init=state, xs=idx)
        return (
            state,
            jnp.sum(a=d_losses) / d_losses.size,
            jnp.sum(a=g_losses) / g_losses.size,
        )

    return one_pass


def _make_eval(
    eval_step: EvalStep,
) -> Callable[[TrainState, EvalSplit], Float[Array, ""]]:
    """Build the scanned forward pass over a pre-batched, masked split."""

    def evaluate(state: TrainState, split: EvalSplit) -> Float[Array, ""]:
        def _body(
            acc: tuple[Float[Array, ""], Float[Array, ""]],
            batch: tuple[Float[Array, "b d"], ...],
        ) -> tuple[tuple[Float[Array, ""], Float[Array, ""]], None]:
            z, x, y, mask = batch
            total, count = eval_step(state, z, x, y, mask)
            return (acc[0] + total, acc[1] + count), None

        zero: Float[Array, ""] = jnp.zeros(shape=(), dtype=split.z.dtype)
        (total, count), _ = lax.scan(
            f=_body, init=(zero, zero), xs=(split.z, split.x, split.y, split.mask)
        )
        return total / count

    return evaluate


def _make_epoch(
    data: DeviceSplits,
    one_pass: Callable[
        [TrainState, PRNGKeyArray],
        tuple[TrainState, Float[Array, ""], Float[Array, ""]],
    ],
    evaluate: Callable[[TrainState, EvalSplit], Float[Array, ""]],
    *,
    n_epochs: int,
    log_every: int,
) -> Callable[
    [RunCarry, Int[Array, ""]],
    tuple[RunCarry, tuple[Float[Array, " metrics"], EpochParams]],
]:
    """Build the pure ``(RunCarry, epoch) -> (RunCarry, outputs)`` scan body.

    Nothing about model quality is decided here. The loop trains, records, and
    emits; selection is a host-side read of what it emitted, keeping `z_true`
    out of the traced program entirely.
    """

    def _log(
        epoch: Int[Array, ""],
        train_d: Float[Array, ""],
        train_g: Float[Array, ""],
        val_d: Float[Array, ""],
    ) -> None:
        logger.info(
            "Epoch %3d/%d  D: %.4f  G: %.4f  | Val: %.4f",
            int(epoch) + 1,
            n_epochs,
            float(train_d),
            float(train_g),
            float(val_d),
        )

    def epoch(
        carry: RunCarry, epoch_idx: Int[Array, ""]
    ) -> tuple[RunCarry, tuple[Float[Array, " metrics"], EpochParams]]:
        key, subkey = jax.random.split(carry.key)
        state, train_d, train_g = one_pass(carry.state, subkey)
        val_d: Float[Array, ""] = evaluate(state, data.val)
        lax.cond(
            pred=(epoch_idx % log_every) == 0,
            true_fun=lambda: jax.debug.callback(
                _log, epoch_idx, train_d, train_g, val_d, ordered=True
            ),
            false_fun=lambda: None,
        )
        row: Float[Array, " metrics"] = jnp.stack(arrays=[train_d, train_g, val_d])
        params = EpochParams(
            g_trainable=state.g_trainable,
            g_non_trainable=state.g_non_trainable,
            d_trainable=state.d_trainable,
            d_non_trainable=state.d_non_trainable,
        )
        return RunCarry(state=state, key=key), (row, params)

    return epoch


def _assign(variables: list[KerasVariable], values: Variables) -> None:
    """Write JAX arrays back into a model's `keras.Variable`s."""
    for var, val in zip(variables, values, strict=False):
        var.assign(value=val)


def _initial_state(
    g: DeconvolveModel,
    d: DeconvolveModel,
    opt_g: StatelessOptimizer,
    opt_d: StatelessOptimizer,
) -> TrainState:
    return TrainState(
        g_trainable=[v.value for v in g.trainable_variables],
        g_non_trainable=[v.value for v in g.non_trainable_variables],
        d_trainable=[v.value for v in d.trainable_variables],
        d_non_trainable=[v.value for v in d.non_trainable_variables],
        opt_g=[v.value for v in opt_g.variables],
        opt_d=[v.value for v in opt_d.variables],
    )


def _run(
    carry: RunCarry,
    epoch: Callable[[RunCarry, Array], tuple[RunCarry, tuple[Array, EpochParams]]],
    *,
    n_epochs: int,
    fused: bool,
) -> tuple[RunCarry, Float[Array, "epochs metrics"], EpochParams]:
    """Drive the epoch loop, either inside XLA or from Python."""
    steps: Int[Array, " epochs"] = jnp.arange(start=n_epochs, dtype=jnp.int32)
    if fused:
        run: JitWrapped = jax.jit(lambda c, xs: lax.scan(epoch, c, xs))  # pyrefly: ignore[unknown-argument-type]
        if is_enabled():
            # Ahead-of-time, so the timer can see where compile ends and
            # execution begins. `lower().compile()` then calling the compiled
            # object does the same work `run(carry, steps)` does internally,
            # persistent cache included, split at a boundary an ordinary call
            # does not expose. Gated, so the default path stays the single
            # call `TestFusion` pins.
            with phase("compile") as timer:
                compiled: Compiled = timer.block(run.lower(carry, steps).compile())
            with phase("epochs") as timer:
                carry, (rows, params) = cast(
                    typ="tuple[RunCarry, tuple[Array, EpochParams]]",
                    val=timer.block(compiled(carry, steps)),  # pyrefly: ignore[unknown-argument-type]
                )
            return carry, rows, params
        carry, (rows, params) = cast(
            typ="tuple[RunCarry, tuple[Array, EpochParams]]",
            val=run(carry, steps),
        )
        return carry, rows, params
    step: JitWrapped = jax.jit(epoch)
    rows_out: list[Array] = []
    params_out: list[EpochParams] = []
    for i in range(n_epochs):
        carry, (row, params) = step(carry, steps[i])
        rows_out.append(row)
        params_out.append(params)
    stacked: EpochParams = cast(
        typ="EpochParams",
        val=jax.tree.map(lambda *leaves: jnp.stack(arrays=leaves), *params_out),
    )
    return carry, jnp.stack(arrays=rows_out), stacked


def _restore(
    g: DeconvolveModel, d: DeconvolveModel, params: EpochParams, epoch: int, /
) -> None:
    """Write one epoch's parameters back into the live Keras models."""
    chosen: EpochParams = cast(
        typ="EpochParams", val=jax.tree.map(f=operator.itemgetter(epoch), tree=params)
    )
    _assign(g.trainable_variables, values=chosen.g_trainable)
    _assign(g.non_trainable_variables, values=chosen.g_non_trainable)
    _assign(d.trainable_variables, values=chosen.d_trainable)
    _assign(d.non_trainable_variables, values=chosen.d_non_trainable)


def _detector_arrays(
    zxy: ZXY, seed: int, m: int, /
) -> tuple[EventArray, EventArray, EventArray]:
    """Subsample the detector-level comparison and the generator's input.

    `z` is indexed **only** where `y == 0`. The nature rows of `z` hold
    `z_true`, and this module must never read them -- which is why selection
    is built from `ZXY` rather than from `partition()`, whose `Populations`
    would put truth in scope even if nothing used it.
    """
    nature: NDArray[np.bool] = zxy.y == 1
    mc: NDArray[np.bool] = ~nature
    x_data: EventArray = zxy.x[nature]
    x_sim: EventArray = zxy.x[mc]
    z_gen: EventArray = zxy.z[mc]
    i_d: NDArray[np.intp] = subsample_indices(seed, x_data.shape[0], m)
    i_m: NDArray[np.intp] = subsample_indices(seed + 1, x_sim.shape[0], m)
    return x_data[i_d], x_sim[i_m], z_gen[i_m]


def _weights_per_epoch(
    g: DeconvolveModel, params: EpochParams, z: EventArray, /
) -> Float[Array, "epochs m"]:
    """`g`'s raw output on a fixed sample, for every retained epoch.

    A plain Python loop rather than `vmap`: the non-trainable lists are empty
    for these architectures, which gives `vmap` no leaf to infer a batch size
    from, and 100 forward passes of a 34k-parameter MLP is milliseconds.
    """

    @jax.jit
    def one(trainable: Variables, non_trainable: Variables) -> Float[Array, " m"]:
        raw, _ = g.stateless_call(
            trainable_variables=trainable,
            non_trainable_variables=non_trainable,
            inputs=jnp.asarray(a=z),
            training=False,
        )
        return jnp.squeeze(a=raw, axis=-1)

    n_epochs: int = params.g_trainable[0].shape[0]
    return jnp.stack(
        arrays=[
            one(
                [leaf[i] for leaf in params.g_trainable],
                [leaf[i] for leaf in params.g_non_trainable],
            )
            for i in range(n_epochs)
        ]
    )


def _select_by_mmd(
    g: DeconvolveModel,
    d: DeconvolveModel,
    splits: DatasetSplits,
    params: EpochParams,
    history: dict[str, list[float]],
    n_epochs: int,
    seed: int,
) -> tuple[int, float, tuple[float, ...]]:
    """Pick the epoch minimizing detector-level MMD against a val subsample.

    Host-side, after `_run` -- nothing about model quality is decided in the
    trace. Both subsamples are drawn from `splits.train.seed` (`data_seed`),
    so every hyperparameter arm compares against an identical kernel and
    identical events.
    """
    x_data, x_sim, z_gen = _detector_arrays(
        splits.val.as_arrays(), splits.train.seed, MMD_SUBSAMPLE
    )
    sigmas: tuple[float, ...] = bandwidths(jnp.asarray(a=x_data))
    cache: MMDCache = build_cache(
        jnp.asarray(a=x_data), jnp.asarray(a=x_sim), sigmas=sigmas
    )
    raw_w: Float[Array, "epochs m"] = _weights_per_epoch(g, params, z_gen)
    mmd, ess = mmd_curve(cache, raw_w)
    history["val_mmd"] = mmd.tolist()
    history["val_ess"] = ess.tolist()

    best_epoch: int = int(np.argmin(a=mmd))
    logger.info(
        "Restoring epoch %d of %d  (val MMD^2 %.3e, ESS %.0f)",
        best_epoch + 1,
        n_epochs,
        mmd[best_epoch],
        ess[best_epoch],
    )
    _restore(g, d, params, best_epoch)

    # Recomputed on test, because selecting 100 times against one val
    # subsample is exactly the regime where the estimator starts fitting the
    # sample rather than the distribution.
    tx_data, tx_sim, tz_gen = _detector_arrays(
        splits.test.as_arrays(), splits.train.seed + 2, MMD_SUBSAMPLE
    )
    test_cache: MMDCache = build_cache(
        jnp.asarray(tx_data), jnp.asarray(tx_sim), sigmas=sigmas
    )
    mmd_test = float(
        weighted_mmd(test_cache, _weights_per_epoch(g, params, tz_gen)[best_epoch])[0]
    )
    logger.info("Test MMD^2: %.3e  (init seed %d)", mmd_test, seed)

    return best_epoch, mmd_test, sigmas


def _use_compilation_cache() -> None:
    """Point XLA's persistent cache at `COMPILE_CACHE_DIR`.

    Compilation is the largest single time cost in a short run:
    `benchmarks/boundary.py` on an A100 measures 4.60s of compile time against
    0.034s per epoch, so a 100-epoch run spends half its wall clock in XLA. The
    cache keys on lowered HLO rather than on Python identity, so a freshly
    built `jax.jit` still hits it. It lives on disk, so an ensemble of N
    interpreters compiling the same architecture pays the cost once instead of
    N times.

    It sets `min_compile_time_secs` to zero rather than leaving JAX's default
    of 1.0s, which would leave RAN's cache entirely empty: the run compiles a
    few dozen executables that total 4.6s and no single one of them clears a
    second.

    Whatever the caller configured wins, so `JAX_COMPILATION_CACHE_DIR`,
    `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`, or a `jax.config.update`
    before `train` still overrides this, and an unwritable directory costs a
    warning from JAX rather than the run.

    The path is resolved before it is handed over: JAX opens the cache once
    and keeps the string, so the default's leading `.` would follow any later
    `chdir` and turn every write into a `FileNotFoundError` -- which JAX also
    reports as a warning rather than an error, so the run would go on quietly
    recompiling. Resolving pins it to the directory the datasets came from.
    """
    if jax.config.jax_compilation_cache_dir is not None:
        return
    jax.config.update(
        name="jax_compilation_cache_dir", val=str(object=COMPILE_CACHE_DIR.resolve())
    )
    if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
        jax.config.update(name="jax_persistent_cache_min_compile_time_secs", val=0.0)
    logger.debug("XLA compilation cache: %s", COMPILE_CACHE_DIR.resolve())


def _unpack_history(history: Float[Array, "epochs metrics"]) -> dict[str, list[float]]:
    rows: NDArray[Any] = np.asarray(a=history)
    return {key: rows[:, i].tolist() for i, key in enumerate(iterable=_HISTORY_KEYS)}


def train(
    splits: DatasetSplits,
    dim: int,
    hidden_units: int,
    n_layers: int,
    seed: int | None,
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 3e-5,
    lr_d: float = 1e-4,
    lambda_dispersion: float = 0.015,
    log_every: int = 1,
    *,
    fused: bool = True,
) -> TrainResult:
    """Train the generator and discriminator, then select a checkpoint.

    `seed` seeds weight initialization *only*. The train/val/test split and
    the per-epoch batch order come from the dataset's own seed
    (`DeconvolveDataset`), which draws from an independent generator. Varying
    `seed` across runs therefore estimates training/initialization variance at
    fixed data, i.e., the usual HEP model-uncertainty ensemble, while varying
    the dataset seed instead would fold in split variance.

    The networks are Dense-only with no dropout or batch norm and Adam is
    deterministic, so the two seeds together fully determine a run (up to
    non-deterministic GPU reductions).

    There is no `patience` or `min_delta`: `n_epochs` is a fixed `lax.scan`
    trip count, and every epoch's parameters are retained so selection can
    happen once the scan is done, reading a detector-level MMD curve rather
    than tracking a running best inside the trace.

    Args:
        seed: The weight-initialization seed. `None` draws one from system
            entropy. Either way the value used is returned, so a run stays
            reproducible after the fact without having to decide up front that
            it is worth reproducing.
        n_disc_steps: Discriminator steps per generator step.
        fused: Whether to run the whole epoch loop as one `lax.scan` (the
            default) or drive the identical per-epoch function from a Python
            `while` for debugging; the two must agree bit-for-bit.
    """
    _use_compilation_cache()

    if seed is None:
        # No-argument SeedSequence always fills `entropy` with int drawn from OS.
        seed = cast(typ=int, val=np.random.SeedSequence().entropy) % 2**31
    keras.utils.set_random_seed(seed)

    batch_size: int = splits.train.batch_size
    with phase("transfer") as timer:
        # The single host->device transfer of a run. Blocked inside the clock,
        # or an async copy is charged to whatever runs next.
        data: DeviceSplits = timer.block(DeviceSplits.from_splits(splits))

    g: DeconvolveModel = build_generator(
        dim=dim, hidden_units=hidden_units, n_layers=n_layers
    )
    d: DeconvolveModel = build_discriminator(
        dim=dim, hidden_units=hidden_units, n_layers=n_layers
    )
    opt_g: StatelessOptimizer = keras.optimizers.Adam(learning_rate=lr_g)
    opt_d: StatelessOptimizer = keras.optimizers.Adam(learning_rate=lr_d)
    opt_g.build(g.trainable_variables)
    opt_d.build(d.trainable_variables)

    state: TrainState = _initial_state(g, d, opt_g, opt_d)
    disc_step, gen_step, eval_step = _make_steps(g, d, opt_g, opt_d, lambda_dispersion)
    one_pass: Callable[
        [TrainState, PRNGKeyArray],
        tuple[TrainState, Float[Array, ""], Float[Array, ""]],
    ] = _make_pass(
        data.train,
        disc_step,
        gen_step,
        batch_size=batch_size,
        n_disc_steps=n_disc_steps,
    )
    evaluate: Callable[[TrainState, EvalSplit], Float[Array, ""]] = _make_eval(
        eval_step
    )
    epoch: Callable[
        [RunCarry, Int[Array, ""]],
        tuple[RunCarry, tuple[Float[Array, " metrics"], EpochParams]],
    ] = _make_epoch(data, one_pass, evaluate, n_epochs=n_epochs, log_every=log_every)

    # Batch order follows `data_seed`, not the init seed: an ensemble loop over
    # `--seed` must see identical data on every arm.
    carry = RunCarry(state=state, key=jax.random.key(data.data_seed))

    _final, rows, params = _run(carry, epoch, n_epochs=n_epochs, fused=fused)
    history: dict[str, list[float]] = _unpack_history(history=rows)

    with phase("select"):
        best_epoch, mmd_test, sigmas = _select_by_mmd(
            g, d, splits, params, history, n_epochs, seed
        )

    logger.info("Init seed %d", seed)
    return TrainResult(g, d, history, seed, best_epoch, params, mmd_test, sigmas)
