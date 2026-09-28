# pyrefly: ignore-errors[unknown-argument-type]
# -- numpy stubs return Any for elementwise operations
"""Tests for the JAX training internals.

These exercise the loss math and a couple of real gradient steps on tiny
models. They do NOT train to convergence -- that is cluster work.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import keras
import numpy as np
import pytest
from deconvolve.coretypes import (
    COMPILE_CACHE_DIR,
    LOG2,
    TRUTH_SENTINEL,
    ZXY,
    Events,
    Split,
)
from deconvolve.data import DeconvolveDataset, DeviceSplits, train_indices
from deconvolve.training.engine import (
    EPS,
    TrainResult,
    TrainState,
    _make_steps,
    _use_compilation_cache,
    bce_sums,
    load_params,
    normalize_weights,
    save_params,
    train,
    weight_dispersion,
    weighted_bce,
)
from deconvolve.training.models import build_generator

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from typing import Any

    from deconvolve.coretypes import (
        DatasetSplits,
        DeconvolveModel,
        EvalStep,
        TrainStep,
        Variables,
    )
    from deconvolve.training.engine import EpochParams
    from jax._src.basearray import Array
    from numpy.typing import NDArray


def test_backend_is_jax_pinned_to_single_precision() -> None:
    """The pin is a package-import side effect, so it needs a guard.

    `JAX_ENABLE_X64=0` means a float64 request is silently truncated rather
    than refused, which is exactly the kind of thing that goes unnoticed until
    a run's numbers drift. Asserting the truncation makes the policy explicit.
    """
    import jax.numpy as jnp

    assert keras.backend.backend() == "jax"
    assert jnp.zeros(shape=1).dtype == np.float32
    with pytest.warns(expected_warning=UserWarning, match="float64"):
        assert jnp.zeros(shape=1, dtype="float64").dtype == np.float32


class TestNormalizeWeights:
    def test_data_events_are_pinned_to_one(self) -> None:
        rng: np.random.Generator = np.random.default_rng(seed=0)
        raw: NDArray[np.single] = rng.uniform(low=0.1, high=3.0, size=64).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=64) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = np.asarray(
            a=normalize_weights(raw, y, np.ones(shape=raw.size, dtype=np.single))
        )
        np.testing.assert_allclose(actual=w[y == 1], desired=1.0)

    def test_mc_weights_preserve_event_count(self) -> None:
        """Reweighted MC must keep the same total as unweighted MC."""
        rng: np.random.Generator = np.random.default_rng(seed=1)
        raw: NDArray[np.single] = rng.uniform(low=0.1, high=3.0, size=256).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=256) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = np.asarray(
            a=normalize_weights(raw, y, np.ones(shape=raw.size, dtype=np.single))
        )
        np.testing.assert_allclose(
            actual=w[y == 0].sum(),
            desired=(y == 0).sum(),
            rtol=1e-6,
        )

    def test_mc_weights_stay_proportional_to_raw_output(self) -> None:
        rng: np.random.Generator = np.random.default_rng(seed=2)
        raw: NDArray[np.single] = rng.uniform(low=0.1, high=3.0, size=128).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=128) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = np.asarray(
            a=normalize_weights(raw, y, np.ones(shape=raw.size, dtype=np.single))
        )
        ratio: NDArray[np.single] = w[y == 0] / raw[y == 0]
        np.testing.assert_allclose(actual=ratio, desired=ratio[0], rtol=1e-6)

    def test_generator_output_on_data_rows_cannot_change_the_result(self) -> None:
        """The z_true guard: g's output on y=1 rows must not affect any weight.

        Data rows carry z_true, so if poisoning g's output there moved the
        weights, z_true would be reachable from the loss.
        """
        rng: np.random.Generator = np.random.default_rng(seed=3)
        raw: NDArray[np.single] = rng.uniform(low=0.1, high=3.0, size=128).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=128) < 0.5).astype(dtype=np.single)
        poisoned: NDArray[np.single] = raw.copy()
        poisoned[y == 1] = -999.0
        np.testing.assert_allclose(
            actual=np.asarray(
                a=normalize_weights(raw, y, np.ones(shape=raw.size, dtype=np.single))
            ),
            desired=np.asarray(
                a=normalize_weights(
                    poisoned, y, np.ones(shape=poisoned.size, dtype=np.single)
                )
            ),
        )


@pytest.mark.slow
def test_a_missing_particle_level_cannot_poison_the_batch() -> None:
    """Why `TRUTH_SENTINEL` is a number: the z_true guard is arithmetic.

    A sample built by `Populations.create` without truth carries the sentinel
    in the nature rows of `z`, so g forward-passes it. Multiplying g's output
    there by (1 - y) = 0 annihilates a number, which is what makes those rows
    irrelevant -- but under IEEE 754 it does not annihilate a NaN, which would
    instead reach every weight in the batch through the normalizing sum.
    """
    rng: np.random.Generator = np.random.default_rng(seed=6)
    mc_z: NDArray[np.single] = rng.normal(size=(6, 1)).astype(dtype=np.single)
    y: NDArray[np.single] = np.array(object=[1.0] * 4 + [0.0] * 6, dtype=np.single)
    g: DeconvolveModel = build_generator(dim=1)

    def weights(nature_fill: float) -> NDArray[np.single]:
        z: NDArray[np.single] = np.concatenate(
            [np.full(shape=(4, 1), fill_value=nature_fill, dtype=np.single), mc_z],
            axis=0,
        )
        raw: NDArray[np.single] = np.squeeze(a=np.asarray(a=g(z)), axis=-1)
        return np.asarray(
            a=normalize_weights(raw, y, np.ones(shape=raw.size, dtype=np.single))
        )

    np.testing.assert_allclose(
        actual=weights(nature_fill=float(TRUTH_SENTINEL)),
        desired=weights(nature_fill=0.0),
    )
    assert np.all(a=np.isnan(weights(nature_fill=np.nan)))


class TestWeightedBCE:
    def test_matches_a_float64_reference_to_single_precision(self) -> None:
        """The reduction must not lose more than single-precision rounding.

        Scored against a float64 numpy reference: 512 terms accumulated well
        stay within a few float32 ULPs of it, while a sloppily ordered or
        narrower-than-float32 accumulation drifts further. This is the test that
        pinned `jnp.sum(...) / n` over a mean back when `keras.ops.mean` picked
        a float32 compute dtype for float64 input; the hazard is gone with the
        pipeline at float32, but the reduction is still the thing being pinned.
        """
        rng: np.random.Generator = np.random.default_rng(seed=4)
        d_out: NDArray[np.single] = rng.uniform(low=0.01, high=0.99, size=512).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=512) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = rng.uniform(low=0.5, high=1.5, size=512).astype(
            dtype=np.single
        )

        wide: tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]] = (
            d_out.astype(dtype=np.double),
            y.astype(dtype=np.double),
            w.astype(dtype=np.double),
        )
        expected: float = -np.mean(
            a=wide[2] * wide[1] * np.log(wide[0] + EPS)
            + wide[2] * (1 - wide[1]) * np.log(1 - wide[0] + EPS)
        )
        got = float(
            weighted_bce(d_out, y, w, np.ones(shape=d_out.size, dtype=np.single))
        )
        np.testing.assert_allclose(actual=got, desired=expected, rtol=1e-6)

    def test_perfect_classifier_scores_near_zero(self) -> None:
        y: NDArray[np.single] = np.array(object=[1.0, 1.0, 0.0, 0.0], dtype=np.single)
        d_out: NDArray[np.single] = np.array(
            object=[1.0, 1.0, 0.0, 0.0], dtype=np.single
        )
        w: NDArray[np.single] = np.ones(shape=4, dtype=np.single)
        mask: NDArray[np.single] = np.ones(shape=d_out.size, dtype=np.single)
        assert float(weighted_bce(d_out, y, w, mask)) < 1e-6

    def test_uninformative_classifier_scores_log2(self) -> None:
        y: NDArray[np.single] = np.array(object=[1.0, 1.0, 0.0, 0.0], dtype=np.single)
        d_out: NDArray[np.single] = np.full(shape=4, fill_value=0.5, dtype=np.single)
        w: NDArray[np.single] = np.ones(shape=4, dtype=np.single)
        mask: NDArray[np.single] = np.ones(shape=d_out.size, dtype=np.single)
        np.testing.assert_allclose(
            actual=float(weighted_bce(d_out, y, w, mask)),
            desired=np.log(2),
            atol=1e-6,
        )

    def test_masked_rows_are_ignored_outright(self) -> None:
        """Zeroing `mask` must equal deleting the rows, denominator included.

        This is what makes a padded eval batch report the number an unpadded
        one would: a zero weight still counts toward the mean, a zero mask does
        not.
        """
        rng: np.random.Generator = np.random.default_rng(seed=9)
        d_out: NDArray[np.single] = rng.uniform(low=0.01, high=0.99, size=8).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=8) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = rng.uniform(low=0.5, high=1.5, size=8).astype(
            dtype=np.single
        )
        mask: NDArray[np.single] = np.ones(shape=8, dtype=np.single)
        mask[-2:] = 0.0
        np.testing.assert_allclose(
            actual=float(weighted_bce(d_out, y, w, mask)),
            desired=float(
                weighted_bce(
                    d_out[:-2], y[:-2], w[:-2], np.ones(shape=6, dtype=np.single)
                )
            ),
            rtol=1e-6,
        )

    def test_sums_divide_to_the_mean(self) -> None:
        """`bce_sums` is the reduction split in two, so a scan can accumulate."""
        rng: np.random.Generator = np.random.default_rng(seed=10)
        d_out: NDArray[np.single] = rng.uniform(low=0.01, high=0.99, size=16).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=16) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = rng.uniform(low=0.5, high=1.5, size=16).astype(
            dtype=np.single
        )
        mask: NDArray[np.single] = np.ones(shape=16, dtype=np.single)
        total, count = bce_sums(d_out, y, w, mask)
        assert int(count) == 16
        np.testing.assert_allclose(
            actual=float(total) / float(count),
            desired=float(weighted_bce(d_out, y, w, mask)),
            rtol=1e-6,
        )

    def test_summing_batches_matches_scoring_them_together(self) -> None:
        """Accumulating (total, count) across batches gives the true mean --
        not the mean of per-batch means the old host loop computed."""
        rng: np.random.Generator = np.random.default_rng(seed=12)
        d_out: NDArray[np.single] = rng.uniform(low=0.01, high=0.99, size=10).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=10) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = rng.uniform(low=0.5, high=1.5, size=10).astype(
            dtype=np.single
        )
        halves: list[tuple[Array, Array]] = [
            bce_sums(d_out[s], y[s], w[s], np.ones(shape=len(y[s]), dtype=np.single))
            for s in (slice(0, 6), slice(6, 10))
        ]
        combined: float = sum(float(t) for t, _ in halves) / sum(
            float(c) for _, c in halves
        )
        np.testing.assert_allclose(
            actual=combined,
            desired=float(
                weighted_bce(d_out, y, w, np.ones(shape=10, dtype=np.single))
            ),
            rtol=1e-6,
        )

    def test_zero_weight_events_are_ignored(self) -> None:
        rng: np.random.Generator = np.random.default_rng(seed=5)
        d_out: NDArray[np.single] = rng.uniform(low=0.01, high=0.99, size=8).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = (rng.random(size=8) < 0.5).astype(dtype=np.single)
        w: NDArray[np.single] = np.ones(shape=8, dtype=np.single)
        w_masked: NDArray[np.single] = w.copy()
        w_masked[-2:] = 0.0
        full: float = float(
            weighted_bce(d_out, y, w_masked, np.ones(shape=8, dtype=np.single))
        )
        trimmed: float = (
            float(
                weighted_bce(
                    d_out[:-2], y[:-2], w[:-2], np.ones(shape=6, dtype=np.single)
                )
            )
            * 6
            / 8
        )
        np.testing.assert_allclose(actual=full, desired=trimmed, rtol=1e-6)


class TestTrainSteps:
    """One real disc/gen update each, on tiny models."""

    @staticmethod
    def _setup(
        dim: int = 2, n: int = 64, lambda_dispersion: float = 0.0
    ) -> tuple[tuple[TrainStep, TrainStep, EvalStep], TrainState, tuple[Array, ...]]:
        from deconvolve.training.models import build_discriminator, build_generator

        keras.utils.set_random_seed(0)
        g: DeconvolveModel = build_generator(dim=dim, hidden_units=8, n_layers=1)
        d: DeconvolveModel = build_discriminator(dim=dim, hidden_units=8, n_layers=1)
        opt_g = keras.optimizers.Adam(learning_rate=1e-2)
        opt_d = keras.optimizers.Adam(learning_rate=1e-2)
        opt_g.build(var_list=g.trainable_variables)
        opt_d.build(var_list=d.trainable_variables)
        state = TrainState(
            g_trainable=[v.value for v in g.trainable_variables],
            g_non_trainable=[v.value for v in g.non_trainable_variables],
            d_trainable=[v.value for v in d.trainable_variables],
            d_non_trainable=[v.value for v in d.non_trainable_variables],
            opt_g=[v.value for v in opt_g.variables],
            opt_d=[v.value for v in opt_d.variables],
        )
        rng: np.random.Generator = np.random.default_rng(seed=6)
        # Device arrays, because that is what the steps see: they are traced
        # inside the epoch program, never called on host NumPy. Labels arrive
        # already promoted to the compute dtype -- `TrainSplit` does that once,
        # on the way to device, rather than per `1 - y`.
        batch: tuple[Array, ...] = tuple(
            jnp.asarray(a)
            for a in (
                rng.normal(size=(n, dim)).astype(dtype=np.single),
                rng.normal(size=(n, dim)).astype(dtype=np.single),
                (rng.random(size=n) < 0.5).astype(dtype=np.double),
                # On the training path the mask is always all ones.
                np.ones(shape=n),
            )
        )
        return _make_steps(g, d, opt_g, opt_d, lambda_dispersion), state, batch

    @pytest.mark.slow
    def test_disc_step_updates_only_the_discriminator(self) -> None:
        (disc_step, _, _), state, batch = self._setup()
        new, loss = disc_step(state, *batch)
        assert np.isfinite(float(loss))
        for before, after in zip(state.d_trainable, new.d_trainable, strict=False):
            assert not np.allclose(a=np.asarray(a=before), b=np.asarray(a=after))
        for before, after in zip(state.g_trainable, new.g_trainable, strict=False):
            np.testing.assert_array_equal(
                actual=np.asarray(a=before), desired=np.asarray(a=after)
            )

    @pytest.mark.slow
    def test_gen_step_updates_only_the_generator(self) -> None:
        (_, gen_step, _), state, batch = self._setup()
        new, loss = gen_step(state, *batch)
        assert np.isfinite(float(loss))
        for before, after in zip(state.g_trainable, new.g_trainable, strict=False):
            assert not np.allclose(a=np.asarray(a=before), b=np.asarray(a=after))
        for before, after in zip(state.d_trainable, new.d_trainable, strict=False):
            np.testing.assert_array_equal(
                actual=np.asarray(a=before), desired=np.asarray(a=after)
            )

    def test_gen_and_disc_losses_are_opposite(self) -> None:
        (disc_step, gen_step, _eval_step), state, batch = self._setup()
        _, d_loss = disc_step(state, *batch)
        _, g_loss = gen_step(state, *batch)
        np.testing.assert_allclose(
            actual=float(g_loss), desired=-float(d_loss), rtol=1e-6
        )

    def test_eval_step_leaves_state_untouched_and_matches_disc_loss(self) -> None:
        (disc_step, _, eval_step), state, batch = self._setup()
        _, d_loss = disc_step(state, *batch)
        # eval_step hands back the two halves unreduced, so a scan can add them
        # up across batches and divide once.
        total, count = eval_step(state, *batch)
        np.testing.assert_allclose(
            actual=float(total) / float(count), desired=float(d_loss), rtol=1e-6
        )
        assert int(count) == len(batch[2])


@pytest.mark.slow
def test_train_runs_and_returns_usable_models(tmp_path: Path) -> None:
    """A few epochs on a tiny problem: shapes, history, and a saveable model."""
    n = 512
    rng: np.random.Generator = np.random.default_rng(seed=7)
    z_true: NDArray[np.single] = rng.normal(loc=0.0, scale=1.0, size=(n, 1)).astype(
        dtype=np.single
    )
    z_gen: NDArray[np.single] = rng.normal(loc=0.5, scale=1.0, size=(n, 1)).astype(
        dtype=np.single
    )
    z: NDArray[np.single] = np.concatenate([z_true, z_gen]) + rng.normal(
        loc=0.0, scale=0.5, size=(2 * n, 1)
    ).astype(dtype=np.single)
    x: NDArray[np.single] = np.concatenate([z_true, z_gen]) + rng.normal(
        loc=0.0, scale=0.5, size=(2 * n, 1)
    ).astype(dtype=np.single)
    y: NDArray[np.ubyte] = np.concatenate(
        [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
    )
    splits: DatasetSplits = DeconvolveDataset(batch_size=128, seed=0).splits_from_data(
        data=ZXY(Events(z, x), y)
    )

    g, _d, history, seed, best_epoch, _params, _mmd_test, _sigmas = train(
        splits=splits,
        dim=1,
        n_epochs=3,
        hidden_units=8,
        n_layers=1,
        seed=None,
    )

    assert isinstance(seed, int)
    # The restored epoch has to be one that actually ran, or the metrics get
    # attributed to weights the run never held.
    assert 0 <= best_epoch < 3
    # Five keys: the three scan columns plus the two host-side MMD/ESS curves
    # selection reads. A fourth scan column "val_g" could only duplicate
    # `val_d` --- which is what it used to do, and what put two identical
    # curves on `losses.pdf`.
    assert set(history) == {"train_d", "train_g", "val_d", "val_mmd", "val_ess"}
    # History values are only SupportsFloat, as the rest of the pipeline reads
    # them: `plot_losses` and `_save_run` both name a dtype to convert them.
    curves: dict[str, NDArray[np.single]] = {
        k: np.array(object=v, dtype=np.single) for k, v in history.items()
    }
    assert all(len(v) == 3 for v in curves.values())
    assert all(np.isfinite(v).all() for v in curves.values())

    w: NDArray[np.single] = np.asarray(a=g(z_gen))
    assert w.shape == (n, 1)
    assert np.all(a=w > 0), "softplus output must stay positive"

    # The trained values must have been written back out of the JAX pytree.
    g.save(filepath=tmp_path / "generator.keras")
    reloaded: keras.Model = keras.saving.load_model(
        filepath=tmp_path / "generator.keras"
    )
    np.testing.assert_allclose(
        actual=np.asarray(a=reloaded(z_gen)), desired=w, rtol=1e-6
    )


@pytest.mark.slow
class TestParameterHistory:
    """A run keeps every epoch's weights, which is what moves selection out."""

    @staticmethod
    def _splits(n: int = 512) -> DatasetSplits:
        rng: np.random.Generator = np.random.default_rng(seed=31)
        z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
        x: NDArray[np.single] = z + rng.normal(
            loc=0, scale=0.4, size=(2 * n, 1)
        ).astype(dtype=np.single)
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        )
        return DeconvolveDataset(batch_size=128, seed=0).splits_from_data(
            data=ZXY(Events(z, x), y)
        )

    def test_every_epoch_is_retained(self) -> None:
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=6, hidden_units=8, n_layers=1, seed=3
        )
        assert len(result.history["val_d"]) == 6
        for leaf in result.params.g_trainable:
            assert leaf.shape[0] == 6
        for leaf in result.params.d_trainable:
            assert leaf.shape[0] == 6

    def test_the_restored_model_holds_the_selected_epochs_weights(self) -> None:
        """`g` must carry epoch `best_epoch`'s parameters, not the last one's."""
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=6, hidden_units=8, n_layers=1, seed=4
        )
        for live, stacked in zip(
            result.g.trainable_variables, result.params.g_trainable, strict=True
        ):
            np.testing.assert_allclose(
                actual=np.asarray(a=live.value),
                desired=np.asarray(a=stacked[result.best_epoch]),
                rtol=1e-6,
            )

    def test_params_round_trip_through_disk(self, tmp_path: Path) -> None:
        """Selection under a different criterion is a re-read, not a rerun."""
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=5, hidden_units=8, n_layers=1, seed=5
        )
        _ = save_params(tmp_path, result.params)
        back: EpochParams = load_params(tmp_path)
        assert back._fields == result.params._fields
        for field in result.params._fields:
            before, after = getattr(result.params, field), getattr(back, field)
            assert len(before) == len(after)
            for a, b in zip(before, after, strict=True):
                np.testing.assert_array_equal(
                    actual=np.asarray(a), desired=np.asarray(a=b)
                )

    def test_loaded_params_are_ordered_not_merely_present(self, tmp_path: Path) -> None:
        """The lists are positional: a permuted reload is a wrong model.

        Shapes alone do not catch a permutation -- a Dense layer's kernel and
        bias differ, but two same-width layers' kernels do not -- so this
        checks values against their original index.
        """
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=4, hidden_units=8, n_layers=2, seed=7
        )
        _ = save_params(tmp_path, result.params)
        loaded: Variables = load_params(tmp_path).g_trainable
        assert len(loaded) == len(result.params.g_trainable) > 2
        for i, original in enumerate(iterable=result.params.g_trainable):
            np.testing.assert_array_equal(
                actual=np.asarray(a=loaded[i]), desired=np.asarray(a=original)
            )

    def test_a_run_always_uses_its_full_epoch_budget(self) -> None:
        """`scan` has a fixed trip count: no early stop, so no patience."""
        rng: np.random.Generator = np.random.default_rng(seed=32)
        n: int = 512
        z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
        x: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(
            dtype=np.single
        )  # x carries nothing
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        )
        splits: DatasetSplits = DeconvolveDataset(
            batch_size=128, seed=0
        ).splits_from_data(data=ZXY(Events(z, x), y))
        result: TrainResult = train(
            splits, dim=1, n_epochs=9, hidden_units=8, n_layers=1, seed=6
        )
        assert len(result.history["val_d"]) == 9


@pytest.mark.slow
class TestMMDSelection:
    @staticmethod
    def _splits(n: int = 768) -> DatasetSplits:
        rng: np.random.Generator = np.random.default_rng(seed=41)
        z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
        x: NDArray[np.single] = z + rng.normal(
            loc=0, scale=0.4, size=(2 * n, 1)
        ).astype(dtype=np.single)
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        )
        return DeconvolveDataset(batch_size=128, seed=0).splits_from_data(
            data=ZXY(Events(z, x), y)
        )

    def test_history_carries_the_mmd_and_ess_curves(self) -> None:
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=5, hidden_units=8, n_layers=1, seed=7
        )
        assert set(result.history) == {
            "train_d",
            "train_g",
            "val_d",
            "val_mmd",
            "val_ess",
        }
        assert all(len(v) == 5 for v in result.history.values())
        assert all(np.isfinite(result.history["val_mmd"]))
        assert all(e > 1.0 for e in result.history["val_ess"])

    def test_the_restored_epoch_is_the_mmd_argmin(self) -> None:
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=7, hidden_units=8, n_layers=1, seed=8
        )
        curve: NDArray[np.double] = np.asarray(
            a=result.history["val_mmd"], dtype=np.double
        )
        assert result.best_epoch == int(np.argmin(a=curve))

    def test_selection_no_longer_tracks_the_bce(self) -> None:
        """Not a tautology: it pins that the criterion actually changed.

        If MMD selection silently fell back to the BCE, this passes only by
        coincidence -- so it asserts the two disagree on at least one of
        several seeds rather than on one.
        """
        picks: list[tuple[int, int]] = []
        for seed in (11, 12, 13, 14, 15):
            r: TrainResult = train(
                self._splits(), dim=1, n_epochs=9, hidden_units=8, n_layers=1, seed=seed
            )
            bce = int(np.argmin(np.abs(np.asarray(a=r.history["val_d"]) - LOG2)))
            picks.append((r.best_epoch, bce))
        assert any(m != b for m, b in picks), picks

    def test_the_reported_test_mmd_is_not_the_one_selection_optimized(self) -> None:
        """Selection runs on val; the quoted number comes from test."""
        result: TrainResult = train(
            self._splits(), dim=1, n_epochs=5, hidden_units=8, n_layers=1, seed=9
        )
        assert np.isfinite(result.mmd_test)
        assert result.mmd_test != min(result.history["val_mmd"])


@pytest.mark.slow
def test_training_never_reads_the_truth_rows_of_z() -> None:
    """The MMD subsample must come from `y == 0` rows only.

    `z[y == 1]` is `z_true`. Poisoning it must leave every recorded number
    bit-identical -- the same guarantee `deconvolve leakage-check` makes, asserted
    here at the seam where the MMD subsample is drawn.
    """
    n = 512
    rng: np.random.Generator = np.random.default_rng(seed=42)
    z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
    x: NDArray[np.single] = z + rng.normal(loc=0, scale=0.4, size=(2 * n, 1)).astype(
        dtype=np.single
    )
    y: NDArray[np.ubyte] = np.concatenate(
        [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
    )

    poisoned: NDArray[np.single] = z.copy()
    poisoned[y == 1] = -999.0

    clean_r: TrainResult = train(
        DeconvolveDataset(batch_size=128, seed=0).splits_from_data(
            data=ZXY(Events(z, x), y)
        ),
        dim=1,
        n_epochs=4,
        hidden_units=8,
        n_layers=1,
        seed=17,
    )
    dirty_r: TrainResult = train(
        DeconvolveDataset(batch_size=128, seed=0).splits_from_data(
            data=ZXY(Events(poisoned, x), y)
        ),
        dim=1,
        n_epochs=4,
        hidden_units=8,
        n_layers=1,
        seed=17,
    )
    for key in ("val_d", "val_mmd", "val_ess"):
        np.testing.assert_array_equal(
            actual=clean_r.history[key], desired=dirty_r.history[key]
        )
    assert clean_r.best_epoch == dirty_r.best_epoch


@pytest.mark.slow
class TestSeeding:
    """Weight-init seeding: reproducible runs that still ensemble."""

    @staticmethod
    def _splits(n: int = 384) -> DatasetSplits:
        rng: np.random.Generator = np.random.default_rng(seed=11)
        z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
        x: NDArray[np.single] = z + rng.normal(
            loc=0, scale=0.3, size=(2 * n, 1)
        ).astype(dtype=np.single)
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        )
        return DeconvolveDataset(batch_size=128, seed=3).splits_from_data(
            data=ZXY(Events(z, x), y)
        )

    @staticmethod
    def _run(splits: DatasetSplits, seed: int | None) -> TrainResult:
        return train(
            splits,
            dim=1,
            n_epochs=2,
            hidden_units=8,
            n_layers=1,
            seed=seed,
        )

    def test_same_seed_reproduces_run_exactly(self) -> None:
        splits: DatasetSplits = self._splits()
        probe: NDArray[np.double] = np.linspace(start=-2, stop=2, num=16).reshape(-1, 1)
        a, b = self._run(splits, 123), self._run(splits, 123)
        assert a.seed == b.seed == 123
        np.testing.assert_array_equal(
            actual=np.asarray(a=a.g(probe)), desired=np.asarray(a=b.g(probe))
        )
        np.testing.assert_allclose(
            actual=np.array(object=a.history["train_d"], dtype=np.single),
            desired=np.array(object=b.history["train_d"], dtype=np.single),
            rtol=0,
        )

    def test_different_seeds_give_different_models(self) -> None:
        """The ensemble spread the HEP error estimate relies on."""
        splits: DatasetSplits = self._splits()
        probe: NDArray[np.double] = np.linspace(start=-2, stop=2, num=16).reshape(-1, 1)
        a, b = self._run(splits, 1), self._run(splits, 2)
        assert not np.allclose(a=np.asarray(a=a.g(probe)), b=np.asarray(a=b.g(probe)))

    def test_omitted_seed_is_drawn_and_reported(self) -> None:
        """A run left unseeded must still be reproducible after the fact."""
        splits: DatasetSplits = self._splits()
        drawn: TrainResult = self._run(splits, seed=None)
        assert isinstance(drawn.seed, int)
        assert drawn.seed >= 0

        probe: NDArray[np.double] = np.linspace(start=-2, stop=2, num=16).reshape(-1, 1)
        replay: TrainResult = self._run(splits, drawn.seed)
        np.testing.assert_array_equal(
            actual=np.asarray(a=drawn.g(probe)), desired=np.asarray(a=replay.g(probe))
        )

    def test_omitted_seed_varies_between_runs(self) -> None:
        splits: DatasetSplits = self._splits()
        seeds: set[int] = {self._run(splits, seed=None).seed for _ in range(3)}
        assert len(seeds) == 3, f"entropy-drawn seeds collided: {seeds}"

    def test_init_seed_does_not_disturb_batch_order(self) -> None:
        """The two randomness axes must stay independent.

        Batch order is drawn from the split's own `data_seed`, so changing the
        init seed must not reshuffle the data. This is what makes the HEP
        ensemble -- a loop over `--seed` at fixed `--data_seed` -- measure
        initialization variance and nothing else.
        """
        splits: DatasetSplits = self._splits()
        curves: list[TrainResult] = [
            train(
                splits,
                dim=1,
                n_epochs=2,
                hidden_units=8,
                n_layers=1,
                seed=s,
            )
            for s in (7, 8)
        ]
        # Different inits, so the losses differ -- but both runs consumed the
        # same batches, which is what the shared key guarantees.
        assert curves[0].seed != curves[1].seed
        key: Array = jax.random.key(splits.train.seed)
        np.testing.assert_array_equal(
            actual=np.asarray(
                a=train_indices(
                    key, n=splits.train.size, batch_size=128, n_disc_steps=5
                )
            ),
            desired=np.asarray(
                a=train_indices(
                    key, n=splits.train.size, batch_size=128, n_disc_steps=5
                )
            ),
        )


class TestFusion:
    """The fused whole-run program against the Python-driven reference."""

    @staticmethod
    def _splits(n: int = 512) -> DatasetSplits:
        rng: np.random.Generator = np.random.default_rng(seed=21)
        z: NDArray[np.single] = rng.normal(size=(2 * n, 1)).astype(dtype=np.single)
        x: NDArray[np.single] = z + rng.normal(
            loc=0, scale=0.3, size=(2 * n, 1)
        ).astype(dtype=np.single)
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        )
        return DeconvolveDataset(batch_size=32, seed=5).splits_from_data(
            data=ZXY(Events(z, x), y)
        )

    @pytest.mark.slow
    def test_fused_and_eager_runs_agree(self) -> None:
        """`fused=False` is the debugging path, so it must not be a second model.

        It runs the identical epoch function from a Python `while` instead of a
        `lax.while_loop`, so the two differ only in how XLA is allowed to
        associate the reductions -- not in the arithmetic.
        """
        splits: DatasetSplits = self._splits()

        def run_train(*, fused: bool) -> TrainResult:
            return train(
                splits,
                dim=1,
                hidden_units=8,
                n_layers=1,
                seed=42,
                n_epochs=3,
                fused=fused,
            )

        fused: TrainResult = run_train(fused=True)
        eager: TrainResult = run_train(fused=False)
        for key in ("train_d", "train_g", "val_d"):
            np.testing.assert_allclose(
                actual=fused.history[key], desired=eager.history[key], rtol=1e-10
            )
        # Selection reads the same curves, so it must land on the same epoch.
        assert fused.best_epoch == eager.best_epoch

    def test_padded_eval_matches_an_exactly_divisible_one(self) -> None:
        """Padding rows must be inert.

        The eval splits are padded to a whole number of batches and the filler
        carries `mask == 0`; it enters no sum, so the reported loss is the one an
        unpadded split would have given.
        """
        splits: DatasetSplits = self._splits()
        n_val: int = splits.val.size
        assert n_val % 7 != 0, "want a size that does not divide evenly"
        exact: DeviceSplits = DeviceSplits.from_splits(splits, eval_batch_size=n_val)
        padded: DeviceSplits = DeviceSplits.from_splits(splits, eval_batch_size=7)
        assert padded.val.n_batches > exact.val.n_batches
        assert int(padded.val.mask.sum()) == n_val
        assert int(exact.val.mask.sum()) == n_val


@pytest.mark.usefixtures("restored")
class TestCompilationCache:
    """XLA compile is the largest single term in a short run.

    `benchmarks/boundary.py` on an A100 measures 4.60s of compile against 0.034s
    per epoch, so a 100-epoch run spends half its wall clock in the compiler.
    The cache is keyed on lowered HLO and lives on disk, so it survives across
    processes -- which is where it pays, since an ensemble is N interpreters
    compiling one architecture N times.

    Two things here are easy to get wrong silently, so both are pinned.
    """

    @pytest.fixture
    def restored(self) -> Generator[None]:
        """`jax.config` is process-global; put it back however the test left it."""
        prior: tuple[object, object] = (
            jax.config.jax_compilation_cache_dir,
            jax.config.jax_persistent_cache_min_compile_time_secs,
        )
        jax.config.update(name="jax_compilation_cache_dir", val=None)
        yield
        jax.config.update(name="jax_compilation_cache_dir", val=prior[0])
        jax.config.update(
            name="jax_persistent_cache_min_compile_time_secs", val=prior[1]
        )

    def test_the_threshold_drops_to_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """JAX's 1.0s default leaves RAN's cache *empty*, not merely sparse.

        A run compiles a few dozen executables totalling ~4.6s and not one of
        them clears a second on its own, so the stock threshold caches nothing
        and reports nothing about having done so.
        """
        monkeypatch.delenv(
            name="JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", raising=False
        )

        _use_compilation_cache()

        assert jax.config.jax_compilation_cache_dir == str(
            object=COMPILE_CACHE_DIR.resolve()
        )
        threshold: float = jax.config.jax_persistent_cache_min_compile_time_secs
        assert threshold == pytest.approx(expected=0.0)

    def test_a_caller_who_already_chose_a_directory_keeps_it(
        self, tmp_path: Path
    ) -> None:
        """`JAX_COMPILATION_CACHE_DIR` is JAX's own knob and predates this one."""
        jax.config.update(name="jax_compilation_cache_dir", val=str(object=tmp_path))

        _use_compilation_cache()

        assert jax.config.jax_compilation_cache_dir == str(object=tmp_path)

    def test_a_caller_who_chose_a_threshold_keeps_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Someone who set the threshold high did so to keep a shared cache
        small; adopting our directory should not also override that."""
        monkeypatch.setenv(
            name="JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", value="2.0"
        )
        jax.config.update(name="jax_persistent_cache_min_compile_time_secs", val=2.0)

        _use_compilation_cache()

        assert jax.config.jax_compilation_cache_dir == str(
            object=COMPILE_CACHE_DIR.resolve()
        )
        threshold: float = jax.config.jax_persistent_cache_min_compile_time_secs
        assert threshold == pytest.approx(expected=2.0)


class TestTrainingNeverSeesTheTestSplit:
    """The split boundaries are disjoint slices, but that is a property of
    `_split_dataset`; this pins the property callers actually care about.

    `train` does transfer the test split to device --- `DeviceSplits.from_splits`
    moves all three at once --- but this task never reads it: the final
    test-split BCE read went with in-loop selection, and Task 3 will
    reintroduce a different read, a test-split MMD. The claim worth testing is
    therefore not "test is absent" but "test cannot influence the result":
    corrupt it beyond recognition and every returned model weight and history
    value must be bit-identical.
    """

    @staticmethod
    def _splits(n: int = 600) -> DatasetSplits:
        rng: np.random.Generator = np.random.default_rng(seed=0)
        z: NDArray[np.single] = rng.normal(size=(2 * n, 2)).astype(dtype=np.single)
        x: NDArray[np.single] = (z + 0.3 * rng.normal(size=z.shape)).astype(
            dtype=np.single
        )
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        ).astype(dtype=np.ubyte)
        return DeconvolveDataset(batch_size=64, seed=1).splits_from_data(
            data=ZXY(Events(z, x), y)
        )

    @staticmethod
    def _poison(splits: DatasetSplits, which: str) -> DatasetSplits:
        """Replace every feature of one split with a value no real event has.

        `as_arrays()` hands back a view, not a copy, so this reaches whatever
        `train` goes on to read --- which is the whole point, and what
        `test_the_poison_reaches_training_at_all` exists to prove.
        """
        arrays: ZXY = getattr(splits, which).as_arrays()
        arrays.z[:] = -1234.0
        arrays.x[:] = 4321.0
        return splits

    @staticmethod
    def _variables(result: TrainResult) -> list[NDArray[Any]]:
        """Every array `train` writes back, generator and discriminator alike.

        `trainable_variables` rather than `get_weights()`: it is what the
        `DeconvolveModel` protocol declares, and it is precisely what `_assign` restores
        the best state into.
        """
        return [
            np.asarray(a=v)
            for model in (result.g, result.d)
            for v in (*model.trainable_variables, *model.non_trainable_variables)
        ]

    @staticmethod
    def _run(splits: DatasetSplits) -> TrainResult:
        return train(splits, dim=2, hidden_units=8, n_layers=1, seed=5, n_epochs=4)

    def test_the_splits_do_not_overlap(self) -> None:
        """The precondition, checked directly rather than assumed."""
        splits: DatasetSplits = self._splits()
        sizes: tuple[int, int, int] = (
            splits.train.size,
            splits.val.size,
            splits.test.size,
        )

        assert sum(sizes) == len(splits.select(Split.ALL))
        assert min(sizes) > 0

    @pytest.mark.slow
    def test_corrupting_test_changes_no_weight_and_no_history_value(self) -> None:
        clean: TrainResult = self._run(self._splits())
        poisoned: TrainResult = self._run(
            splits=self._poison(self._splits(), which="test")
        )

        for a, b in zip(
            self._variables(result=clean), self._variables(result=poisoned), strict=True
        ):
            np.testing.assert_array_equal(actual=a, desired=b)

        assert clean.history.keys() == poisoned.history.keys()
        for key in clean.history:
            np.testing.assert_array_equal(
                actual=clean.history[key], desired=poisoned.history[key]
            )

    @pytest.mark.slow
    def test_the_poison_reaches_training_at_all(self) -> None:
        """The negative control, without which the test above is vacuous.

        If `_poison` mutated a copy, corrupting *any* split would look clean and
        the leakage test would pass for the wrong reason. Val is the tell: it is
        read every epoch for early stopping, so garbage there has to move the
        `val_d` column.
        """
        clean: TrainResult = self._run(self._splits())
        poisoned: TrainResult = self._run(
            splits=self._poison(self._splits(), which="val")
        )

        assert not np.array_equal(
            a1=clean.history["val_d"], a2=poisoned.history["val_d"]
        )


class TestWeightDispersion:
    """How far the generator has travelled from `w = 1`, as one number.

    `benchmarks/README.md` §2 measures the oracle's ESS at 80.1% against RAN's
    73.3%: RAN's weights are *more* dispersed than the truth's, so dispersion is
    a knob with a target rather than a free parameter. For weights normalised to
    mean 1 the relation is `ESS/n = 1 / (1 + Var(w))`, which puts the oracle at
    Var 0.249 and RAN at 0.364.

    The variance is taken over the MC rows only. Nature's weights are pinned to
    1 by `normalize_weights` and carry no gradient, so including them would
    dilute the penalty by the class balance rather than measure anything.
    """

    def test_uniform_weights_have_no_dispersion(self) -> None:
        y: NDArray[np.single] = np.zeros(shape=4, dtype=np.single)
        w: NDArray[np.single] = np.ones(shape=4, dtype=np.single)

        assert float(
            weight_dispersion(w, y, np.ones(shape=4, dtype=np.single))
        ) == pytest.approx(expected=0.0)

    def test_matches_the_variance_of_the_mc_weights(self) -> None:
        y: NDArray[np.single] = np.zeros(shape=4, dtype=np.single)
        w: NDArray[np.single] = np.asarray(a=[0.5, 1.5, 0.5, 1.5], dtype=np.single)

        assert float(
            weight_dispersion(w, y, np.ones(shape=4, dtype=np.single))
        ) == pytest.approx(expected=0.25)

    def test_ignores_the_nature_rows(self) -> None:
        """Nature's weights are fixed at 1 and must not dilute the measure."""
        y: NDArray[np.single] = np.asarray(a=[1.0, 1.0, 0.0, 0.0], dtype=np.single)
        w: NDArray[np.single] = np.asarray(a=[1.0, 1.0, 0.5, 1.5], dtype=np.single)

        assert float(
            weight_dispersion(w, y, np.ones(shape=4, dtype=np.single))
        ) == pytest.approx(expected=0.25)

    def test_ignores_padded_rows(self) -> None:
        """A padded batch must report what an unpadded one would."""
        y: NDArray[np.single] = np.zeros(shape=6, dtype=np.single)
        w: NDArray[np.single] = np.asarray(
            a=[0.5, 1.5, 0.5, 1.5, 0.0, 0.0], dtype=np.single
        )
        mask: NDArray[np.single] = np.asarray(
            a=[1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.single
        )

        assert float(weight_dispersion(w, y, mask)) == pytest.approx(expected=0.25)

    def test_recovers_the_effective_sample_size(self) -> None:
        """The identity that makes the oracle's 80.1% a target for a coefficient."""
        rng: np.random.Generator = np.random.default_rng(seed=0)
        raw: NDArray[np.single] = rng.lognormal(sigma=0.5, size=4096).astype(
            dtype=np.single
        )
        y: NDArray[np.single] = np.zeros(shape=raw.size, dtype=np.single)
        w: Array = normalize_weights(raw, y, np.ones(shape=raw.size, dtype=np.single))

        ess_fraction = float(np.sum(a=w) ** 2 / (raw.size * np.sum(a=np.square(w))))

        assert ess_fraction == pytest.approx(
            expected=1.0
            / (
                1.0
                + float(
                    weight_dispersion(w, y, np.ones(shape=raw.size, dtype=np.single))
                )
            ),
            rel=1e-5,
        )


@pytest.mark.slow
class TestDispersionPenalty:
    """The penalty enters `g`'s gradient without entering what is recorded.

    `CLAUDE.md`'s Training Loop section documents the history's three columns as
    all being the same weighted BCE on the same scale. Adding the penalty to the
    number `g` reports would silently break that -- `losses.pdf` would compare a
    penalised curve against two unpenalised ones -- so the penalty is applied to
    the gradient and the BCE is what comes back out.
    """

    @staticmethod
    def _splits(n: int = 1024) -> DatasetSplits:
        """Nature shifted from MC, so `g` has something to reweight toward."""
        rng: np.random.Generator = np.random.default_rng(seed=17)
        z: NDArray[np.single] = np.concatenate(
            [
                rng.normal(loc=0.6, scale=1.0, size=(n, 1)),
                rng.normal(loc=0.0, scale=1.0, size=(n, 1)),
            ]
        ).astype(dtype=np.single)
        x: NDArray[np.single] = z + rng.normal(
            loc=0.0, scale=0.4, size=(2 * n, 1)
        ).astype(dtype=np.single)
        y: NDArray[np.ubyte] = np.concatenate(
            [np.ones(shape=n, dtype=np.ubyte), np.zeros(shape=n, dtype=np.ubyte)]
        ).astype(dtype=np.ubyte)
        return DeconvolveDataset(batch_size=128, seed=0).splits_from_data(
            data=ZXY(Events(z, x), y)
        )

    def _run(self, lambda_dispersion: float, epochs: int = 40) -> TrainResult:
        """`lr_g` well above the default: the weights have to actually spread.

        At the shipped 3e-5 over a handful of epochs `g` barely leaves `w = 1`,
        the ESS sits at 96.7% in both arms and the test cannot tell a working
        penalty from a no-op.
        """
        return train(
            self._splits(),
            dim=1,
            hidden_units=8,
            n_layers=1,
            seed=0,
            n_epochs=epochs,
            lr_g=1e-2,
            lambda_dispersion=lambda_dispersion,
        )

    def test_zero_coefficient_turns_the_penalty_off(self) -> None:
        """`--lambda-dispersion 0` recovers the pure adversarial objective.

        This used to assert that an explicit 0 matched the *default*, which
        held only while the default was itself 0. It is now 0.015, so the
        invariant worth pinning is that 0 still reaches the unpenalised
        behaviour -- that is what reproduces a run from before the penalty
        existed.
        """
        first: TrainResult = self._run(lambda_dispersion=0.0)
        second: TrainResult = self._run(lambda_dispersion=0.0)
        penalised: TrainResult = self._run(lambda_dispersion=0.015)

        assert first.history["train_g"] == pytest.approx(second.history["train_g"])
        assert first.history["train_g"] != pytest.approx(penalised.history["train_g"])

    def test_the_coefficient_reduces_the_weight_dispersion(self) -> None:
        """The behavioural claim: a bigger coefficient means tamer weights."""
        loose: TrainResult = self._run(lambda_dispersion=0.0)
        tight: TrainResult = self._run(lambda_dispersion=100.0)

        assert min(tight.history["val_ess"]) > min(loose.history["val_ess"])

    def test_the_recorded_loss_stays_on_the_bce_scale(self) -> None:
        """`train_g` must not absorb the penalty; the history is one scale."""
        penalised: TrainResult = self._run(lambda_dispersion=100.0)

        # A weighted BCE sits near log 2 while d is confused, and is bounded
        # below by 0. A penalty folded into this number would push it far above.
        assert all(0.0 < value < 2.0 for value in penalised.history["train_g"])
