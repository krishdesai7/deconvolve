"""Tests for the device-resident training form.

These cover the batching contract that `ArrayDataset.__iter__` used to hold:
where batch order comes from, what the epoch drops, and why padded evaluation
rows are inert.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from deconvolve.coretypes import ZXY, DatasetSplits, Events
from deconvolve.data import DeconvolveDataset
from deconvolve.data.device import (
    DEFAULT_EVAL_BATCH_SIZE,
    DeviceSplits,
    EvalSplit,
    TrainSplit,
    gather,
    grouping,
    train_indices,
)


def _toy(n: int = 200, batch_size: int = 32, seed: int = 4) -> DatasetSplits:
    """z counts up, x is its negation, so pairing is checkable by eye."""
    z = np.arange(2 * n, dtype=np.single).reshape(-1, 1)
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    return DeconvolveDataset(batch_size=batch_size, seed=seed).splits_from_data(
        ZXY(Events(z, -z), y)
    )


class TestTrainSplit:
    def test_labels_are_promoted_to_the_compute_dtype(self) -> None:
        """`y` is ubyte on host and float on device, converted once."""
        host = _toy().train.as_arrays()
        assert host.y.dtype == np.ubyte
        split = TrainSplit.from_zxy(host)
        assert split.z.dtype == split.y.dtype == np.single
        np.testing.assert_array_equal(np.asarray(split.y), host.y.astype(np.single))

    def test_is_a_pytree(self) -> None:
        """The splits have to survive `tree_map` to cross a jit boundary."""
        split = TrainSplit.from_zxy(_toy().train.as_arrays())
        doubled = jax.tree.map(lambda a: a * 2, split)
        assert isinstance(doubled, TrainSplit)
        np.testing.assert_allclose(np.asarray(doubled.z), np.asarray(split.z) * 2)

    def test_gather_keeps_rows_aligned(self) -> None:
        split = TrainSplit.from_zxy(_toy().train.as_arrays())
        idx = jnp.asarray([3, 0, 7])
        z, x, y = gather(split, idx)
        np.testing.assert_array_equal(np.asarray(x), -np.asarray(z))
        np.testing.assert_array_equal(np.asarray(y), np.asarray(split.y)[[3, 0, 7]])


class TestGrouping:
    def test_groups_tile_the_whole_batches(self) -> None:
        # 10 whole batches, 5 disc steps each -> 2 generator updates.
        assert grouping(n=330, batch_size=32, n_disc_steps=5) == (2, 5)

    def test_disc_steps_clamp_to_the_batches_available(self) -> None:
        """A split too small for one group still trains, as one short group.

        This is what the host loop did when `step % n_disc_steps == 0` fired
        only at step 0 because the pass deconvolve out of batches first.
        """
        assert grouping(n=100, batch_size=32, n_disc_steps=5) == (1, 3)

    def test_a_split_below_one_batch_is_refused(self) -> None:
        with pytest.raises(ValueError, match="do not fill a single batch"):
            _ = grouping(n=10, batch_size=32, n_disc_steps=5)


class TestTrainIndices:
    def test_shape_and_no_repeats_within_an_epoch(self) -> None:
        idx = train_indices(jax.random.key(0), 330, 32, 5)
        assert idx.shape == (2, 5, 32)
        flat = np.asarray(idx).ravel()
        assert len(set(flat.tolist())) == flat.size

    def test_the_dropped_tail_is_what_does_not_fill_a_group(self) -> None:
        # 330 events at batch 32 give 10 whole batches (320 events) plus 10
        # left over; 10 batches make exactly 2 groups of 5, so only those 10
        # events are dropped -- and a different 10 each epoch, since the
        # permutation is redrawn.
        idx = train_indices(jax.random.key(0), 330, 32, 5)
        assert idx.size == 320
        assert not np.array_equal(
            np.sort(np.asarray(idx).ravel()),
            np.sort(np.asarray(train_indices(jax.random.key(1), 330, 32, 5)).ravel()),
        )

    def test_order_depends_only_on_the_key(self) -> None:
        """A pure function of the key, so nothing can advance it out from under
        a caller -- the property the old `reset()` existed to protect."""
        key = jax.random.key(3)
        a = train_indices(key, 330, 32, 5)
        _ = train_indices(jax.random.key(99), 330, 32, 5)  # unrelated draw
        np.testing.assert_array_equal(
            np.asarray(a), np.asarray(train_indices(key, 330, 32, 5))
        )

    def test_successive_epoch_keys_reshuffle(self) -> None:
        k0, k1 = jax.random.split(jax.random.key(3))
        assert not np.array_equal(
            np.asarray(train_indices(k0, 330, 32, 5)),
            np.asarray(train_indices(k1, 330, 32, 5)),
        )


class TestEvalSplit:
    def test_padding_is_masked_out(self) -> None:
        host = _toy(n=100).val.as_arrays()
        split = EvalSplit.from_zxy(host, batch_size=7)
        assert split.z.shape[:2] == (-(-len(host) // 7), 7)
        assert int(split.mask.sum()) == len(host)
        # The pad sits at the very end, so the real rows come back in order.
        flat = np.asarray(split.z).reshape(-1, 1)
        keep = np.asarray(split.mask).ravel() > 0.5
        np.testing.assert_array_equal(flat[keep], host.z)

    def test_padding_rows_carry_finite_values(self) -> None:
        """`mask` keeps them out of every sum, but a NaN would still spread
        through a gradient, so the filler repeats a real row."""
        split = EvalSplit.from_zxy(_toy(n=100).val.as_arrays(), batch_size=7)
        assert np.all(np.isfinite(np.asarray(split.z)))
        assert np.all(np.isfinite(np.asarray(split.y)))

    def test_an_exactly_divisible_split_gets_no_padding(self) -> None:
        host = _toy(n=100).val.as_arrays()
        split = EvalSplit.from_zxy(host, batch_size=len(host))
        assert split.n_batches == 1
        np.testing.assert_array_equal(np.asarray(split.mask), np.ones((1, len(host))))


class TestDeviceSplits:
    def test_carries_the_data_seed_not_the_init_seed(self) -> None:
        """Batch order must follow `data_seed`, so an ensemble over init seeds
        holds the data fixed."""
        splits = _toy(seed=17)
        assert DeviceSplits.from_splits(splits).data_seed == 17

    def test_eval_batch_size_defaults_wider_than_training(self) -> None:
        splits = _toy(n=100, batch_size=32)
        device = DeviceSplits.from_splits(splits)
        assert DEFAULT_EVAL_BATCH_SIZE > 32
        # Both eval splits fit in one batch at the default.
        assert device.val.n_batches == device.test.n_batches == 1

    def test_every_split_reaches_the_device(self) -> None:
        device = DeviceSplits.from_splits(_toy())
        assert isinstance(device.train, TrainSplit)
        assert isinstance(device.val, EvalSplit)
        assert isinstance(device.test, EvalSplit)


class TestDtypeIsPinnedAtTheBoundary:
    """The host->device seam is where `EVENT_DTYPE` stops being an annotation.

    Nothing coerces at the `Populations` boundary, so a caller that builds one
    out of raw float64 -- `benchmarks/boundary.py` did exactly this -- hands
    float64 all the way down to here. JAX truncates it either way; what these
    pin is that the truncation is deliberate and silent rather than incidental
    and warned about.
    """

    @staticmethod
    def _wide(n: int = 200, batch_size: int = 32) -> DatasetSplits:
        """`_toy`, but float64 on host: the shape of the leak.

        Both checkers reject the `Events` call below, which is the contract
        working as designed -- and is how this would have been caught in `src`.
        `benchmarks/**` is excluded from both, which is exactly why
        `benchmarks/boundary.py` got to make this mistake unchallenged.
        """
        z = np.arange(2 * n, dtype=np.double).reshape(-1, 1)
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return DeconvolveDataset(batch_size=batch_size, seed=4).splits_from_data(
            # pyrefly: ignore[bad-argument-type]
            # ty: ignore[invalid-argument-type]
            ZXY(Events(z, -z), y)
        )

    def test_float64_host_arrays_narrow_without_warning(self) -> None:
        splits = self._wide()
        assert splits.train.as_arrays().z.dtype == np.double

        with warnings.catch_warnings():
            # A dtype JAX cannot honour is a UserWarning, not an error; the
            # point of asking for EVENT_DTYPE is that it never has to refuse.
            warnings.simplefilter("error", UserWarning)
            device = DeviceSplits.from_splits(splits)

        assert device.train.z.dtype == np.single
        assert device.train.x.dtype == np.single
        assert device.train.y.dtype == np.single

    def test_every_eval_field_including_the_mask_is_pinned(self) -> None:
        """`mask` multiplies into every sum, so a wider mask would silently
        promote the reductions it guards."""
        device = DeviceSplits.from_splits(self._wide(n=100), eval_batch_size=7)
        for split in (device.val, device.test):
            assert split.z.dtype == np.single
            assert split.x.dtype == np.single
            assert split.y.dtype == np.single
            assert split.mask.dtype == np.single
