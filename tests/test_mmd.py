"""Tests for the weighted MMD estimator.

MMD is a divergence -- zero iff the distributions match, monotone in
mismatch -- which is the property the validation BCE does not have and the
reason selection is built on this instead.
"""

from itertools import pairwise
from typing import TYPE_CHECKING, Literal

import jax.numpy as jnp
import numpy as np
from deconvolve.training.mmd import (
    MMDCache,
    bandwidths,
    build_cache,
    median_bandwidth,
    mmd_curve,
    squared_distances,
    subsample_indices,
    weighted_mmd,
)
from jax._src.basearray import Array  # ruff: ignore[typing-only-third-party-import]

if TYPE_CHECKING:
    from numpy.random._generator import Generator


def _samples(n: int = 1024, d: int = 6, shift: float = 0.0, seed: int = 0) -> Array:
    rng: Generator = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(loc=shift, scale=1.0, size=(n, d)), dtype=jnp.float32)


def _cache(x: Array, y: Array) -> MMDCache:
    return build_cache(x, y, sigmas=bandwidths(x))


class TestSquaredDistances:
    def test_matches_the_broadcast_form(self) -> None:
        """The expansion is an optimization, so it must be exact.

        `sum((a[:,None,:] - b[None,:,:])**2, -1)` is the obvious form and
        materializes an (n, m, d) intermediate -- 6.4 GB at m=16384. The
        expansion below is (n, m) only, and this pins that the saving is free.
        """
        a, b = _samples(n=64, d=3, seed=1), _samples(n=48, d=3, shift=0.7, seed=2)
        naive: Array = jnp.sum(a=(a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        np.testing.assert_allclose(
            actual=np.asarray(squared_distances(a, b)),
            desired=np.asarray(naive),
            atol=1e-4,
        )

    def test_self_distance_is_zero_on_the_diagonal(self) -> None:
        a: Array = _samples(n=32, d=4, seed=3)
        np.testing.assert_allclose(
            actual=np.diag(np.asarray(a=squared_distances(a, a))),
            desired=0.0,
            atol=1e-4,
        )


class TestBandwidths:
    def test_median_heuristic_puts_the_kernel_at_exp_minus_one(self) -> None:
        """sigma = sqrt(median/2) makes k(median distance) = exp(-1)."""
        x: Array = _samples(n=512, d=6, seed=4)
        sigma: float = median_bandwidth(x)
        med = float(jnp.median(a=squared_distances(x, x)))
        np.testing.assert_allclose(
            actual=float(jnp.exp(-med / (2 * sigma**2))),
            desired=np.exp(-1.0),
            rtol=1e-5,
        )

    def test_five_scales_bracket_the_median(self) -> None:
        x: Array = _samples(n=256, d=6, seed=5)
        sig: tuple[float, ...] = bandwidths(x)
        assert len(sig) == 5
        assert sig[0] < sig[2] < sig[4]
        np.testing.assert_allclose(sig[2], median_bandwidth(x), rtol=1e-6)


class TestWeightedMMD:
    def test_identical_distributions_score_near_zero(self) -> None:
        x, y = _samples(n=1024, d=6, seed=6), _samples(n=1024, d=6, seed=7)
        mmd, _ = weighted_mmd(_cache(x, y), jnp.ones(1024))
        assert abs(float(mmd)) < 5e-3

    def test_a_shifted_distribution_scores_far_higher(self) -> None:
        x, y = _samples(n=1024, d=6, seed=8), _samples(n=1024, d=6, shift=0.5, seed=9)
        near, _ = weighted_mmd(_cache(x, _samples(1024, 6, seed=10)), jnp.ones(1024))
        far, _ = weighted_mmd(_cache(x, y), jnp.ones(1024))
        assert float(far) > 10 * abs(float(near))

    def test_exact_importance_weights_undo_a_shift(self) -> None:
        """The known answer: N(0,1)/N(0.5,1) weights must recover the match."""
        x, y = _samples(n=2048, d=6, seed=11), _samples(n=2048, d=6, shift=0.5, seed=12)
        cache = _cache(x, y)
        unweighted, _ = weighted_mmd(cache, jnp.ones(2048))
        logw: Array = -0.5 * jnp.sum(a=y**2, axis=1) + 0.5 * jnp.sum(
            a=(y - 0.5) ** 2, axis=1
        )
        weighted, _ = weighted_mmd(cache, jnp.exp(logw - logw.max()))
        assert float(weighted) < 0.2 * float(unweighted)

    def test_matches_a_fully_materialized_reference(self) -> None:
        """The precomputation must be an optimization and nothing more.

        The reference below stores `k_xy` and forms `w[:,None]*w[None,:]*k_yy`
        outright -- 3.22 GB and 30 ms/eval at m=16384 against 1.07 GB and
        2.8 ms. If the two ever disagree, the collapse of `k_xy` to a vector
        is wrong, not merely slower.
        """
        x, y = _samples(n=512, d=6, seed=28), _samples(n=512, d=6, shift=0.4, seed=29)
        sig: tuple[float, ...] = bandwidths(x)
        rng: Generator = np.random.default_rng(30)
        w: Array = jnp.asarray(
            rng.uniform(low=0.1, high=3.0, size=512), dtype=jnp.float32
        )

        def _k(a: Array, b: Array) -> Literal[0] | Array:
            d2: Array = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
            return sum(jnp.exp(-d2 / (2 * s**2)) for s in sig)

        k_xx, k_yy, k_xy = _k(x, x), _k(y, y), _k(x, y)
        q: Array = w / jnp.sum(w)
        sq: Array = jnp.sum(q**2)
        term_xx: Array = (jnp.sum(k_xx) - jnp.trace(k_xx)) / (512 * 511)
        term_yy: Array = (
            jnp.sum(a=q[:, None] * q[None, :] * k_yy)
            - jnp.sum(a=q**2 * jnp.diag(v=k_yy))
        ) / (1.0 - sq)
        reference: Array = term_xx + term_yy - 2.0 * jnp.sum(a=q[None, :] * k_xy) / 512

        got, _ = weighted_mmd(build_cache(x, y, sigmas=sig), w)
        np.testing.assert_allclose(float(got), float(reference), rtol=1e-5)

    def test_rescaling_the_raw_weights_changes_nothing(self) -> None:
        """Only normalized weights enter, so callers need not pre-normalize."""
        x, y = _samples(512, 6, seed=13), _samples(512, 6, shift=0.3, seed=14)
        cache: MMDCache = _cache(x, y)
        rng: Generator = np.random.default_rng(seed=15)
        w: Array = jnp.asarray(
            rng.uniform(low=0.1, high=3.0, size=512), dtype=jnp.float32
        )
        a, _ = weighted_mmd(cache, w)
        b, _ = weighted_mmd(cache, w * 1000.0)
        np.testing.assert_allclose(actual=float(a), desired=float(b), rtol=1e-5)

    def test_ess_is_the_inverse_sum_of_squared_normalized_weights(self) -> None:
        x, y = _samples(256, 6, seed=16), _samples(256, 6, seed=17)
        cache: MMDCache = _cache(x, y)
        _, flat = weighted_mmd(cache, jnp.ones(256))
        np.testing.assert_allclose(float(flat), 256.0, rtol=1e-4)
        w: Array = jnp.asarray(
            a=np.concatenate([np.ones(shape=1), np.full(shape=255, fill_value=1e-6)]),
            dtype=jnp.float32,
        )
        _, spiked = weighted_mmd(cache, w)
        assert float(spiked) < 2.0

    def test_collapsed_weights_return_infinity_not_a_blow_up(self) -> None:
        """As ESS -> 1 the unbiased estimator is undefined: one effective
        sample cannot estimate E[k(y,y')]. Selection must never pick it."""
        x, y = _samples(256, 6, seed=18), _samples(256, 6, seed=19)
        w = np.zeros(256, dtype=np.float32)
        w[0] = 1.0
        mmd, ess = weighted_mmd(_cache(x, y), jnp.asarray(w))
        assert np.isinf(float(mmd))
        assert not np.isnan(float(mmd))
        np.testing.assert_allclose(float(ess), 1.0, rtol=1e-4)

    def test_float32_matches_a_float64_reference(self) -> None:
        """MMD^2 is a small difference of O(1) terms, so cancellation is the
        worry. XLA's blocked reduction absorbs it; this pins that."""
        x32, y32 = _samples(1024, 6, seed=20), _samples(1024, 6, shift=0.4, seed=21)
        rng = np.random.default_rng(22)
        w = rng.uniform(0.1, 3.0, 1024)
        sig = bandwidths(x32)
        a, _ = weighted_mmd(
            build_cache(x32, y32, sigmas=sig), jnp.asarray(w, jnp.float32)
        )
        x64 = np.asarray(x32, dtype=np.float64)
        y64 = np.asarray(y32, dtype=np.float64)
        # Passing float64 is the whole point of this test; the signature is
        # pinned to float32.
        b, _ = weighted_mmd(
            build_cache(x64, y64, sigmas=sig),  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
            jnp.asarray(w),
        )
        np.testing.assert_allclose(float(a), float(b), atol=1e-6)

    def test_monotone_along_the_path_to_the_exact_weights(self) -> None:
        """Outside the resolution floor, MMD must fall as the weights improve.

        The floor is real: below ~5e-4 in MMD^2 at m=8192 the ranking inverts,
        because the empirical MMD is minimized by weights matching the sample
        rather than the distribution. `t` stops at 0.9 to stay above it.
        """
        x, y = _samples(2048, 6, seed=23), _samples(2048, 6, shift=0.5, seed=24)
        cache = _cache(x, y)
        logw = -0.5 * jnp.sum(y**2, 1) + 0.5 * jnp.sum((y - 0.5) ** 2, 1)
        star = jnp.exp(logw - logw.max())
        star = star / jnp.sum(star)
        flat = jnp.full(2048, 1.0 / 2048)
        vals = [
            float(weighted_mmd(cache, (1 - t) * flat + t * star)[0])
            for t in np.linspace(0, 0.9, 10)
        ]
        assert all(b < a for a, b in pairwise(vals)), vals


class TestCurve:
    def test_curve_matches_evaluating_each_row(self) -> None:
        x, y = _samples(512, 6, seed=25), _samples(512, 6, shift=0.3, seed=26)
        cache = _cache(x, y)
        rng = np.random.default_rng(27)
        w = jnp.asarray(rng.uniform(0.1, 3.0, (7, 512)), dtype=jnp.float32)
        mmds, esss = mmd_curve(cache, w)
        assert mmds.shape == (7,)
        assert esss.shape == (7,)
        # `rtol` is 1e-4 rather than 1e-5 for the reason `TestFusedMetrics`
        # gives: the batched path and the single-row path reduce the same
        # 512x512 float32 kernel matrix in different orders, and sharing one
        # reduction between them would delete the question being asked. The
        # gap was measured over 70 rows at 3.4e-5 (jax 0.7) and 1.3e-5 (jax
        # 0.11) -- so 1e-5 was already under the noise on the floor that
        # shipped, and passed only on the seeds this test happens to use.
        for i in range(7):
            one_mmd, one_ess = weighted_mmd(cache, w[i])
            np.testing.assert_allclose(mmds[i], float(one_mmd), rtol=1e-4)
            np.testing.assert_allclose(esss[i], float(one_ess), rtol=1e-4)


class TestSubsample:
    def test_is_reproducible_and_within_range(self) -> None:
        a = subsample_indices(42, 1000, 100)
        b = subsample_indices(42, 1000, 100)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (100,)
        assert len(np.unique(a)) == 100
        assert a.min() >= 0
        assert a.max() < 1000

    def test_a_different_seed_draws_differently(self) -> None:
        assert not np.array_equal(
            subsample_indices(1, 1000, 100), subsample_indices(2, 1000, 100)
        )

    def test_asking_for_more_than_exists_takes_everything(self) -> None:
        idx = subsample_indices(0, 50, 4096)
        assert idx.shape == (50,)
        assert len(np.unique(idx)) == 50
