"""Tests for the sliced Wasserstein distance.

Every metric in `deconvolve.evaluation.evaluate` is computed per coordinate
axis, so all of them are blind to joint structure by construction: two
distributions with identical marginals and different correlations score
identically. The sliced Wasserstein distance projects onto random directions
instead of axes, which is exactly the gap.
`test_sees_correlation_that_the_axis_metrics_miss` is the whole point of the
module and the rest is there to make sure the number is right.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wasserstein_distance

from benchmarks.sliced import null_floors, sliced_wasserstein, w1_weighted


class TestWeightedW1:
    """RAN emits weights, not resampled events, so W1 has to be the weighted one.

    The textbook `mean(|sort(x) - sort(y)|)` is the equal-size, uniform-weight
    special case. Using it here would mean either discarding the weights --
    which measures the wrong distribution -- or resampling events to represent
    them, which injects sampling noise into the quantity being measured.
    """

    def test_matches_scipy_with_weights(self) -> None:
        rng = np.random.default_rng(0)
        u = rng.normal(size=400)
        v = rng.normal(loc=0.3, size=600)
        wv = rng.gamma(2.0, size=600)

        reference: float = wasserstein_distance(u, v, v_weights=wv)
        assert w1_weighted(u, v, v_weights=wv) == pytest.approx(reference, rel=1e-9)

    def test_matches_scipy_without_weights(self) -> None:
        rng = np.random.default_rng(1)
        u = rng.normal(size=300)
        v = rng.normal(loc=-0.5, scale=1.4, size=300)

        reference: float = wasserstein_distance(u, v)
        assert w1_weighted(u, v) == pytest.approx(reference, rel=1e-9)

    def test_is_zero_for_a_sample_against_itself(self) -> None:
        x = np.random.default_rng(2).normal(size=256)

        assert w1_weighted(x, x) == pytest.approx(0.0, abs=1e-12)

    def test_weights_are_normalised_not_taken_as_counts(self) -> None:
        """Scaling every weight by a constant cannot change a distance."""
        rng = np.random.default_rng(3)
        u = rng.normal(size=200)
        v = rng.normal(size=200)
        wv = rng.gamma(2.0, size=200)

        assert w1_weighted(u, v, v_weights=wv) == pytest.approx(
            w1_weighted(u, v, v_weights=17.0 * wv), rel=1e-9
        )


class TestSlicedWasserstein:
    def test_sees_correlation_that_the_axis_metrics_miss(self) -> None:
        """The reason the metric exists, as a test.

        Both samples have standard normal marginals on both axes. Only the
        correlation differs, so every per-axis Wasserstein distance is ~0 while
        the joint distributions are plainly different.
        """
        rng = np.random.default_rng(0)
        n = 20_000
        independent = rng.normal(size=(n, 2))
        a = rng.normal(size=n)
        b = rng.normal(size=n)
        correlated = np.column_stack([a, 0.9 * a + np.sqrt(1.0 - 0.81) * b])

        per_axis = max(
            w1_weighted(independent[:, i], correlated[:, i]) for i in range(2)
        )
        sliced = sliced_wasserstein(independent, correlated, seed=0, n_projections=128)

        assert per_axis < 0.03
        assert sliced > 5.0 * per_axis

    def test_is_near_zero_for_two_draws_from_one_distribution(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(8_000, 3))
        y = rng.normal(size=(8_000, 3))

        assert sliced_wasserstein(x, y, seed=0, n_projections=128) < 0.05

    def test_grows_with_a_shift(self) -> None:
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4_000, 3))

        near = sliced_wasserstein(x, x + 0.1, seed=0, n_projections=64)
        far = sliced_wasserstein(x, x + 0.5, seed=0, n_projections=64)

        assert near < far

    def test_reweighting_reduces_the_distance(self) -> None:
        """The use RAN puts it to: does `w` move `comp` toward `ref`?"""
        rng = np.random.default_rng(3)
        ref = rng.normal(loc=0.4, size=(6_000, 2))
        comp = rng.normal(size=(6_000, 2))
        # Importance weights taking N(0,1) to N(0.4,1), on both axes.
        logw = (comp * 0.4 - 0.5 * 0.4**2).sum(axis=1)
        w = np.exp(logw)

        before = sliced_wasserstein(ref, comp, seed=0, n_projections=128)
        after = sliced_wasserstein(ref, comp, comp_weights=w, seed=0, n_projections=128)

        assert after < 0.5 * before

    def test_is_reproducible_from_its_seed(self) -> None:
        rng = np.random.default_rng(4)
        x, y = rng.normal(size=(500, 4)), rng.normal(size=(500, 4))

        first = sliced_wasserstein(x, y, seed=7, n_projections=32)
        second = sliced_wasserstein(x, y, seed=7, n_projections=32)

        assert first == pytest.approx(second)

    def test_different_projection_draws_disagree_a_little(self) -> None:
        """The estimate carries Monte-Carlo error from the projection draw.

        It is an average over `n_projections` directions, so a single number is
        not exact and the benchmark reports the spread across seeds.
        """
        rng = np.random.default_rng(5)
        x, y = rng.normal(size=(2_000, 6)), rng.normal(size=(2_000, 6)) + 0.2

        values = [sliced_wasserstein(x, y, seed=s, n_projections=16) for s in range(8)]

        assert 0.0 < float(np.std(values)) < 0.05

    def test_standardises_against_the_reference(self) -> None:
        """Directions are drawn on the sphere, so the axes must be commensurable.

        Without it a variable carrying a larger numerical scale dominates every
        projection and the metric quietly becomes a measurement of that one
        axis. Scaling one axis of both inputs must not change the answer.
        """
        rng = np.random.default_rng(6)
        x = rng.normal(size=(4_000, 2))
        y = rng.normal(size=(4_000, 2)) + 0.3
        stretch = np.array([100.0, 1.0])

        assert sliced_wasserstein(x, y, seed=0, n_projections=64) == pytest.approx(
            sliced_wasserstein(x * stretch, y * stretch, seed=0, n_projections=64),
            rel=1e-6,
        )


def test_one_dimensional_sliced_equals_the_single_axis_distance() -> None:
    """In d=1 every unit direction is +-1, so slicing can add nothing.

    The axis-aligned metrics are the special case of this one, and in a single
    dimension they must agree exactly rather than approximately.
    """
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(3_000, 1))
    comp = rng.normal(loc=0.4, size=(3_000, 1))
    weights = rng.gamma(2.0, size=3_000)

    scale = ref.std(axis=0)
    axis = w1_weighted(
        ((ref - ref.mean(axis=0)) / scale)[:, 0],
        ((comp - ref.mean(axis=0)) / scale)[:, 0],
        v_weights=weights,
    )
    sliced = sliced_wasserstein(
        ref, comp, seed=0, n_projections=16, comp_weights=weights
    )

    assert sliced == pytest.approx(axis, rel=1e-9)


class TestNullFloor:
    """What does 'as good as it gets' look like for each metric?

    Comparing a sliced improvement to a per-axis one is comparing two numbers
    with different denominators: random directions may simply have had more to
    fix. The residuals are the comparable quantity, and they are only
    interpretable against the floor each metric reaches when the two samples
    are drawn from the *same* distribution -- which is what splitting the
    reference in half gives, by construction.
    """

    def test_a_split_reference_reaches_both_floors(self) -> None:
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(8_000, 4))

        floor = null_floors(ref, n=4_000, seed=0, n_projections=64)

        assert floor.sliced < 0.05
        assert floor.axis < 0.05

    def test_the_floors_are_not_identical_in_more_than_one_dimension(self) -> None:
        """They measure different directions, so they need separate floors.

        Quoting one residual against the other's floor is the error this
        exists to prevent.
        """
        rng = np.random.default_rng(1)
        ref = rng.normal(size=(4_000, 8))

        floor = null_floors(ref, n=2_000, seed=0, n_projections=64)

        assert floor.sliced != pytest.approx(floor.axis, rel=1e-6)

    def test_the_halves_are_disjoint(self) -> None:
        """Overlapping halves would report a floor of zero."""
        rng = np.random.default_rng(2)
        ref = np.arange(1000, dtype=np.double).reshape(-1, 1)
        del rng

        floor = null_floors(ref, n=500, seed=0, n_projections=4)

        assert floor.sliced > 0.0


class TestFloorSampleSize:
    """The floor has to be measured at the size of the actual comparison.

    W1 between two empirical measures of the same distribution falls as
    n^-1/2, so a floor computed from two halves of an n-event sample is the
    floor for n/2 -- about 1.41x too large -- and every residual quoted against
    it comes out that much too small. The reference pool is larger than the
    comparison, so the two null draws can both be full size.
    """

    def test_the_null_draws_are_the_size_of_the_comparison(self) -> None:
        rng = np.random.default_rng(0)
        pool = rng.normal(size=(4_000, 3))

        big = null_floors(pool, n=2_000, seed=0, n_projections=64)
        small = null_floors(pool, n=500, seed=0, n_projections=64)

        # Same pool, same distribution: only the draw size differs, and the
        # floor must fall with it.
        assert big.sliced < small.sliced
        assert big.axis < small.axis

    def test_it_refuses_a_pool_too_small_for_two_draws(self) -> None:
        pool = np.random.default_rng(1).normal(size=(600, 2))

        with pytest.raises(ValueError, match="two disjoint"):
            _ = null_floors(pool, n=400, seed=0, n_projections=8)

    def test_the_floor_falls_roughly_as_the_square_root(self) -> None:
        rng = np.random.default_rng(2)
        pool = rng.normal(size=(32_000, 2))

        near = null_floors(pool, n=2_000, seed=0, n_projections=64).sliced
        far = null_floors(pool, n=8_000, seed=0, n_projections=64).sliced

        # 4x the events should be about 2x tighter; allow the estimator slack.
        assert 1.5 < near / far < 2.7
