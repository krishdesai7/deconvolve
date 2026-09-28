"""The per-dimension metric kernels behind `metrics.json`.

Every one of them is now computed with `jnp` rather than scipy, so scipy is the
oracle these assert against rather than the implementation: `wasserstein_distance`
and `jensenshannon` are what the numbers in every run predating the port were
produced by, and a port that shifts them is a port that invalidates the archive.
"""

from typing import TYPE_CHECKING

import numpy as np
import pytest
from deconvolve.evaluation.evaluate import (
    _bin_edges,
    _js_from_histograms,
    _js_per_dim,
    _metrics_per_dim,
    _normalized_histograms,
    _triangular_from_histograms,
    _triangular_per_dim,
    _wd_per_dim,
)
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _scipy_wd_per_dim(
    ref: np.ndarray, comp: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """`_wd_per_dim`'s pre-port body, kept here as the oracle."""
    ref_2d: NDArray[np.double] = ref.reshape(-1, 1) if ref.ndim == 1 else ref
    comp_2d: NDArray[np.double] = comp.reshape(-1, 1) if comp.ndim == 1 else comp
    return np.array(
        object=[
            wasserstein_distance(ref_2d[:, i], comp_2d[:, i], v_weights=weights)
            for i in range(ref_2d.shape[1])
        ]
    )


class TestNormalizedHistograms:
    def test_treats_1d_samples_as_one_feature(self) -> None:
        p, q = _normalized_histograms(
            ref=np.array(object=[0.0, 0.0, 1.0, 1.0]),
            comp=np.array(object=[0.0, 1.0, 1.0, 1.0]),
            n_bins=2,
        )

        assert p.shape == (1, 2)
        assert q.shape == (1, 2)
        np.testing.assert_allclose(actual=p, desired=[[0.5, 0.5]])
        np.testing.assert_allclose(actual=q, desired=[[0.25, 0.75]])

    def test_rejects_mixed_rank_inputs(self) -> None:
        with pytest.raises(expected_exception=ValueError, match="same rank"):
            _ = _normalized_histograms(
                ref=np.array(object=[0.0, 1.0]),
                comp=np.array(object=[[0.0, 1.0], [1.0, 0.0]]),
                n_bins=2,
            )

    def test_weights_only_comparison_samples(self) -> None:
        ref = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        comp = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

        p, q = _normalized_histograms(
            ref, comp, weights=np.array([1.0, 1.0, 2.0]), n_bins=2
        )

        np.testing.assert_allclose(p, [[0.5, 0.5], [0.5, 0.5]])
        np.testing.assert_allclose(q, [[0.25, 0.75], [0.5, 0.5]])

    def test_matches_numpy_binning_on_a_continuous_sample(self) -> None:
        """The bin a value lands in must agree with `np.histogram`'s edges.

        `jnp.searchsorted` against the same `linspace` is the whole of the port;
        an off-by-one at a bin boundary would move mass between neighbours and
        would not show up on the small hand-checked cases above.

        The reference is float64 on purpose. `np.histogram` accumulates in the
        weights' own dtype, so passing it float32 weights -- which is what the
        pre-port code did -- makes it the *less* accurate of the two, and
        asserting against it would pin this implementation to someone else's
        rounding. Tolerance is absolute rather than relative because an empty
        bin has no relative error and a near-empty one has an unbounded amount
        of it; absolute mass is what the divergences sum.
        """
        rng = np.random.default_rng(0)
        ref = rng.normal(size=(4000, 3)).astype(np.float32)
        comp = (rng.normal(size=(4000, 3)) * 1.4).astype(np.float32)
        weights = rng.gamma(2.0, size=4000).astype(np.float32)

        p, q = _normalized_histograms(ref, comp, weights=weights, n_bins=64)
        edges = _bin_edges(ref, comp, 64)

        for i in range(3):
            h_ref = np.histogram(ref[:, i], bins=edges[i])[0].astype(np.float64)
            h_comp = np.histogram(
                comp[:, i], bins=edges[i], weights=weights.astype(np.float64)
            )[0]
            np.testing.assert_allclose(p[i], h_ref / h_ref.sum(), atol=1e-9)
            np.testing.assert_allclose(q[i], h_comp / h_comp.sum(), atol=1e-7)

    def test_leaves_an_empty_histogram_unnormalized(self) -> None:
        ref = np.array([[0.0], [1.0]], dtype=np.float32)
        comp = np.array([[0.0], [1.0]], dtype=np.float32)

        _, q = _normalized_histograms(
            ref, comp, weights=np.zeros(2, dtype=np.float32), n_bins=2
        )

        np.testing.assert_allclose(q, [[0.0, 0.0]])


class TestBinEdges:
    def test_spans_each_column_s_combined_range(self) -> None:
        rng = np.random.default_rng(20)
        ref = rng.normal(size=(500, 3)).astype(np.float32)
        comp = (rng.normal(size=(500, 3)) * 2.0).astype(np.float32)

        edges = _bin_edges(ref, comp, 32)

        assert edges.shape == (3, 33)
        for i in range(3):
            lo = min(ref[:, i].min(), comp[:, i].min())
            hi = max(ref[:, i].max(), comp[:, i].max())
            np.testing.assert_array_equal(
                edges[i], np.linspace(lo, hi, 33, dtype=np.float32)
            )

    def test_edges_are_float32(self) -> None:
        """`JAX_ENABLE_X64=0` would truncate float64 edges at the trace
        boundary, so the host and the device would bin against different
        numbers. Deciding the dtype here is what stops that.
        """
        rng = np.random.default_rng(21)
        ref = rng.normal(size=(100, 2)).astype(np.float32)
        comp = rng.normal(size=(100, 2)).astype(np.float32)

        assert _bin_edges(ref, comp, 8).dtype == np.float32


class TestWassersteinPerDim:
    def test_matches_scipy_on_weighted_columns(self) -> None:
        rng = np.random.default_rng(1)
        ref = rng.normal(size=(5000, 4)).astype(np.float32)
        comp = (rng.normal(size=(5000, 4)) * 1.3 + 0.2).astype(np.float32)
        weights = rng.gamma(2.0, size=5000).astype(np.float32)

        np.testing.assert_allclose(
            _wd_per_dim(ref=ref, comp=comp, weights=weights),
            _scipy_wd_per_dim(ref, comp, weights),
            rtol=1e-5,
        )

    def test_matches_scipy_unweighted(self) -> None:
        rng = np.random.default_rng(2)
        ref = rng.normal(size=(5000, 4)).astype(np.float32)
        comp = (rng.normal(size=(5000, 4)) * 1.3 + 0.2).astype(np.float32)

        np.testing.assert_allclose(
            _wd_per_dim(ref=ref, comp=comp),
            _scipy_wd_per_dim(ref, comp),
            rtol=1e-5,
        )

    def test_matches_scipy_when_the_two_samples_share_values(self) -> None:
        """Ties across the two samples are where a merged-CDF port goes wrong.

        Pooling and sorting puts equal values from `ref` and `comp` adjacent in
        an order the sort picks; the distance is only invariant to that choice
        because the gaps it multiplies are zero. Integers force the case.
        """
        rng = np.random.default_rng(3)
        ref = rng.integers(0, 8, size=(3000, 2)).astype(np.float32)
        comp = rng.integers(0, 8, size=(3000, 2)).astype(np.float32)
        weights = rng.gamma(2.0, size=3000).astype(np.float32)

        np.testing.assert_allclose(
            _wd_per_dim(ref=ref, comp=comp, weights=weights),
            _scipy_wd_per_dim(ref, comp, weights),
            rtol=1e-5,
        )

    def test_is_invariant_to_rescaling_the_weights(self) -> None:
        rng = np.random.default_rng(4)
        ref = rng.normal(size=(2000, 2)).astype(np.float32)
        comp = rng.normal(size=(2000, 2)).astype(np.float32)
        weights = rng.gamma(2.0, size=2000).astype(np.float32)

        np.testing.assert_allclose(
            _wd_per_dim(ref=ref, comp=comp, weights=weights),
            _wd_per_dim(ref=ref, comp=comp, weights=weights * 7.5),
            rtol=1e-5,
        )

    def test_handles_a_flat_1d_sample(self) -> None:
        rng = np.random.default_rng(5)
        ref = rng.normal(size=2000).astype(np.float32)
        comp = (rng.normal(size=2000) + 0.5).astype(np.float32)

        result = _wd_per_dim(ref=ref, comp=comp)

        assert result.shape == (1,)
        np.testing.assert_allclose(result, _scipy_wd_per_dim(ref, comp), rtol=1e-5)


class TestDivergencesPerDim:
    def test_js_reduces_over_histogram_bins(self) -> None:
        ref = np.array([[0.0, 0.0], [0.0, 1.0]])
        comp = np.array([[1.0, 0.0], [1.0, 1.0]])

        result = _js_per_dim(ref, comp, n_bins=2)

        np.testing.assert_allclose(result, [np.log(2.0), 0.0], atol=1e-15)

    def test_triangular_reduces_over_histogram_bins(self) -> None:
        ref = np.array([[0.0, 0.0], [0.0, 1.0]])
        comp = np.array([[1.0, 0.0], [1.0, 1.0]])

        result = _triangular_per_dim(ref, comp, n_bins=2)

        np.testing.assert_allclose(result, [2.0, 0.0], atol=1e-15)

    def test_js_of_an_empty_histogram_is_not_a_number(self) -> None:
        """A histogram with no mass has no distribution to be a divergence from.

        `scipy.spatial.distance.jensenshannon` divides each input by its own
        sum, so an all-zero row comes back NaN rather than 0 --- and 0 would be
        the actively wrong answer, since it reads as "these agree perfectly".
        `_normalized_histograms` leaves such a row unnormalized, so preserving
        the NaN is the reimplementation's job rather than something it inherits.
        """
        p = np.array([[0.5, 0.5]])
        q = np.array([[0.0, 0.0]])

        assert np.isnan(_js_from_histograms(p, q)).all()

    def test_js_matches_scipy_on_a_continuous_sample(self) -> None:
        """Our JS and scipy's must agree on *one* pair of histograms.

        The histograms are built once and handed to both sides rather than
        rebuilt for each. `_counts` scatters into its bins with `.at[].add`,
        which on a GPU is an atomic scatter-add: the summation order varies
        between two invocations of the same kernel on the same input, so a
        second call returns counts differing in the last float32 ulp. Calling
        `_js_per_dim` here for the actual and `_normalized_histograms` for the
        expected therefore compared two slightly different distributions and
        failed at 4e-8 relative on GPU while passing on CPU, where the scatter
        is sequential. What is being asserted is the divergence, not the
        determinism of the binning; `_js_per_dim`'s own wiring is covered by
        `test_js_reduces_over_histogram_bins`.
        """
        rng = np.random.default_rng(6)
        ref = rng.normal(size=(4000, 3)).astype(np.float32)
        comp = (rng.normal(size=(4000, 3)) * 1.4).astype(np.float32)
        weights = rng.gamma(2.0, size=4000).astype(np.float32)

        p, q = _normalized_histograms(ref, comp, weights=weights, n_bins=100)
        expected = np.array([jensenshannon(p[i], q[i]) ** 2 for i in range(3)])

        np.testing.assert_allclose(_js_from_histograms(p, q), expected, rtol=1e-9)


class TestFloat32Histograms:
    """What the device float32 scatter costs, measured against float64.

    This is the one place the port could have moved a published number, so it
    is pinned rather than argued about. It is pinned against float64 and not
    against the code being replaced: `np.histogram` sums weights in the dtype
    it is handed, so the pre-port path was itself float32 and is the weaker of
    the two references.
    """

    @staticmethod
    def _float64_histograms(
        ref: np.ndarray, comp: np.ndarray, weights: np.ndarray, n_bins: int
    ) -> tuple[np.ndarray, np.ndarray]:
        p = np.empty((ref.shape[1], n_bins))
        q = np.empty((ref.shape[1], n_bins))
        edges = _bin_edges(ref, comp, n_bins)
        for i in range(ref.shape[1]):
            h_ref: NDArray[np.float64] = np.histogram(ref[:, i], bins=edges[i])[
                0
            ].astype(np.float64)
            h_comp: NDArray[np.float64] = np.histogram(
                comp[:, i], bins=edges[i], weights=weights.astype(np.float64)
            )[0]
            p[i] = h_ref / h_ref.sum()
            q[i] = h_comp / h_comp.sum()
        return p, q

    def test_the_divergences_land_within_a_printed_digit_of_float64(self) -> None:
        """JS is displayed at six decimals, the triangular discriminator at four.

        The two bounds are stated differently because the metrics are: JS is a
        probability-scale number bounded by log 2, so an absolute bound says
        what reaches its printed digit, while the triangular discriminator is
        displayed x1e3 and runs to ~100 there, where the same statement has to
        be relative. Both come to the same place -- a few times 1e-7 of the
        value, two digits below anything printed.

        **The JS bound is 1e-7 and not the 9e-9 the gap actually measures,
        because the gap is platform-dependent.** `_counts` accumulates the
        scatter in float32, and the order it accumulates in is the hardware's
        choice: on a CPU the float32-to-float64 gap here is ~9e-9, on an A100
        it is ~1.3e-8. This assertion was originally `atol=1e-8`, which against
        `assert_allclose`'s default `rtol=1e-7` came to an effective 1.26e-8 --
        and the A100 returned 1.288e-8, failing by two percent. Pinning a
        measured constant was the error; what the test is for is the claim in
        its own name, that float32 binning does not disturb the digit
        `metrics.json` prints. JS prints at six decimals, so 1e-7 is a tenth of
        the last printed digit, eight times the largest gap either platform has
        shown, and still well inside the 5.9e-7 of the `np.histogram` path this
        replaced -- so a regression to that would still fail here.
        """
        rng = np.random.default_rng(11)
        ref = rng.normal(size=(20000, 3)).astype(np.float32)
        comp = (rng.normal(size=(20000, 3)) * 1.4).astype(np.float32)
        weights = rng.gamma(2.0, size=20000).astype(np.float32)

        p, q = _normalized_histograms(ref, comp, weights=weights, n_bins=100)
        exact_p, exact_q = self._float64_histograms(ref, comp, weights, 100)

        np.testing.assert_allclose(
            _js_from_histograms(p, q),
            _js_from_histograms(exact_p, exact_q),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _triangular_from_histograms(p, q),
            _triangular_from_histograms(exact_p, exact_q),
            rtol=1e-6,
            atol=1e-12,
        )

    def test_beats_the_float32_numpy_path_it_replaced(self) -> None:
        """Centering the weights before the scatter is what buys this.

        Scattering them raw would sum ~200 magnitudes per bin, which is what
        `np.histogram` does with float32 weights. Both are float32 reductions;
        only one of them sums residuals.
        """
        rng = np.random.default_rng(12)
        ref = rng.normal(size=(20000, 3)).astype(np.float32)
        comp = (rng.normal(size=(20000, 3)) * 1.4).astype(np.float32)
        weights = rng.gamma(2.0, size=20000).astype(np.float32)

        _, q = _normalized_histograms(ref, comp, weights=weights, n_bins=100)
        _, exact_q = self._float64_histograms(ref, comp, weights, 100)

        legacy_q = np.empty_like(exact_q)
        edges = _bin_edges(ref, comp, 100)
        for i in range(3):
            h = np.histogram(comp[:, i], bins=edges[i], weights=weights)[0].astype(
                np.float64
            )
            legacy_q[i] = h / h.sum()

        assert np.abs(q - exact_q).max() < np.abs(legacy_q - exact_q).max()


class TestFusedMetrics:
    """`_metrics_per_dim` is the path `evaluate_run` takes.

    It shares one histogram between the two divergences instead of building it
    twice, so what has to hold is that sharing changed no number.

    Unlike `test_js_matches_scipy_on_a_continuous_sample`, these cannot build
    the histogram once and hand it to both sides: that the fused and unfused
    paths each bin the sample for themselves **is** the claim. So the two
    divergences are compared at `rtol=1e-6` rather than `assert_allclose`'s
    default `1e-7`. `_counts` scatters with `.at[].add`, which on a GPU is an
    atomic accumulation whose summation order varies between calls, and the
    resulting last-ulp disagreement is worth ~4e-8 relative on a divergence
    and has been measured at 1.05e-7 --- i.e. straddling the default, which
    made these two a coin flip on the cluster and a certainty on a CPU. 1e-6
    clears the noise and stays three orders inside anything a real fusion bug
    could produce, since sharing a histogram either changes the algebra
    outright or changes nothing.

    Wasserstein stays at the default. Its scan is an ordered reduction with no
    atomics, so the agreement there is exact and worth continuing to pin.
    """

    def test_agrees_with_the_individual_helpers(self) -> None:
        rng = np.random.default_rng(7)
        ref = rng.normal(size=(4000, 3)).astype(np.float32)
        comp = (rng.normal(size=(4000, 3)) * 1.4 + 0.1).astype(np.float32)

        fused = _metrics_per_dim(ref, comp)

        np.testing.assert_allclose(fused.wasserstein, _wd_per_dim(ref=ref, comp=comp))
        np.testing.assert_allclose(
            fused.jensenshannon, _js_per_dim(ref, comp), rtol=1e-6
        )
        np.testing.assert_allclose(
            fused.triangular, _triangular_per_dim(ref, comp), rtol=1e-6
        )

    def test_agrees_with_the_individual_helpers_when_weighted(self) -> None:
        rng = np.random.default_rng(8)
        ref = rng.normal(size=(4000, 3)).astype(np.float32)
        comp = (rng.normal(size=(4000, 3)) * 1.4 + 0.1).astype(np.float32)
        w = rng.gamma(2.0, size=4000).astype(np.float32)

        fused = _metrics_per_dim(ref, comp, weights=w)

        np.testing.assert_allclose(
            fused.wasserstein, _wd_per_dim(ref=ref, comp=comp, weights=w)
        )
        np.testing.assert_allclose(
            fused.jensenshannon, _js_per_dim(ref, comp, weights=w), rtol=1e-6
        )
        np.testing.assert_allclose(
            fused.triangular, _triangular_per_dim(ref, comp, weights=w), rtol=1e-6
        )

    def test_accepts_weights_already_on_device(self) -> None:
        """`evaluate_run` hands over the generator's output without a host copy."""
        import jax.numpy as jnp

        rng = np.random.default_rng(9)
        ref = rng.normal(size=(2000, 2)).astype(np.float32)
        comp = rng.normal(size=(2000, 2)).astype(np.float32)
        w = rng.gamma(2.0, size=2000).astype(np.float32)

        np.testing.assert_allclose(
            _metrics_per_dim(ref, comp, weights=jnp.asarray(w)).wasserstein,
            _metrics_per_dim(ref, comp, weights=w).wasserstein,
        )
