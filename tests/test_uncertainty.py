"""Tests for the bootstrap x seed variance decomposition.

The claim the module exists to support is that quadrature-summing two
one-dimensional sweeps overstates the total by exactly one interaction term.
`TestNaiveQuadrature` is that claim as a test; everything else is there to
make sure the components it is built from are the right numbers.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from deconvolve.coretypes import Events, Populations, Resample, Split
from deconvolve.uncertainty import (
    DesignSpec,
    binned_spectra,
    bootstrap,
    bootstrap_multiplicities,
    cell_path,
    component_covariances,
    correlation,
    decompose,
    evaluation_covariance,
    evaluation_variance,
    load_cells,
    multinomial_off_diagonal,
    quantile_edges,
    replicate_splits,
    reserve_evaluation_set,
    weighted_means,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from deconvolve.coretypes import DatasetSplits
    from deconvolve.uncertainty import (
        Covariances,
        Design,
        EvaluationSet,
        VarianceComponents,
    )
    from numpy.typing import NDArray


def _simulate(
    *,
    n_data: int,
    n_init: int,
    sigma_a: float,
    sigma_b: float,
    sigma_eps: float,
    seed: int = 0,
    n_components: int = 1,
) -> NDArray[np.double]:
    """A grid drawn from the model the decomposition assumes: mu + a + b + eps."""
    rng: np.random.Generator = np.random.default_rng(seed)
    a: NDArray[np.double] = rng.normal(scale=sigma_a, size=(n_data, 1, n_components))
    b: NDArray[np.double] = rng.normal(scale=sigma_b, size=(1, n_init, n_components))
    eps: NDArray[np.double] = rng.normal(
        scale=sigma_eps, size=(n_data, n_init, n_components)
    )
    return 3.0 + a + b + eps


class TestDecompose:
    def test_recovers_the_three_components_it_was_simulated_from(self) -> None:
        grid: NDArray[np.double] = _simulate(
            n_data=200, n_init=200, sigma_a=0.5, sigma_b=0.2, sigma_eps=0.3
        )

        components: VarianceComponents = decompose(grid)

        assert float(components.data[0]) == pytest.approx(expected=0.25, rel=0.15)
        assert float(components.init[0]) == pytest.approx(expected=0.04, rel=0.15)
        assert float(components.interaction[0]) == pytest.approx(
            expected=0.09, rel=0.05
        )

    def test_the_components_sum_to_the_variance_of_the_whole_grid(self) -> None:
        """`total` has to be the thing anyone would compute without the model."""
        grid: NDArray[np.double] = _simulate(
            n_data=120, n_init=120, sigma_a=0.4, sigma_b=0.3, sigma_eps=0.2
        )

        assert float(decompose(grid).total[0]) == pytest.approx(
            expected=float(np.var(grid)), rel=0.05
        )

    def test_it_is_elementwise_over_the_trailing_axes(self) -> None:
        grid: NDArray[np.double] = _simulate(
            n_data=40,
            n_init=8,
            sigma_a=0.3,
            sigma_b=0.1,
            sigma_eps=0.2,
            n_components=5,
        )

        components: VarianceComponents = decompose(grid)

        assert components.data.shape == (5,)
        for k in range(5):
            one: VarianceComponents = decompose(grid[:, :, k])
            assert float(one.data) == pytest.approx(expected=float(components.data[k]))

    def test_a_component_with_no_real_effect_can_come_out_negative(self) -> None:
        """Reported raw, not clamped: a negative value means 'unresolved'.

        Clamping would turn an estimate consistent with zero into a confident
        zero, and a variance budget that never admits it cannot resolve a
        source is a variance budget nobody should trust.
        """
        negatives: list[float] = [
            float(
                decompose(
                    _simulate(
                        n_data=4,
                        n_init=4,
                        sigma_a=0.0,
                        sigma_b=0.0,
                        sigma_eps=1.0,
                        seed=s,
                    )
                ).data[0]
            )
            for s in range(40)
        ]

        assert min(negatives) < 0.0

    @pytest.mark.parametrize(
        argnames=("n_data", "n_init"), argvalues=[(1, 8), (8, 1), (1, 1)]
    )
    def test_it_refuses_a_grid_with_a_degenerate_axis(
        self, n_data: int, n_init: int
    ) -> None:
        with pytest.raises(
            expected_exception=ValueError, match="at least 2 datasets and 2 seeds"
        ):
            _ = decompose(np.zeros(shape=(n_data, n_init, 3)))

    def test_it_refuses_a_grid_holding_a_failed_run(self) -> None:
        grid: NDArray[np.double] = _simulate(
            n_data=4, n_init=4, sigma_a=0.1, sigma_b=0.1, sigma_eps=0.1
        )
        grid[2, 1, 0] = np.nan

        with pytest.raises(expected_exception=ValueError, match="non-finite"):
            _ = decompose(grid)


class TestNaiveQuadrature:
    """The module's reason for existing, as a test.

    Varying the seed at one fixed dataset measures `sigma_b^2 + sigma_eps^2`;
    varying the dataset at one fixed seed measures `sigma_a^2 + sigma_eps^2`.
    Adding those counts the interaction twice.
    """

    def test_it_exceeds_the_total_by_exactly_one_interaction_term(self) -> None:
        components: VarianceComponents = decompose(
            _simulate(n_data=30, n_init=30, sigma_a=0.4, sigma_b=0.3, sigma_eps=0.5)
        )

        assert float(
            (components.naive_quadrature - components.total)[0]
        ) == pytest.approx(expected=float(components.interaction[0]))

    def test_the_two_one_dimensional_sweeps_reproduce_the_overstatement(self) -> None:
        """Measured the way the one-dimensional sweeps actually measure it."""
        grid: NDArray[np.double] = _simulate(
            n_data=400, n_init=400, sigma_a=0.4, sigma_b=0.3, sigma_eps=0.5
        )
        # One column is a seed sweep at fixed data; one row is a data sweep at
        # fixed seed. Neither can see more than its own effect plus eps.
        seed_sweep: float = float(np.var(a=grid[0, :, 0], ddof=1))
        data_sweep: float = float(np.var(a=grid[:, 0, 0], ddof=1))
        components: VarianceComponents = decompose(grid)

        assert seed_sweep + data_sweep == pytest.approx(
            expected=float(components.naive_quadrature[0]), rel=0.15
        )
        assert seed_sweep + data_sweep > float(components.total[0])


class TestComponentCovariances:
    def test_the_diagonals_are_the_scalar_components(self) -> None:
        """Two code paths, one answer: the matrix version cannot drift."""
        grid: NDArray[np.double] = _simulate(
            n_data=25,
            n_init=9,
            sigma_a=0.3,
            sigma_b=0.2,
            sigma_eps=0.4,
            n_components=6,
        )

        covariances: Covariances = component_covariances(grid)
        components: VarianceComponents = decompose(grid)

        for field in ("data", "init", "interaction"):
            assert np.diag(getattr(covariances, field)) == pytest.approx(
                expected=getattr(components, field)
            )

    def test_it_finds_correlation_that_bin_by_bin_variances_cannot(self) -> None:
        """The whole off-diagonal argument, in a case with a known answer."""
        rng: np.random.Generator = np.random.default_rng(seed=0)
        # One dataset effect shared by both bins with opposite sign: perfectly
        # anti-correlated across replicates, identical in per-bin variance to
        # an uncorrelated pair.
        shared: NDArray[np.double] = rng.normal(scale=0.5, size=(60, 1))
        grid: NDArray[np.double] = np.stack(
            arrays=[shared, -shared], axis=-1
        ) + rng.normal(scale=0.02, size=(60, 8, 2))

        corr: NDArray[np.double] = correlation(component_covariances(grid).data)

        assert corr[0, 1] == pytest.approx(expected=-1.0, abs=0.05)

    def test_it_refuses_anything_that_is_not_a_grid_of_spectra(self) -> None:
        with pytest.raises(ValueError, match=r"\(B, S, K\)"):
            _ = component_covariances(np.zeros(shape=(4, 4, 3, 2)))

    def test_the_correction_is_not_optional(self) -> None:
        """Skipping it inflates the data covariance, in the flattering direction.

        The dataset-averaged spectrum still carries `mean_S eps`, so the raw
        between-dataset covariance overstates `Cov_a` by `Cov_eps / S`.
        """
        grid: NDArray[np.double] = _simulate(
            n_data=50,
            n_init=4,
            sigma_a=0.05,
            sigma_b=0.0,
            sigma_eps=0.5,
            n_components=3,
        )

        corrected: NDArray[np.double] = component_covariances(grid).data
        raw: NDArray[np.double] = np.cov(m=grid.mean(axis=1), rowvar=False, ddof=1)

        assert np.trace(a=raw) > 2.0 * np.trace(a=corrected)


class TestCorrelation:
    def test_the_diagonal_is_one(self) -> None:
        cov: NDArray[np.double] = np.array(object=[[4.0, 1.0], [1.0, 9.0]])

        assert np.diag(v=correlation(cov)) == pytest.approx(expected=[1.0, 1.0])

    def test_a_non_positive_variance_gives_nan_rather_than_a_confident_zero(
        self,
    ) -> None:
        """A corrected component covariance can go negative on the diagonal."""
        cov: NDArray[np.double] = np.array(object=[[4.0, 1.0], [1.0, -0.1]])

        corr: NDArray[np.double] = correlation(cov)

        assert corr[0, 0] == pytest.approx(expected=1.0)
        assert np.isnan(corr[1, 1])
        assert np.isnan(corr[0, 1])


class TestBinning:
    def test_quantile_edges_give_roughly_equal_occupancy(self) -> None:
        """Linear edges over a jet observable put most bins in an empty tail."""
        column: NDArray[np.single] = (
            np.random.default_rng(seed=0).lognormal(size=40_000).astype(np.single)
        )

        counts: NDArray[np.intp] = np.histogram(
            column, bins=quantile_edges(column, n_bins=20)
        )[0]

        assert counts.min() > 0.9 * counts.max()

    def test_the_edges_keep_the_extreme_events(self) -> None:
        column = np.linspace(0.0, 1.0, 1000, dtype=np.single)

        edges = quantile_edges(column, n_bins=10)

        assert np.histogram(column, bins=edges)[0].sum() == column.size

    def test_duplicate_edges_collapse_rather_than_raising(self) -> None:
        """A discrete observable such as `mult` has repeated quantiles."""
        column = np.repeat(np.arange(3, dtype=np.single), 100)

        edges = quantile_edges(column, n_bins=20)

        assert edges.size < 21
        assert np.all(np.diff(edges) > 0)

    def test_a_constant_column_is_refused(self) -> None:
        with pytest.raises(expected_exception=ValueError, match="constant"):
            _ = quantile_edges(np.ones(shape=100, dtype=np.single), n_bins=5)

    def test_binned_spectra_keep_the_design_shape_and_sum_to_one(self) -> None:
        rng: np.random.Generator = np.random.default_rng(seed=1)
        column: NDArray[np.single] = rng.normal(size=5_000).astype(dtype=np.single)
        weights: NDArray[np.double] = rng.gamma(shape=4.0, size=(3, 4, 5_000))

        spectra: NDArray[np.double] = binned_spectra(
            column, weights, edges=quantile_edges(column, n_bins=8)
        )

        assert spectra.shape == (3, 4, 8)
        assert spectra.sum(axis=-1) == pytest.approx(expected=np.ones(shape=(3, 4)))

    def test_a_weight_vector_of_the_wrong_length_is_refused(self) -> None:
        column: NDArray[np.single] = np.zeros(shape=10, dtype=np.single)

        with pytest.raises(expected_exception=ValueError, match="same events"):
            _ = binned_spectra(
                column,
                np.ones(shape=(2, 2, 9)),
                edges=np.linspace(start=-1, stop=1, num=4),
            )

    def test_the_multinomial_floor_is_the_pure_closure_correlation(self) -> None:
        """A fixed total alone forces -1/(K-1) between equal-occupancy bins."""
        assert multinomial_off_diagonal(20) == pytest.approx(expected=-1.0 / 19)
        with pytest.raises(expected_exception=ValueError, match="at least 2 bins"):
            _ = multinomial_off_diagonal(1)


def test_weighted_means_are_the_unfolded_mean_per_run() -> None:
    column: NDArray[np.single] = np.array(object=[0.0, 1.0, 2.0], dtype=np.single)
    weights: NDArray[np.double] = np.array(object=[[[1.0, 1.0, 1.0], [0.0, 0.0, 3.0]]])

    assert weighted_means(column, weights)[0] == pytest.approx(expected=[1.0, 2.0])


def _populations(n: int = 400, seed: int = 0) -> Populations:
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    z_gen: NDArray[np.single] = rng.normal(size=(n, 2)).astype(dtype=np.single)
    z_true: NDArray[np.single] = rng.normal(size=(n, 2)).astype(dtype=np.single)
    return Populations(
        mc=Events(z=z_gen, x=(z_gen + 0.1).astype(dtype=np.single)),
        data=(z_true + 0.1).astype(dtype=np.single),
        truth=z_true,
    )


class TestBootstrap:
    def test_a_replicate_is_the_same_size_as_the_original(self) -> None:
        """`n` of `n`: the variance estimated is the one at the size collected."""
        pops: Populations = _populations()

        replicate: Populations = bootstrap(pops, seed=0)

        assert len(replicate.mc) == len(pops.mc)
        assert replicate.data.shape == pops.data.shape

    def test_it_draws_with_replacement(self) -> None:
        replicate: Populations = bootstrap(_populations(), seed=0)

        assert len(np.unique(ar=replicate.mc.z, axis=0)) < len(replicate.mc)

    def test_it_keeps_the_pairing_inside_each_sample(self) -> None:
        """`(z_gen, x_sim)` is one event seen twice, not two draws."""
        pops: Populations = _populations()

        replicate: Populations = bootstrap(pops, seed=1)

        assert replicate.mc.x == pytest.approx(expected=replicate.mc.z + 0.1)
        assert replicate.data == pytest.approx(expected=replicate.truth + 0.1)

    def test_mc_and_nature_resample_independently(self) -> None:
        """They are two separate samples; coupling them invents a correlation."""
        pops: Populations = _populations()

        replicate: Populations = bootstrap(pops, seed=2)

        # If the two shared one index draw, row i of each would come from the
        # same original row and the two columns would agree everywhere.
        assert not np.allclose(a=replicate.mc.z, b=replicate.truth)

    def test_it_is_reproducible_from_its_seed(self) -> None:
        pops: Populations = _populations()

        assert bootstrap(pops, seed=(42, 3)).mc.z == pytest.approx(
            expected=bootstrap(pops, seed=(42, 3)).mc.z
        )
        assert not np.allclose(
            a=bootstrap(pops, seed=(42, 3)).mc.z, b=bootstrap(pops, seed=(42, 4)).mc.z
        )


def _sorted_rows(a: NDArray[Any]) -> NDArray[Any]:
    """Rows in a canonical order, so two samples compare as multisets."""
    return a[np.lexsort(keys=a.T[::-1])]


class TestBootstrapMultiplicities:
    def test_each_sample_keeps_its_size(self) -> None:
        pops: Populations = _populations()

        mc, nature = bootstrap_multiplicities(pops, seed=0)

        assert int(mc.sum()) == len(pops.mc)
        assert int(nature.sum()) == pops.data.shape[0]
        # With replacement: some events are drawn more than once, some never.
        assert (mc == 0).any()
        assert (mc > 1).any()

    @pytest.mark.parametrize(
        argnames=("resample", "untouched"),
        argvalues=[(Resample.data, "mc"), (Resample.mc, "nature")],
    )
    def test_the_other_sample_is_left_as_collected(
        self, resample: Resample, untouched: str
    ) -> None:
        """Resampling one side at a time separates data from MC statistics."""
        mc, nature = bootstrap_multiplicities(_populations(), seed=0, resample=resample)

        fixed, varied = (mc, nature) if untouched == "mc" else (nature, mc)
        assert np.all(fixed == 1)
        assert not np.all(varied == 1)


class TestReplicateSplits:
    """Split the original events first, then expand each split by its draws.

    Splitting a resampled sample instead puts copies of one event into
    different splits: about half of every replicate's validation set would
    also be in its training set, which is not the procedure being measured.
    """

    @staticmethod
    def _splits(split_seed: int, *, bootstrap_seed: int = 0) -> DatasetSplits:
        pops: Populations = _populations(n=2000)
        return replicate_splits(
            pops,
            bootstrap_multiplicities(pops, seed=bootstrap_seed),
            split_seed=split_seed,
            batch_size=64,
        )

    def test_no_event_is_in_two_splits(self) -> None:
        parts: list[Populations] = [
            split.as_arrays().partition() for split in self._splits(split_seed=5)
        ]

        for side in ("mc", "data"):
            keys: list[set[bytes]] = [
                {row.tobytes() for row in (part.mc.z if side == "mc" else part.data)}
                for part in parts
            ]
            assert not keys[0] & keys[1]
            assert not keys[0] & keys[2]
            assert not keys[1] & keys[2]

    def test_the_split_does_not_change_the_replicate(self) -> None:
        """So the dataset axis means the same sample in every cell of a row."""
        one: Populations = self._splits(split_seed=1).select(Split.ALL).partition()
        two: Populations = self._splits(split_seed=2).select(Split.ALL).partition()

        np.testing.assert_array_equal(_sorted_rows(one.mc.z), _sorted_rows(two.mc.z))
        np.testing.assert_array_equal(_sorted_rows(one.data), _sorted_rows(two.data))

    def test_a_different_split_seed_splits_differently(self) -> None:
        one: Populations = self._splits(split_seed=1).train.as_arrays().partition()
        two: Populations = self._splits(split_seed=2).train.as_arrays().partition()

        assert {r.tobytes() for r in one.mc.z} != {r.tobytes() for r in two.mc.z}

    def test_every_draw_is_trained_on_somewhere(self) -> None:
        pops: Populations = _populations(n=2000)
        mc, nature = bootstrap_multiplicities(pops, seed=3)

        splits: DatasetSplits = replicate_splits(
            pops, (mc, nature), split_seed=4, batch_size=64
        )

        assert sum(len(split.as_arrays()) for split in splits) == int(
            mc.sum() + nature.sum()
        )


class TestEvaluationUncertainty:
    """The evaluation set's own sampling noise, which no cell contrast can see.

    Every cell reads the same evaluation events, so their fluctuation cancels
    from every component of the decomposition. It is still an uncertainty on
    the unfolded result, which *is* those weighted events. Checked against its
    definition: the spread of the result over repeated evaluation sets drawn
    with the weight function held fixed.
    """

    @staticmethod
    def _repeat(n_sets: int, n_events: int) -> tuple[NDArray[Any], ...]:
        rng: np.random.Generator = np.random.default_rng(seed=0)
        columns: NDArray[np.double] = rng.normal(size=(n_sets, n_events))
        weights: NDArray[np.double] = np.exp(0.5 * columns)
        return columns, weights

    def test_the_variance_of_the_mean_matches_repeated_evaluation_sets(
        self,
    ) -> None:
        columns, weights = self._repeat(n_sets=4000, n_events=2000)

        means: list[float] = []
        predicted: list[float] = []
        for i in range(len(columns)):
            column: NDArray[Any] = columns[i]
            run: NDArray[Any] = weights[i][None]
            means.append(float(weighted_means(column, run)[0]))
            predicted.append(float(evaluation_variance(column, run)[0]))

        assert np.var(means) == pytest.approx(expected=np.mean(predicted), rel=0.1)

    def test_the_bin_covariance_matches_repeated_evaluation_sets(self) -> None:
        columns, weights = self._repeat(n_sets=4000, n_events=2000)
        edges: NDArray[np.double] = np.array(object=[-10.0, -0.5, 0.5, 10.0])

        spectra_list: list[NDArray[np.double]] = []
        covariances: list[NDArray[np.double]] = []
        for i in range(len(columns)):
            column: NDArray[Any] = columns[i]
            run: NDArray[Any] = weights[i][None]
            spectra_list.append(binned_spectra(column, run, edges=edges)[0])
            covariances.append(evaluation_covariance(column, run, edges=edges)[0])
        spectra: NDArray[np.double] = np.array(object=spectra_list)
        predicted: NDArray[np.double] = np.mean(a=covariances, axis=0)

        # Each entry against its own sampling error. A relative tolerance is
        # the wrong test for an entry near zero, whose empirical estimate
        # carries an error comparable to the entry itself:
        # se(c_kl) = sqrt((c_kk c_ll + c_kl^2) / n_sets) for normal sampling.
        empirical: NDArray[np.double] = np.cov(m=spectra, rowvar=False)
        diagonal: NDArray[np.double] = np.diag(v=predicted)
        se: NDArray[np.double] = np.sqrt(
            (np.outer(a=diagonal, b=diagonal) + predicted**2) / len(spectra)
        )
        assert np.all(a=np.abs(empirical - predicted) < 4.0 * se)

    def test_the_bin_covariance_respects_the_normalization(self) -> None:
        """Every spectrum sums to one, so every row sums to zero."""
        columns, weights = self._repeat(n_sets=1, n_events=500)
        edges: NDArray[np.double] = quantile_edges(columns[0], n_bins=4)

        cov: NDArray[np.double] = evaluation_covariance(
            columns[0], weights, edges=edges
        )[0]

        np.testing.assert_allclose(cov.sum(axis=1), 0.0, atol=1e-12)


class TestEvaluationSet:
    def test_the_held_out_events_are_absent_from_the_training_pool(self) -> None:
        """Every replicate is read on events no replicate could have trained on."""
        pops: Populations = _populations()

        evaluation: EvaluationSet = reserve_evaluation_set(pops, n_eval=100, seed=7)

        pool: set[bytes] = {row.tobytes() for row in evaluation.pool.mc.z}
        assert all(row.tobytes() not in pool for row in evaluation.z)  # pyrefly: ignore[unknown-argument-type]
        assert len(evaluation.pool.mc) + evaluation.z.shape[0] == len(pops.mc)

    def test_nature_events_are_left_alone(self) -> None:
        """`g` is never evaluated on one, so there is nothing to hold out."""
        pops: Populations = _populations()

        evaluation: EvaluationSet = reserve_evaluation_set(pops, n_eval=100, seed=7)

        assert evaluation.pool.data.shape == pops.data.shape

    def test_the_same_seed_reserves_the_same_events(self) -> None:
        """Cells that disagree here produce weight vectors that cannot be stacked."""
        pops: Populations = _populations()

        assert reserve_evaluation_set(pops, n_eval=50, seed=42).z == pytest.approx(
            expected=reserve_evaluation_set(pops, n_eval=50, seed=42).z
        )

    @pytest.mark.parametrize(argnames="n_eval", argvalues=[0, 400, 401])
    def test_it_refuses_a_reservation_that_leaves_nothing_to_train_on(
        self, n_eval: int
    ) -> None:
        with pytest.raises(
            expected_exception=ValueError, match="n_eval must be between"
        ):
            _ = reserve_evaluation_set(_populations(n=400), n_eval=n_eval, seed=0)


class TestDesignSpec:
    def test_cells_are_numbered_seed_major(self) -> None:
        """So a design cut short is a complete grid over the datasets that ran."""
        spec = DesignSpec(n_datasets=4, n_seeds=3)

        assert [spec.cell_of_index(i) for i in range(6)] == [
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 1),
            (1, 2),
        ]
        assert spec.n_cells == 12

    @pytest.mark.parametrize(argnames="index", argvalues=[-1, 12])
    def test_an_index_outside_the_grid_is_refused(self, index: int) -> None:
        with pytest.raises(expected_exception=IndexError, match="outside a 4x3 design"):
            _ = DesignSpec(n_datasets=4, n_seeds=3).cell_of_index(index)


def _write_cell(
    design_dir: Path,
    spec: DesignSpec,
    index: int,
    weights: np.ndarray,
    *,
    n_eval: int | None = None,
    n_epochs: int = 100,
    n_layers: int = 2,
    hidden_units: int = 64,
    n_disc_steps: int = 5,
    lr_g: float = 3e-5,
    lr_d: float = 1e-4,
    lambda_dispersion: float = 0.015,
) -> None:
    b, s = spec.cell_of_index(index)
    design_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        cell_path(design_dir, index),
        weights=weights,
        meta=np.array(
            json.dumps(
                {
                    "index": index,
                    "dataset_index": b,
                    "seed_index": s,
                    "init_seed": spec.init_seed + s,
                    "data_seed": spec.data_seed,
                    "n_eval": weights.size if n_eval is None else n_eval,
                    "n_samples": 400,
                    "batch_size": 64,
                    "dim": 2,
                    "dataset": "jets",
                    "variables": ["m", "w"],
                    "gaussian_params": None,
                    "hidden_units": hidden_units,
                    "n_layers": n_layers,
                    "n_epochs": n_epochs,
                    "n_disc_steps": n_disc_steps,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "lambda_dispersion": lambda_dispersion,
                    "mmd_test": 1e-4,
                }
            )
        ),
    )


class TestLoadCells:
    def test_it_stacks_the_grid_in_design_order(self, tmp_path: Path) -> None:
        spec = DesignSpec(n_datasets=2, n_seeds=3)
        for index in range(spec.n_cells):
            _write_cell(
                tmp_path, spec, index, weights=np.full(shape=5, fill_value=float(index))
            )

        design: Design = load_cells(tmp_path, spec)

        assert design.weights.shape == (2, 3, 5)
        assert design.weights[1, 2, 0] == pytest.approx(expected=5.0)

    def test_the_per_cell_fields_are_stripped_from_the_shared_metadata(
        self, tmp_path: Path
    ) -> None:
        """`meta` describes the design, so a cell's own indices do not belong."""
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(spec.n_cells):
            _write_cell(tmp_path, spec, index, weights=np.ones(shape=4))

        meta: dict[str, Any] = load_cells(tmp_path, spec).meta

        assert "seed_index" not in meta
        assert meta["variables"] == ["m", "w"]

    def test_an_incomplete_grid_is_refused_by_name(self, tmp_path: Path) -> None:
        """A ragged design would charge the imbalance to the dataset axis."""
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in (0, 1, 3):
            _write_cell(tmp_path, spec, index, weights=np.ones(shape=4))

        with pytest.raises(
            expected_exception=FileNotFoundError, match=r"missing cells \[2\]"
        ):
            _ = load_cells(tmp_path, spec)

    def test_cells_from_different_designs_are_refused(self, tmp_path: Path) -> None:
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(3):
            _write_cell(tmp_path, spec, index, weights=np.ones(shape=4))
        _write_cell(tmp_path, spec, index=3, weights=np.ones(shape=6))

        with pytest.raises(expected_exception=ValueError, match="not from one design"):
            _ = load_cells(tmp_path, spec)

    def test_a_cell_trained_under_different_settings_is_refused(
        self, tmp_path: Path
    ) -> None:
        """I-2's regression test: a sanctioned `--flag` override must leave a trace.

        A hand-rerun of one failed cell at a different `n_epochs` than the
        rest of the array must not be silently averaged into the
        decomposition as if it were noise.
        """
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(spec.n_cells):
            _write_cell(tmp_path, spec, index, weights=np.ones(shape=4))
        # Cell 3 was hand-rerun at a different setting than the rest.
        _write_cell(tmp_path, spec, index=3, weights=np.ones(shape=4), n_epochs=250)

        with pytest.raises(expected_exception=ValueError, match="n_epochs"):
            _ = load_cells(tmp_path, spec)

    def test_cells_agreeing_on_settings_load_cleanly(self, tmp_path: Path) -> None:
        """The guard must not fire on an ordinary, uniformly-trained design."""
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(spec.n_cells):
            _write_cell(tmp_path, spec, index, weights=np.ones(shape=4))

        design: Design = load_cells(tmp_path, spec)

        assert design.weights.shape == (2, 2, 4)


class TestRunCell:
    """The wiring, with training stubbed out: what each cell is handed.

    Two properties matter and neither is visible from one run. Every cell has
    to reserve the *same* evaluation events, or the weight vectors are not
    stackable and the covariance is of nothing. And every cell has to see a
    *different* bootstrap replicate, or the dataset axis measures zero.
    """

    @staticmethod
    def _stub(monkeypatch: pytest.MonkeyPatch, seen: list[dict[str, Any]]) -> None:
        import deconvolve.evaluation.evaluate
        import deconvolve.training.engine
        import deconvolve.uncertainty.design as design_module
        from deconvolve.training import TrainResult

        monkeypatch.setattr(
            target=design_module,
            name="base_populations",
            value=lambda *_args, **_kwargs: (_populations(n=600), 2),
        )

        def fake_train(
            splits: DatasetSplits,
            dim: int,
            _units: int,
            _layers: int,
            seed: int,
            **kwargs: dict[str, Any],
        ) -> TrainResult:
            seen.append(
                {
                    "seed": seed,
                    "dim": dim,
                    "z": splits.select(Split.ALL).partition().mc.z.copy(),
                    "train_z": splits.train.as_arrays().partition().mc.z.copy(),
                    **kwargs,
                }
            )
            return TrainResult(
                g=None,  # ty: ignore[invalid-argument-type]
                d=None,  # ty: ignore[invalid-argument-type]
                history={},
                seed=seed,
                mmd_test=1.5e-4,
            )

        monkeypatch.setattr(
            target=deconvolve.training.engine, name="train", value=fake_train
        )
        monkeypatch.setattr(
            target=deconvolve.evaluation.evaluate,
            name="_get_weights",
            value=lambda _g, z_gen, **_kw: np.ones(shape=len(z_gen), dtype=np.single),  # pyrefly: ignore[unknown-argument-type]
        )

    def test_every_cell_holds_out_the_same_evaluation_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deconvolve.uncertainty import run_cell

        seen: list[dict[str, Any]] = []
        self._stub(monkeypatch, seen)
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(spec.n_cells):
            _ = run_cell(index, tmp_path, spec, n_samples=600, n_eval=100)

        design: Design = load_cells(tmp_path, spec)

        assert design.weights.shape == (2, 2, 100)

    def test_the_dataset_axis_varies_and_the_seed_axis_does_not(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deconvolve.uncertainty import run_cell

        seen: list[dict[str, Any]] = []
        self._stub(monkeypatch, seen)
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(spec.n_cells):
            _ = run_cell(index, tmp_path, spec, n_samples=600, n_eval=100)

        # Cells 0 and 1 are dataset 0 at two seeds; cell 2 is dataset 1. Each
        # cell splits and orders its replicate differently, so the replicate
        # is compared as a multiset of events, not row by row.
        np.testing.assert_array_equal(
            _sorted_rows(seen[0]["z"]), _sorted_rows(seen[1]["z"])
        )
        assert not np.array_equal(
            _sorted_rows(seen[0]["z"]), _sorted_rows(seen[2]["z"])
        )

    def test_the_split_varies_from_cell_to_cell(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Independent of both axes, so split and order variance cannot land in
        the dataset component, which is reported as the statistical uncertainty.
        """
        from deconvolve.uncertainty import run_cell

        seen: list[dict[str, Any]] = []
        self._stub(monkeypatch, seen)
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        for index in range(spec.n_cells):
            _ = run_cell(index, tmp_path, spec, n_samples=600, n_eval=100)

        from deconvolve.uncertainty.design import _read_cell

        split_seeds: list[int] = [
            int(_read_cell(cell_path(tmp_path, i))[1]["split_seed"])
            for i in range(spec.n_cells)
        ]
        assert len(set(split_seeds)) == spec.n_cells
        assert {r.tobytes() for r in seen[0]["train_z"]} != {
            r.tobytes() for r in seen[1]["train_z"]
        }

    def test_resampling_only_the_data_leaves_the_simulation_as_collected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deconvolve.uncertainty import run_cell

        seen: list[dict[str, Any]] = []
        self._stub(monkeypatch, seen)
        spec = DesignSpec(n_datasets=2, n_seeds=2)
        _ = run_cell(
            0, tmp_path, spec, n_samples=600, n_eval=100, resample=Resample.data
        )

        pool: Populations = reserve_evaluation_set(
            _populations(n=600), n_eval=100, seed=spec.data_seed
        ).pool
        np.testing.assert_array_equal(
            _sorted_rows(seen[0]["z"]), _sorted_rows(pool.mc.z)
        )

    def test_the_init_seed_advances_with_the_seed_index_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deconvolve.uncertainty import run_cell

        seen: list[dict[str, Any]] = []
        self._stub(monkeypatch, seen)
        spec = DesignSpec(n_datasets=2, n_seeds=3, init_seed=10)
        for index in range(spec.n_cells):
            _ = run_cell(index, tmp_path, spec, n_samples=600, n_eval=100)

        assert [record["seed"] for record in seen] == [10, 11, 12, 10, 11, 12]

    def test_the_hyperparameters_reach_training(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Defaults that differ from the paper's would not be a measurement of it."""
        from deconvolve.uncertainty import run_cell

        seen: list[dict[str, Any]] = []
        self._stub(monkeypatch, seen)
        _ = run_cell(
            0,
            tmp_path,
            DesignSpec(n_datasets=2, n_seeds=2),
            n_samples=600,
            n_eval=100,
            lr_g=1e-4,
            lambda_dispersion=0.02,
            n_epochs=7,
        )

        assert seen[0]["lr_g"] == pytest.approx(expected=1e-4)
        assert seen[0]["lambda_dispersion"] == pytest.approx(expected=0.02)
        assert seen[0]["n_epochs"] == 7


class TestCollect:
    """The report end to end, with the data source stubbed but nothing else."""

    @staticmethod
    def _design(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        spec: DesignSpec,
        *,
        n_eval: int = 300,
    ) -> None:
        import deconvolve.uncertainty.report as report_module

        pops: Populations = _populations(n=n_eval * 3)
        monkeypatch.setattr(
            target=report_module,
            name="base_populations",
            value=lambda *_a, **_k: (pops, 2),
        )
        rng: np.random.Generator = np.random.default_rng(seed=0)
        # One fixed direction in event space that every cell tilts along, so a
        # cell's weights actually move its unfolded mean rather than rescaling
        # every event by one constant (which no metric can see).
        tilt: NDArray[np.double] = rng.normal(size=n_eval)
        alpha: NDArray[np.double] = rng.normal(scale=0.30, size=spec.n_datasets)
        beta: NDArray[np.double] = rng.normal(scale=0.03, size=spec.n_seeds)
        for index in range(spec.n_cells):
            b, s = spec.cell_of_index(index)
            # A dataset effect an order of magnitude above the seed effect, so
            # a working decomposition has to attribute most of the variance to
            # the bootstrap axis.
            weights: NDArray[np.single] = np.exp(
                (alpha[b] + beta[s]) * tilt + 0.01 * rng.normal(size=n_eval),  # pyrefly: ignore[unknown-argument-type]
                dtype=np.single,
            )
            _write_cell(tmp_path, spec, index, weights)

    def test_it_writes_the_table_the_npz_and_the_figure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from deconvolve.uncertainty import collect

        spec = DesignSpec(n_datasets=4, n_seeds=3)
        self._design(tmp_path, monkeypatch, spec)

        summary: dict[str, dict[str, float]] = collect(tmp_path, spec, n_bins=5)

        assert (tmp_path / "variance.npz").exists()
        assert (tmp_path / "correlation.pdf").exists()
        assert set(summary) == {"m", "w"}
        assert summary["m"]["sd_total"] > 0.0
        # The design was built with the dataset effect dominant, so a report
        # that does not say so is not reading the grid it was handed.
        assert summary["m"]["var_data"] > 0.5 * summary["m"]["var_total"]
        assert summary["m"]["sd_naive_quadrature"] > summary["m"]["sd_total"]
        assert summary["m"]["var_evaluation"] > 0.0
        with np.load(tmp_path / "variance.npz") as saved:
            assert saved["m_cov_evaluation"].shape == saved["m_cov_data"].shape

    def test_the_recorded_json_carries_the_closure_floor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """So the measured off-diagonals can be read against what closure forces."""
        from deconvolve.uncertainty import collect

        spec = DesignSpec(n_datasets=4, n_seeds=3)
        self._design(tmp_path, monkeypatch, spec)
        _ = collect(tmp_path, spec, n_bins=5)

        recorded: dict[str, Any] = json.loads((tmp_path / "variance.json").read_text())

        assert recorded["multinomial_off_diagonal"]["m"] == pytest.approx(-0.25)
        assert recorded["n_bins"] == 5
        assert recorded["design"]["n_datasets"] == 4

    def test_a_metadata_mismatch_is_caught_rather_than_silently_misaligned(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The evaluation set is regenerated, so it has to regenerate identically."""
        from deconvolve.uncertainty import collect

        spec = DesignSpec(n_datasets=2, n_seeds=2)
        self._design(tmp_path, monkeypatch, spec, n_eval=50)
        # A cell whose recorded `n_eval` disagrees with the weights it holds:
        # regenerating from that metadata gives a set of the wrong size, and
        # lining the two up anyway would score every run on the wrong events.
        _write_cell(tmp_path, spec, index=0, weights=np.ones(shape=50), n_eval=40)

        with pytest.raises(
            expected_exception=ValueError, match="does not reproduce the design"
        ):
            _ = collect(tmp_path, spec, n_bins=3)

    def test_it_warns_when_there_are_too_few_replicates_for_the_binning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """B <= K makes every correlation +-1, which looks like a strong result."""
        from deconvolve.uncertainty import collect

        spec = DesignSpec(n_datasets=3, n_seeds=2)
        self._design(tmp_path, monkeypatch, spec)

        with caplog.at_level(level="WARNING"):
            _ = collect(tmp_path, spec, n_bins=8)

        assert "saturated by construction" in caplog.text

    def test_it_does_not_warn_when_the_grid_supports_the_binning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        from deconvolve.uncertainty import collect

        spec = DesignSpec(n_datasets=6, n_seeds=2)
        self._design(tmp_path, monkeypatch, spec)

        with caplog.at_level(level="WARNING"):
            _ = collect(tmp_path, spec, n_bins=3)

        assert "saturated by construction" not in caplog.text
