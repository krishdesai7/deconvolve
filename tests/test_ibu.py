from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from deconvolve.baselines import _shared as shared
from deconvolve.baselines import ibu
from deconvolve.coretypes import ZXY, DatasetSplits, Events, Populations, artifacts_dir
from deconvolve.data import ArrayDataset, DeconvolveDataset
from deconvolve.evaluation import evaluate
from deconvolve.training.models import build_generator

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from deconvolve.baselines import VariableUnfolding
    from numpy.typing import NDArray


def _split(z: list[list[float]], x: list[list[float]], y: list[int]) -> ArrayDataset:
    return ArrayDataset(
        ZXY(
            Events(np.asarray(z, dtype=np.single), np.asarray(x, dtype=np.single)),
            np.asarray(y, dtype=np.ubyte),
        ),
        batch_size=8,
    )


def _splits() -> DatasetSplits:
    return DatasetSplits(
        train=_split(
            [[0.2], [0.4], [1.2], [1.4]],
            [[-1.0], [0.5], [1.1], [3.0]],
            [0, 1, 0, 1],
        ),
        val=_split([[0.6], [1.6]], [[0.7], [1.7]], [0, 1]),
        test=_split(
            [[0.8], [1.8], [0.9], [1.9]],
            [[0.9], [2.5], [0.8], [2.2]],
            [0, 0, 1, 1],
        ),
    )


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "dataset": "gaussian",
        "dim": 2,
        "n_samples": 100,
        "batch_size": 16,
        "data_seed": 7,
    }
    config.update(overrides)
    return config


def test_parse_config_builds_gaussian_variable_names() -> None:
    parsed = shared.parse_run_config(_config())

    assert parsed.dataset == "gaussian"
    assert parsed.variable_names == ("dim_0", "dim_1")
    assert parsed.dim == 2
    assert parsed.data_seed == 7


def test_parse_config_requires_a_json_object() -> None:
    with pytest.raises(ValueError, match="JSON object"):
        _ = shared.parse_run_config(["not", "a", "mapping"])


def test_parse_config_validates_jet_variable_count() -> None:
    with pytest.raises(ValueError, match=r"variables.*dim"):
        _ = shared.parse_run_config(_config(dataset="jets", variables=["mass"], dim=2))


@pytest.mark.parametrize("key", ["dim", "n_samples", "batch_size"])
def test_parse_config_requires_positive_integer_fields(key: str) -> None:
    with pytest.raises(ValueError, match=key):
        _ = shared.parse_run_config(_config(**{key: 0}))


def test_parse_config_reports_missing_required_field() -> None:
    config = _config()
    del config["n_samples"]

    with pytest.raises(ValueError, match="n_samples"):
        _ = shared.parse_run_config(config)


def test_parse_config_rejects_non_integer_data_seed() -> None:
    with pytest.raises(TypeError, match="data_seed"):
        _ = shared.parse_run_config(_config(data_seed="seven"))


def test_parse_config_rejects_unknown_dataset() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        _ = shared.parse_run_config(_config(dataset="other"))


def test_parse_config_rejects_non_string_jet_variable() -> None:
    with pytest.raises(ValueError, match=r"variables.*strings"):
        _ = shared.parse_run_config(
            _config(dataset="jets", variables=["mass", 4], dim=2)
        )


def test_assign_bins_saturates_underflow_and_overflow() -> None:
    edges = np.array([0.0, 1.0, 2.0], dtype=np.single)
    values = np.array([-3.0, 0.0, 0.4, 1.0, 2.0, 8.0], dtype=np.single)

    indices = ibu._assign_bins(values, edges)

    np.testing.assert_array_equal(indices, [0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(ibu._bin_counts(indices, 2), [3.0, 3.0])


def test_saturating_counts_conserve_observed_population() -> None:
    edges = np.array([0.0, 1.0, 2.0], dtype=np.single)
    observed = np.array([-2.0, 0.5, 4.0], dtype=np.single)

    counts = ibu._bin_counts(ibu._assign_bins(observed, edges), 2)

    assert counts.sum() == observed.size
    np.testing.assert_array_equal(counts, [2.0, 1.0])


def test_prepare_data_names_response_and_test_populations() -> None:
    fit, test = shared.prepare_populations(_splits(), expected_dim=1)

    # Train (2 MC + 2 nature) plus val (1 + 1) --- the test split's 2 + 2 are
    # excluded, which is the whole point of the field being called `fit`.
    assert fit.mc.z.shape == (3, 1)
    assert fit.mc.x.shape == (3, 1)
    assert fit.data.shape == (3, 1)
    assert test.mc.z.shape == (2, 1)
    assert test.data.shape == (2, 1)


def test_the_fit_population_contains_no_test_event() -> None:
    """The claim the metrics rest on, checked by value rather than by count.

    Fitting on every split and scoring a subset of it is the convention in the
    unfolding literature; it is also how this code behaved until the populations
    were split in two, so it is worth a test that fails loudly if `Split.ALL`
    ever comes back.
    """
    splits = _splits()
    fit, test = shared.prepare_populations(splits, expected_dim=1)

    def z_values(pops: Populations) -> set[float]:
        both = np.concatenate([pops.mc.z.ravel(), pops.data.ravel()])
        return {float(v) for v in both}

    fitted, held_out = z_values(fit), z_values(test)

    assert fitted.isdisjoint(held_out)
    # And accounts for every row: fit is exactly train+val, test exactly test.
    assert len(fit.mc) + len(fit.data) == splits.train.size + splits.val.size
    assert len(test.mc) + len(test.data) == splits.test.size


def test_prepare_data_rejects_configured_dimension_mismatch() -> None:
    with pytest.raises(ValueError, match="expected dim=2"):
        _ = shared.prepare_populations(_splits(), expected_dim=2)


def test_prepare_data_rejects_nonfinite_values() -> None:
    splits = _splits()
    splits.train.data.x[0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        _ = shared.prepare_populations(splits, expected_dim=1)


def test_prepare_data_rejects_empty_test_data_population() -> None:
    splits = _splits()
    without_test_data = DatasetSplits(
        train=splits.train,
        val=splits.val,
        test=_split([[0.8], [1.8]], [[0.9], [2.5]], [0, 0]),
    )

    with pytest.raises(ValueError, match="test split: populations must be nonempty"):
        _ = shared.prepare_populations(without_test_data, expected_dim=1)


def test_unfolded_to_bin_weights_uses_explicit_zero_prior_semantics() -> None:
    weights = ibu._unfolded_to_bin_weights(
        np.array([4.0, 0.0], dtype=np.single),
        np.array([2.0, 0.0], dtype=np.single),
    )

    np.testing.assert_array_equal(weights, [2.0, 0.0])


def test_unfolded_to_bin_weights_rejects_mass_without_prior() -> None:
    with pytest.raises(ValueError, match="zero-prior"):
        _ = ibu._unfolded_to_bin_weights(
            np.array([4.0, 1.0], dtype=np.single),
            np.array([2.0, 0.0], dtype=np.single),
        )


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        (np.array([0.0, 0.0]), "strictly positive"),
        (np.array([-1.0, 2.0]), "nonnegative"),
        (np.array([1.0, np.inf]), "finite"),
        (np.array([1.0, np.nan]), "finite"),
    ],
)
def test_normalize_weights_rejects_invalid_vectors(
    weights: NDArray[np.single], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _ = ibu._normalize_weights(weights)


def test_normalize_weights_returns_mean_one() -> None:
    normalized = ibu._normalize_weights(np.array([1.0, 2.0, 3.0], dtype=np.single))

    assert normalized.mean() == pytest.approx(1.0)
    assert np.all(normalized >= 0)


def testunfold_variable_reports_insufficient_bins_as_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0], dtype=np.single),
    )

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=np.array([0.2, 0.8], dtype=np.single),
        mc_sim=np.array([0.3, 0.7], dtype=np.single),
        observed=np.array([-1.0, 2.0], dtype=np.single),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )

    assert result.outcome.status == "skipped"
    assert result.outcome.skip_reason == "fewer than two purity bins"
    assert result.reweighting is None
    np.testing.assert_array_equal(
        result.weights_for(np.array([0.4, 0.6], dtype=np.single)), [1.0, 1.0]
    )


def testunfold_variable_returns_safe_mean_one_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.single),
    )

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.single),
        mc_sim=np.array([-1.0, 0.7, 1.3, 3.0], dtype=np.single),
        observed=np.array([-2.0, 0.4, 1.4, 4.0], dtype=np.single),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )
    weights: NDArray[np.single] = result.weights_for(
        gen=np.array(object=[0.2, 1.8], dtype=np.single)
    )

    assert result.outcome.status == "completed"
    assert result.outcome.n_bins == 2
    assert np.all(a=np.isfinite(weights))
    assert np.all(a=weights >= 0)
    assert weights.mean() == pytest.approx(expected=1.0)


def test_unfolds_in_single_precision_end_to_end() -> None:
    """IBU runs at float32 to match the precision its published results use.

    Real purity binning on a realistic number of events, so the population
    count checks and the mean-one postcondition are exercised against float32
    arithmetic rather than assumed to survive it.
    """
    rng: np.random.Generator = np.random.default_rng(seed=0)
    n = 20_000
    mc_gen: NDArray[np.single] = rng.normal(size=n).astype(dtype=np.single)
    mc_sim: NDArray[np.single] = (
        mc_gen + 0.3 * rng.normal(size=n).astype(dtype=np.single)
    ).astype(np.single)
    observed: NDArray[np.single] = (
        0.5 + rng.normal(size=n).astype(dtype=np.single)
    ).astype(dtype=np.single)

    result: VariableUnfolding = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=mc_gen,
        mc_sim=mc_sim,
        observed=observed,
        n_iterations=4,
        purity_threshold=0.5,
    )

    assert result.outcome.status == "completed"
    assert result.reweighting is not None
    assert result.reweighting.bin_weights.dtype == np.single
    assert result.reweighting.edges.dtype == np.single

    weights = result.weights_for(mc_gen)
    assert weights.dtype == np.single
    assert weights.mean(dtype=np.single) == pytest.approx(1.0, rel=1e-5)


def test_unfolding_applies_to_a_sample_it_was_not_fit_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The learned object is per-bin, so the sample it scores is a free choice."""
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.single),
    )

    result = ibu.unfold_variable(
        variable_name="dim_0",
        mc_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.single),
        mc_sim=np.array([-1.0, 0.7, 1.3, 3.0], dtype=np.single),
        observed=np.array([-2.0, 0.4, 1.4, 4.0], dtype=np.single),
        n_iterations=2,
        purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
    )

    assert result.reweighting is not None
    assert result.reweighting.bin_weights.size == 2
    # Any number of events, in any order, so long as they are the same variable.
    for size in (2, 7, 50):
        gen = np.linspace(0.0, 2.0, size, dtype=np.single)
        assert result.weights_for(gen).shape == (size,)


@pytest.mark.parametrize(
    ("corrupt_call", "population"),
    [(1, "prior"), (2, "observed")],
)
def testunfold_variable_reports_population_count_mismatch(
    monkeypatch: pytest.MonkeyPatch, corrupt_call: int, population: str
) -> None:
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0, 2.0], dtype=np.single),
    )
    real_bin_counts: Callable[[NDArray[np.intp], int], NDArray[np.intp]] = (
        ibu._bin_counts
    )
    call_count = 0

    def dropping_bin_counts(indices: NDArray[np.intp], n_bins: int) -> NDArray[np.intp]:
        nonlocal call_count
        call_count += 1
        counts = real_bin_counts(indices, n_bins)
        if call_count == corrupt_call:
            # numpy stubs type a scalar element access as Any
            counts[-1] -= 1  # pyrefly: ignore[unknown-argument-type]
        return counts

    monkeypatch.setattr(ibu, "_bin_counts", dropping_bin_counts)

    with pytest.raises(ValueError, match=rf"{population}.*3.*4"):
        _ = ibu.unfold_variable(
            variable_name="dim_0",
            mc_gen=np.array([0.2, 0.8, 1.2, 1.8], dtype=np.single),
            mc_sim=np.array([0.3, 0.7, 1.3, 1.7], dtype=np.single),
            observed=np.array([0.1, 0.9, 1.1, 1.9], dtype=np.single),
            n_iterations=2,
            purity_threshold=ibu.DEFAULT_PURITY_THRESHOLD,
        )


def test_evaluate_dimension_accepts_one_dimensional_arrays() -> None:
    record = shared.evaluate_dimension(
        reference=np.array([0.0, 1.0, 2.0], dtype=np.single),
        comparison=np.array([0.0, 1.5, 3.0], dtype=np.single),
        weights=np.ones(3, dtype=np.single),
    )

    assert set(record) == {
        "wasserstein_before",
        "wasserstein_after",
        "wasserstein_improvement_pct",
        "jensenshannon_before",
        "jensenshannon_after",
        "jensenshannon_improvement_pct",
        "triangular_before",
        "triangular_after",
        "triangular_improvement_pct",
    }
    assert all(np.isfinite(cast("float", value)) for value in record.values())


def test_run_and_evaluate_returns_named_aligned_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shared, "_load_splits", lambda **_kwargs: _splits())
    monkeypatch.setattr(
        ibu,
        "_purity_bins",
        lambda *_args, **_kwargs: np.array([0.0, 1.0], dtype=np.single),
    )
    config = shared.parse_run_config(_config(dim=1, gaussian_params={"dim": 1}))

    result = ibu._run_and_evaluate(config, n_iterations=2)

    assert isinstance(result, ibu.IBUResult)
    assert result.variable_names == ("dim_0",)
    assert result.weights.shape == (1, 2)
    # Cast at IBU's boundary: the splits above are the float64 RAN trains on.
    assert result.weights.dtype == np.single
    assert len(result.outcomes) == 1
    assert result.outcomes[0].status == "skipped"
    assert set(result.metrics) == {"detector_dim_0", "particle_dim_0"}
    for record in result.metrics.values():
        assert record["wasserstein_after"] == pytest.approx(
            record["wasserstein_before"]
        )


@pytest.mark.parametrize(
    ("n_iterations", "purity_threshold", "message"),
    [
        (0, ibu.DEFAULT_PURITY_THRESHOLD, "n_iterations"),
        (True, ibu.DEFAULT_PURITY_THRESHOLD, "n_iterations"),
        (1, np.double(np.nan), "purity_threshold"),
        (1, np.double(-0.1), "purity_threshold"),
        (1, np.single(1.1), "purity_threshold"),
    ],
)
def test_run_and_evaluate_validates_controls_before_loading_data(
    monkeypatch: pytest.MonkeyPatch,
    n_iterations: int,
    purity_threshold: np.double,
    message: str,
) -> None:
    monkeypatch.setattr(
        shared,
        "_load_splits",
        lambda **_kwargs: pytest.fail("loaded data before validating controls"),
    )
    config = shared.parse_run_config(_config())

    with pytest.raises(ValueError, match=message):
        _ = ibu._run_and_evaluate(
            config,
            n_iterations=n_iterations,
            purity_threshold=purity_threshold,
        )


def _jets_like_splits(dim: int, n: int = 64, seed: int = 51) -> DatasetSplits:
    """A `Populations` with truth, shaped like a jets run but requiring no
    Zenodo download -- see `TestParticleCurve._truthless_splits` in
    `tests/test_workflow.py` for the same construction without truth.
    """
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    z_gen: NDArray[np.single] = rng.normal(size=(n, dim)).astype(dtype=np.single)
    x_sim: NDArray[np.single] = (z_gen + rng.normal(0, 0.4, size=(n, dim))).astype(
        dtype=np.single
    )
    x_data: NDArray[np.single] = rng.normal(size=(n, dim)).astype(dtype=np.single)
    truth: NDArray[np.single] = (z_gen + rng.normal(0, 0.1, size=(n, dim))).astype(
        dtype=np.single
    )
    pops: Populations = Populations.create(
        mc=Events(z_gen, x_sim), data=x_data, truth=truth
    )
    return DeconvolveDataset(batch_size=8, seed=seed).splits_from_data(
        pops.interleave()
    )


def _skip_forcing_splits(dim: int = 2, n: int = 64, seed: int = 51) -> DatasetSplits:
    """Like `_jets_like_splits`, but the second gen column is a constant.

    A constant `mc_gen` column makes `_purity_bins` return a single edge (its
    `while` loop never runs, since `edges[0] == gen.max()` from the start), so
    `unfold_variable` genuinely takes its `n_bins < 2` skip branch -- no
    monkeypatch of `_purity_bins`/`unfold_variable` involved.
    """
    rng: np.random.Generator = np.random.default_rng(seed=seed)
    z_gen: NDArray[np.single] = rng.normal(size=(n, dim)).astype(dtype=np.single)
    z_gen[:, 1] = np.single(5.0)
    x_sim: NDArray[np.single] = (z_gen + rng.normal(0, 0.4, size=(n, dim))).astype(
        dtype=np.single
    )
    x_data: NDArray[np.single] = rng.normal(size=(n, dim)).astype(dtype=np.single)
    truth: NDArray[np.single] = (z_gen + rng.normal(0, 0.1, size=(n, dim))).astype(
        dtype=np.single
    )
    pops: Populations = Populations.create(
        mc=Events(z_gen, x_sim), data=x_data, truth=truth
    )
    return DeconvolveDataset(batch_size=8, seed=seed).splits_from_data(
        pops.interleave()
    )


def _run_with_ibu(
    tmp_path: Path,
    variables: tuple[str, ...],
    monkeypatch: pytest.MonkeyPatch,
    splits: DatasetSplits | None = None,
) -> Path:
    """A run directory with real `metrics.json`/`metrics_ibu.json` output, with
    the dataset load stubbed so the test needs no jets download.
    """
    dim: int = len(variables)
    if splits is None:
        splits = _jets_like_splits(dim)
    monkeypatch.setattr(evaluate, "_load_splits", lambda *_a, **_k: splits)
    monkeypatch.setattr(shared, "_load_splits", lambda *_a, **_k: splits)

    run_dir: Path = tmp_path / "run"
    config: dict[str, Any] = {
        "dataset": "jets",
        "dim": dim,
        "variables": list(variables),
        "n_samples": 64,
        "batch_size": 8,
        "data_seed": 51,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    _ = (run_dir / "config.json").write_text(data=json.dumps(config))

    generator = build_generator(dim=dim, hidden_units=4, n_layers=1)
    generator.save(artifacts_dir(run_dir) / "generator.keras")

    _ = evaluate.evaluate_run(run_dir)
    _ = ibu.evaluate_single(run_dir)
    return run_dir


def test_ibu_and_ran_agree_on_metric_key_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same nominal format, two orders, is how a positional zip goes wrong."""
    run_dir: Path = _run_with_ibu(tmp_path, ("m", "w"), monkeypatch)

    ran_keys = list(json.loads((run_dir / "artifacts/metrics.json").read_text()))
    ibu_keys = list(json.loads((run_dir / "artifacts/metrics_ibu.json").read_text()))
    assert ibu_keys == ran_keys
    assert ibu_keys == ["detector_m", "detector_w", "particle_m", "particle_w"]


def test_ibu_records_which_variables_it_gave_up_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 0% improvement from a skipped variable must not read as a measurement."""
    run_dir: Path = _run_with_ibu(tmp_path, ("m", "w"), monkeypatch)

    outcomes: list[dict[str, Any]] = json.loads(
        (run_dir / "artifacts/ibu_outcomes.json").read_text()
    )
    assert {o["variable_name"] for o in outcomes} == {"m", "w"}
    assert all(o["status"] in {"completed", "skipped"} for o in outcomes)
    assert all(
        set(o) == {"variable_name", "status", "n_bins", "skip_reason"} for o in outcomes
    )


def test_ibu_persists_a_skipped_variables_reason_and_bin_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch the outcomes file exists for, genuinely triggered.

    `skip_reason` and `n_bins` are what Task 12 renders for a variable IBU
    refused to unfold, and this is also the branch that fires on the
    project's real reference run, for `z_g`.
    """
    variables = ("m", "w")
    run_dir: Path = _run_with_ibu(
        tmp_path,
        variables,
        monkeypatch,
        splits=_skip_forcing_splits(dim=len(variables)),
    )

    outcomes: list[dict[str, Any]] = json.loads(
        (run_dir / "artifacts/ibu_outcomes.json").read_text()
    )
    by_name: dict[str, dict[str, Any]] = {o["variable_name"]: o for o in outcomes}

    assert by_name["m"]["status"] == "completed"

    skipped: dict[str, Any] = by_name["w"]
    assert skipped["status"] == "skipped"
    assert isinstance(skipped["skip_reason"], str)
    assert skipped["skip_reason"]
    assert isinstance(skipped["n_bins"], int)
    assert type(skipped["n_bins"]) is int  # not a numpy integer subclass


def test_ibu_cache_recomputes_when_outcomes_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`metrics_ibu.json` alone is an incomplete result, not a cache hit.

    A directory holding it without `ibu_outcomes.json` beside it -- an older
    run, or one interrupted between the two writes -- must be recomputed
    rather than treated as done, or Task 12 never gets an outcomes file short
    of `--force`.
    """
    run_dir: Path = _run_with_ibu(tmp_path, ("m", "w"), monkeypatch)
    metrics_path: Path = run_dir / "artifacts/metrics_ibu.json"
    outcomes_path: Path = run_dir / "artifacts/ibu_outcomes.json"
    assert outcomes_path.exists()

    outcomes_path.unlink()
    _ = metrics_path.write_text(data="{}")  # simulates the incomplete state

    _ = ibu.evaluate_single(run_dir)

    assert outcomes_path.exists()
    assert json.loads(metrics_path.read_text()) != {}
