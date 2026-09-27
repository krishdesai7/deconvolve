from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, assert_type, cast

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

    from deconvolve.coretypes import EventArray

import numpy as np
import pytest
import yaml
from deconvolve.coretypes import (
    EVENT_DTYPE,
    TRUTH_SENTINEL,
    ZXY,
    DatasetSplits,
    Events,
    GaussianConfig,
    Populations,
    Split,
    constants,
)
from deconvolve.data import DeconvolveDataset


def _write_config(params: dict[str, Any], tmp_path: Path) -> Path:
    """Write a config in the YAML *input* format: mu_* plus sigma_* keys.

    `parse_gaussian_config` promotes the sigmas to the covariance matrices of a
    `GaussianConfig`, so that type is its output, never its input. The `params=`
    argument below is the one that takes a `GaussianConfig` -- that is the
    already-promoted path a reloaded run uses.
    """
    p: Path = tmp_path / "config.yaml"
    _ = p.write_text(yaml.dump(params))
    return p


class TestGenerateGaussianDataset:
    """Test multivariate Gaussian dataset generation."""

    @pytest.mark.writes_default_cache
    def test_1d_uncorrelated(self, tmp_path: Path) -> None:
        """1D scalar sigma should produce valid splits."""
        cfg = {
            "mu_gen": [0.5],
            "mu_true": [0.0],
            "sigma_gen": 0.9,
            "sigma_true": 1.0,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        ds = DeconvolveDataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=1000)
        assert splits.train is not None
        assert splits.val is not None
        assert splits.test is not None

    @pytest.mark.writes_default_cache
    def test_2d_correlated_shapes(self, tmp_path: Path) -> None:
        """2D with full covariance should produce correct shapes."""
        cfg = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [[1.0, -0.54], [-0.54, 2.25]],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        ds = DeconvolveDataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=2000)
        events = splits.test.as_arrays().events
        assert events.z.shape[1] == 2
        assert events.x.shape[1] == 2

    @pytest.mark.writes_default_cache
    def test_params_dict_interface(self) -> None:
        """Passing promoted params directly should work (for --load_run)."""
        params = GaussianConfig(
            dim=1,
            mu_gen=np.array([0.0]),
            mu_true=np.array([0.5]),
            cov_gen=np.array([[1.0]]),
            cov_true=np.array([[0.9]]),
            cov_detector=np.array([[0.5]]),
        )
        ds = DeconvolveDataset(batch_size=64, seed=42)
        splits = ds.generate_gaussian_dataset(params=params, n_samples=1000)
        assert splits.train is not None

    def test_dtype_controls_generated_array_runtime_type(self, tmp_path: Path) -> None:
        params = GaussianConfig(
            dim=1,
            mu_gen=np.array([0.0]),
            mu_true=np.array([0.5]),
            cov_gen=np.array([[1.0]]),
            cov_true=np.array([[0.9]]),
            cov_detector=np.array([[0.5]]),
        )
        ds = DeconvolveDataset(cache_dir=tmp_path)
        assert_type(ds, DeconvolveDataset)

        splits = ds.generate_gaussian_dataset(params=params, n_samples=100)

        assert ds.dtype == np.single
        assert splits.select(Split.ALL).z.dtype == np.single
        assert splits.select(Split.ALL).x.dtype == np.single

    def test_both_config_and_params_raises(self, tmp_path: Path) -> None:
        """Providing both config_path and params should error."""
        cfg = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        params = GaussianConfig(
            dim=1,
            mu_gen=np.array([0.0]),
            mu_true=np.array([0.5]),
            cov_gen=np.array([[1.0]]),
            cov_true=np.array([[0.81]]),
            cov_detector=np.array([[0.25]]),
        )
        ds = DeconvolveDataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            _ = ds.generate_gaussian_dataset(
                config_path=path, params=params, n_samples=100
            )

    def test_neither_config_nor_params_raises(self) -> None:
        """Providing neither config_path nor params should error."""
        ds = DeconvolveDataset(batch_size=64, seed=42)
        with pytest.raises(ValueError, match="Exactly one"):
            _ = ds.generate_gaussian_dataset(n_samples=100)

    def test_caching(self, tmp_path: Path) -> None:
        """Second call with same config should hit cache."""
        cfg = {
            "mu_gen": [0.0],
            "mu_true": [0.5],
            "sigma_gen": 1.0,
            "sigma_true": 0.9,
            "sigma_detector": 0.5,
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"
        ds = DeconvolveDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        _ = ds.generate_gaussian_dataset(config_path=path, n_samples=500)
        cache_files = list(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files) == 1
        ds2 = DeconvolveDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        _ = ds2.generate_gaussian_dataset(config_path=path, n_samples=500)

    @pytest.mark.writes_default_cache
    def test_smearing_preserves_event_coupling(self, tmp_path: Path) -> None:
        """Detector-level values should be correlated with particle-level."""
        cfg = {
            "mu_gen": [0.0, 0.0],
            "mu_true": [0.0, 0.0],
            "sigma_gen": [1.0, 1.0],
            "sigma_true": [1.0, 1.0],
            "sigma_detector": [0.1, 0.1],
        }
        path = _write_config(cfg, tmp_path)
        ds = DeconvolveDataset(batch_size=10000, seed=42)
        splits = ds.generate_gaussian_dataset(config_path=path, n_samples=10000)
        events = splits.test.as_arrays().events
        for d in range(2):
            corr = np.corrcoef(events.z[:, d], events.x[:, d])[0, 1]
            assert corr > 0.95, f"dim {d}: corr={corr}, expected >0.95"

    def test_yaml_and_params_share_cache(self, tmp_path: Path) -> None:
        """A YAML config and the params it promotes to must share a cache key.

        `sigma_gen: [1.0, 1.5]` promotes to diag(1.0, 2.25) and
        `sigma_detector: [0.5, 0.8]` to diag(0.25, 0.64) -- exactly the
        covariances the reloaded `GaussianConfig` below carries. Hashing the
        promoted matrices, not the raw sigma form, is what makes a reloaded run
        hit the cache its original YAML filled.
        """
        cfg = {
            "mu_gen": [0.0, 1.0],
            "mu_true": [0.2, 0.8],
            "sigma_gen": [1.0, 1.5],
            "sigma_true": [[0.81, -0.5], [-0.5, 1.69]],
            "sigma_detector": [0.5, 0.8],
        }
        path = _write_config(cfg, tmp_path)
        cache_dir = tmp_path / "cache"

        ds1 = DeconvolveDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        _ = ds1.generate_gaussian_dataset(config_path=path, n_samples=500)
        cache_files_after_yaml = set(cache_dir.glob("gaussian_*.npz"))
        assert len(cache_files_after_yaml) == 1

        reload_params = GaussianConfig(
            dim=2,
            mu_gen=np.array([0.0, 1.0]),
            mu_true=np.array([0.2, 0.8]),
            cov_gen=np.array([[1.0, 0.0], [0.0, 2.25]]),
            cov_true=np.array([[0.81, -0.5], [-0.5, 1.69]]),
            cov_detector=np.array([[0.25, 0.0], [0.0, 0.64]]),
        )
        ds2 = DeconvolveDataset(batch_size=64, seed=42, cache_dir=cache_dir)
        _ = ds2.generate_gaussian_dataset(params=reload_params, n_samples=500)

        cache_files_after_params = set(cache_dir.glob("gaussian_*.npz"))
        assert cache_files_after_params == cache_files_after_yaml


def test_splits_from_data_builds_three_nonempty_splits() -> None:
    n = 200
    z = np.random.default_rng(0).normal(size=(2 * n, 1)).astype(np.single)
    x = np.random.default_rng(1).normal(size=(2 * n, 1)).astype(np.single)
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])

    splits = DeconvolveDataset(batch_size=32).splits_from_data(ZXY(Events(z, x), y))

    for ds in (splits.train, splits.val, splits.test):
        data = ds.as_arrays()
        assert data.z.shape[-1] == 1
        assert data.y.shape[0] > 0


def _toy_splits(n: int = 200, batch_size: int = 32, **kwargs: Any) -> DatasetSplits:
    z = np.arange(2 * n, dtype=np.single).reshape(-1, 1)
    x = -z
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
    return DeconvolveDataset(batch_size=batch_size, **kwargs).splits_from_data(
        ZXY(Events(z, x), y)
    )


class TestArrayDataset:
    """Behaviour the tf.data pipeline used to provide."""

    def test_splits_partition_events_without_overlap(self) -> None:
        """Every event lands in exactly one split."""
        splits = _toy_splits(n=200)
        ids = [set(ds.as_arrays().z.ravel().tolist()) for ds in splits]
        assert sum(len(s) for s in ids) == 400
        assert set.union(*ids) == set(range(400))
        assert not (ids[0] & ids[1])
        assert not (ids[0] & ids[2])
        assert not (ids[1] & ids[2])

    def test_default_split_fractions(self) -> None:
        splits = _toy_splits(n=500)
        assert splits.test.size == 200  # 20% of 1000
        assert splits.val.size == 100  # 10% of 1000
        assert splits.train.size == 700

    def test_shuffle_interleaves_classes(self) -> None:
        """Splits must not be single-class: data and MC arrive stacked."""
        for ds in _toy_splits(n=500):
            frac = float(ds.as_arrays().y.mean())
            assert 0.4 < frac < 0.6, f"class fraction {frac} — split is not mixed"

    def test_z_and_x_stay_paired(self) -> None:
        """Shuffling and splitting must not decouple particle/detector rows."""
        for ds in _toy_splits(n=200):
            events = ds.as_arrays().events
            np.testing.assert_array_equal(events.x, -events.z)

    def test_batch_count_covers_the_split(self) -> None:
        """`__len__` counts a short trailing batch, which the device layer drops.

        The container reports how the split divides; whether the remainder is
        used is `train_indices`' call, not this object's.
        """
        train = _toy_splits(n=200, batch_size=32).train
        assert train.size % 32 != 0, "test needs a ragged final batch"
        assert len(train) == -(-train.size // 32)

    def test_same_seed_gives_same_split(self) -> None:
        a, b = _toy_splits(n=200, seed=7), _toy_splits(n=200, seed=7)
        np.testing.assert_array_equal(a.test.as_arrays().z, b.test.as_arrays().z)

    def test_too_few_events_to_split_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one event"):
            _ = _toy_splits(n=2)


class TestLabelledAndPhysicsForms:
    """The two representations and the conversions between them."""

    @staticmethod
    def _populations(n: int = 4) -> Populations:
        truth = np.arange(n, dtype=np.single).reshape(-1, 1)
        gen = np.arange(n, 2 * n, dtype=np.single).reshape(-1, 1)
        return Populations(mc=Events(gen, -gen), data=-truth, truth=truth)

    def test_events_reject_unaligned_rows(self) -> None:
        with pytest.raises(ValueError, match="row-aligned"):
            _ = Events(
                np.zeros((5, 1), dtype=np.single), np.zeros((4, 1), dtype=np.single)
            )

    def test_populations_reject_a_float64_source(self) -> None:
        """A source that forgot to narrow is caught where its events enter."""
        valid = self._populations()
        with pytest.raises(ValueError, match="data is float64"):
            _ = Populations(
                mc=valid.mc,
                data=cast("EventArray", valid.data.astype(np.double)),
                truth=valid.truth,
            )

    def test_populations_reject_a_flat_array(self) -> None:
        """One feature is a column, not a vector: (n, 1), never (n,)."""
        valid = self._populations()
        with pytest.raises(ValueError, match="truth has shape"):
            _ = Populations(mc=valid.mc, data=valid.data, truth=valid.truth.ravel())

    def test_populations_reject_a_feature_count_mismatch(self) -> None:
        """Truth describes the same features as the generated sample."""
        valid = self._populations()
        with pytest.raises(ValueError, match="truth has 2 features"):
            _ = Populations(
                mc=valid.mc,
                data=valid.data,
                truth=np.hstack([valid.truth, valid.truth]),
            )

    def test_labels_must_be_zero_or_one(self) -> None:
        events = Events(
            np.zeros((3, 1), dtype=np.single), np.zeros((3, 1), dtype=np.single)
        )
        with pytest.raises(ValueError, match=r"zero \(MC\) or one"):
            _ = ZXY(events, np.array([0, 1, 2], dtype=np.ubyte))

    def test_one_label_per_event(self) -> None:
        events = Events(
            np.zeros((3, 1), dtype=np.single), np.zeros((3, 1), dtype=np.single)
        )
        with pytest.raises(ValueError, match="one label per event"):
            _ = ZXY(events, np.ones(2, dtype=np.ubyte))

    def test_partition_recovers_the_populations_it_was_built_from(self) -> None:
        original = self._populations()

        recovered = original.interleave().partition()

        np.testing.assert_array_equal(recovered.truth, original.truth)
        np.testing.assert_array_equal(recovered.data, original.data)
        np.testing.assert_array_equal(recovered.mc.z, original.mc.z)
        np.testing.assert_array_equal(recovered.mc.x, original.mc.x)

    def test_interleave_labels_nature_one_and_mc_zero(self) -> None:
        labelled = self._populations(n=3).interleave()

        np.testing.assert_array_equal(labelled.y, [1, 1, 1, 0, 0, 0])
        assert len(labelled) == 6

    def test_create_stands_in_a_sentinel_for_absent_truth(self) -> None:
        mc = Events(
            np.zeros((3, 2), dtype=np.single), np.zeros((3, 2), dtype=np.single)
        )

        measured = Populations.create(mc=mc, data=np.ones((5, 2), dtype=np.single))

        assert not measured.has_truth
        assert measured.truth.shape == (5, 2)
        np.testing.assert_array_equal(measured.truth, TRUTH_SENTINEL)

    def test_sentinel_is_exact_in_the_pinned_dtype(self) -> None:
        """`has_truth` compares against TRUTH_SENTINEL by equality.

        That only works while the sentinel is exactly representable. -2**15 is,
        in every IEEE binary format, which is why it was chosen over a value
        that merely looks far off-manifold --- and it is what lets the pipeline
        pin itself to float32 without `has_truth` becoming a rounding question.
        """
        assert np.single(TRUTH_SENTINEL) == TRUTH_SENTINEL
        assert np.double(np.single(TRUTH_SENTINEL)) == TRUTH_SENTINEL

        mc = Events(
            np.zeros((3, 2), dtype=EVENT_DTYPE), np.zeros((3, 2), dtype=EVENT_DTYPE)
        )
        measured = Populations.create(mc=mc, data=np.ones((5, 2), dtype=EVENT_DTYPE))
        assert measured.truth.dtype == EVENT_DTYPE
        assert not measured.has_truth
        assert self._populations().has_truth

    def test_require_truth_refuses_the_sentinel(self) -> None:
        mc = Events(
            np.zeros((3, 2), dtype=np.single), np.zeros((3, 2), dtype=np.single)
        )

        measured = Populations.create(mc=mc, data=np.ones((5, 2), dtype=np.single))

        with pytest.raises(ValueError, match="no particle-level truth"):
            _ = measured.require_truth()

    def test_require_truth_returns_real_answers(self) -> None:
        original = self._populations()

        np.testing.assert_array_equal(original.require_truth(), original.truth)

    def test_create_keeps_truth_when_given(self) -> None:
        original = self._populations()

        rebuilt = Populations.create(original.mc, original.data, original.truth)

        assert rebuilt.has_truth
        np.testing.assert_array_equal(rebuilt.truth, original.truth)

    def test_absent_truth_reaches_z_as_a_finite_number(self) -> None:
        """`interleave` puts truth in the nature rows of z, so g will see this.

        Training survives that only because the stand-in is an ordinary number
        -- see `test_a_missing_particle_level_cannot_poison_the_batch`.
        """
        mc = Events(
            np.zeros((3, 1), dtype=np.single), np.zeros((3, 1), dtype=np.single)
        )

        labelled = Populations.create(
            mc=mc, data=np.ones((2, 1), dtype=np.single)
        ).interleave()

        assert np.all(np.isfinite(labelled.z))
        np.testing.assert_array_equal(labelled.z[labelled.y == 1], TRUTH_SENTINEL)

    def test_partition_rejects_a_single_class_sample(self) -> None:
        events = Events(
            np.zeros((3, 1), dtype=np.single), np.zeros((3, 1), dtype=np.single)
        )

        with pytest.raises(ValueError, match="nonempty"):
            _ = ZXY(events, np.zeros(3, dtype=np.ubyte)).partition()

    def test_select_concatenates_only_the_requested_splits(self) -> None:
        splits = _toy_splits(n=200)

        assert len(splits.select(Split.TEST)) == splits.test.size
        assert len(splits.select(Split.TRAIN | Split.VAL)) == (
            splits.train.size + splits.val.size
        )
        assert len(splits.select(Split.ALL)) == 400

    def test_select_keeps_events_row_aligned_across_splits(self) -> None:
        everything = _toy_splits(n=200).select(Split.ALL)

        np.testing.assert_array_equal(everything.x, -everything.z)


class TestCacheDirIsRelocatable:
    """`DECONVOLVE_CACHE_DIR` moves the whole regenerable tree at once.

    The default is wrong in both directions on a cluster: `$HOME` is quota'd and
    shared across nodes, and a checkout's `.cache/` may sit on a filesystem the
    compute nodes cannot write to. One variable moves the Zenodo npz caches and
    the XLA compilation cache together -- which is the reason they share a root
    rather than each carrying their own knob.

    These reload the module because `CACHE_DIR` is resolved at import; that is
    deliberate, since the `cache_dir=` defaults throughout `deconvolve.data` bind to it
    at import too.
    """

    @pytest.fixture(autouse=True)
    def _restore(self) -> Iterator[None]:
        yield
        _ = importlib.reload(constants)

    @staticmethod
    def _with(env: str | None, monkeypatch: pytest.MonkeyPatch) -> Path:
        if env is None:
            monkeypatch.delenv(constants.CACHE_ENV_VAR, raising=False)
        else:
            monkeypatch.setenv(constants.CACHE_ENV_VAR, env)
        return cast("Path", importlib.reload(constants).CACHE_DIR)

    def test_unset_gives_the_project_local_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert self._with(None, monkeypatch) == Path(".cache")

    def test_both_caches_move_together(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        root: Path = tmp_path / "scratch"
        monkeypatch.setenv(constants.CACHE_ENV_VAR, str(root))
        reloaded = importlib.reload(constants)

        assert root == reloaded.CACHE_DIR
        assert root / "jax" == reloaded.COMPILE_CACHE_DIR

    def test_a_tilde_expands(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A SLURM `--export` carries the string through without a shell to
        expand it, so `~/ran-cache` would otherwise become a literal directory
        named `~`."""
        expected: Path = Path.home() / "ran-cache"

        assert self._with("~/ran-cache", monkeypatch) == expected

    def test_empty_falls_back_rather_than_meaning_cwd(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unset variable forwarded by `--export` arrives as empty, not
        absent. `Path("")` is `Path(".")`, so taking it literally would scatter
        the cache into whatever directory each job happened to start in."""
        assert self._with("", monkeypatch) == Path(".cache")
