"""Tests for the paired hyperparameter comparison.

The arithmetic here is the whole point of the tool. Every negative result in
`benchmarks/README.md` under "What was ruled out" was an n=1 arm compared
against an n=1 baseline at a *different* initialization seed, against a
per-run SD of 7.2 on particle jet mass -- a test with no power to detect
anything it was looking for. This module exists to make that mistake hard:
runs are grouped by the knobs that actually vary, paired on the seed they
share, and reported with the spread they were measured against.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

from benchmarks.hparam_collect import (
    ArmSummary,
    RunRecord,
    arm_label,
    load_records,
    observable_deltas,
    paired_delta,
    particle_improvements,
    summarize_arm,
    tied_to_best,
    varying_keys,
)


def _write_run(
    root: Path, name: str, config: dict[str, Any], particle: dict[str, float]
) -> None:
    run_dir: Path = root / name
    artifacts_dir: Path = run_dir / "artifacts"
    artifacts_dir.mkdir(parents=True)
    _ = (run_dir / "config.json").write_text(json.dumps(config))
    metrics = {
        f"particle_{var}": {
            "wasserstein_improvement_pct": value,
            "jensenshannon_improvement_pct": 0.0,
        }
        for var, value in particle.items()
    }
    metrics["detector_m"] = {"wasserstein_improvement_pct": 95.0}
    _ = (artifacts_dir / "metrics.json").write_text(json.dumps(metrics))


class TestParticleImprovements:
    def test_reads_particle_entries_and_strips_the_prefix(self) -> None:
        metrics = {
            "detector_m": {"wasserstein_improvement_pct": 95.7},
            "particle_m": {"wasserstein_improvement_pct": 22.5},
            "particle_zg": {"wasserstein_improvement_pct": 78.2},
        }

        assert particle_improvements(metrics) == {"m": 22.5, "zg": 78.2}

    def test_ignores_a_run_whose_metrics_carry_no_particle_level(self) -> None:
        """A measurement without truth is not an error, just not comparable."""
        assert particle_improvements({"detector_m": {}}) == {}


class TestVaryingKeys:
    def test_names_only_the_knobs_that_differ(self) -> None:
        configs = [
            {"lr_g": 1e-4, "lr_d": 1e-4, "n_layers": 3, "seed": 0},
            {"lr_g": 3e-4, "lr_d": 1e-4, "n_layers": 3, "seed": 1},
        ]

        assert varying_keys(configs) == ("lr_g",)

    def test_excludes_the_seed_that_pairs_the_arms(self) -> None:
        configs = [{"lr_g": 1e-4, "seed": 0}, {"lr_g": 1e-4, "seed": 1}]

        assert varying_keys(configs) == ()

    def test_excludes_recorded_results_that_are_not_knobs(self) -> None:
        """`config.json` carries outcomes next to the settings.

        `best_epoch`, `mmd_test` and the kernel bandwidths differ in every run
        because they are what the run produced. Treating them as knobs would
        put every run in an arm of its own and there would be nothing to pair.
        """
        configs = [
            {"lr_g": 1e-4, "seed": 0, "best_epoch": 90, "mmd_test": 3.6e-4},
            {"lr_g": 1e-4, "seed": 1, "best_epoch": 8, "mmd_test": 5.1e-4},
        ]

        assert varying_keys(configs) == ()

    def test_handles_a_knob_whose_value_is_a_list(self) -> None:
        configs = [
            {"variables": ["m", "w"], "seed": 0},
            {"variables": ["m", "w", "zg"], "seed": 1},
        ]

        assert varying_keys(configs) == ("variables",)


class TestArmLabel:
    def test_names_an_arm_by_its_varying_knobs(self) -> None:
        config = {"lr_g": 0.0003, "lr_d": 1e-4, "seed": 5}

        assert arm_label(config, ("lr_g",)) == "lr_g=0.0003"

    def test_a_single_arm_with_nothing_varying_is_the_baseline(self) -> None:
        assert arm_label({"seed": 5}, ()) == "baseline"


class TestPairedDelta:
    def test_reports_the_mean_within_pair_difference(self) -> None:
        baseline = {0: 70.0, 1: 72.0, 2: 74.0}
        arm = {0: 73.0, 1: 73.0, 2: 77.0}

        result = paired_delta(baseline, arm)

        assert result.n == 3
        assert result.delta == pytest.approx(7.0 / 3.0)
        # sd of the within-pair differences (3, 1, 3) is 2/sqrt(3); the standard
        # error of their mean divides by sqrt(3) again.
        assert result.standard_error == pytest.approx(2.0 / 3.0)
        assert result.t_statistic == pytest.approx(3.5)
        assert 0.0 < result.p_value < 1.0

    def test_pairs_only_the_seeds_both_arms_ran(self) -> None:
        """An arm that lost a run to a node failure still pairs on the rest."""
        baseline = {0: 70.0, 1: 72.0, 2: 74.0, 3: 76.0}
        arm = {1: 73.0, 2: 74.0, 3: 78.0, 9: 99.0}

        result = paired_delta(baseline, arm)

        assert result.n == 3
        assert result.delta == pytest.approx(1.0)

    def test_refuses_to_report_a_spread_it_cannot_estimate(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            _ = paired_delta({0: 70.0}, {0: 73.0})


class TestSummarizeArm:
    def test_reports_the_seed_spread_the_arm_was_measured_against(self) -> None:
        """The noise floor is the headline number, not a footnote.

        With SD 7.2 on particle jet mass, a 5-point difference between two
        single runs is the expected spacing of two draws from one distribution.
        """
        records = [
            RunRecord(name="a", config={"seed": 0}, particle={"m": 20.0, "w": 90.0}),
            RunRecord(name="b", config={"seed": 1}, particle={"m": 30.0, "w": 94.0}),
            RunRecord(name="c", config={"seed": 2}, particle={"m": 40.0, "w": 92.0}),
        ]

        summary = summarize_arm("baseline", records)

        assert summary.n == 3
        # Per run: (20+90)/2 = 55, (30+94)/2 = 62, (40+92)/2 = 66.
        assert summary.mean == pytest.approx(61.0)
        assert summary.sd == pytest.approx(math.sqrt(31.0))
        assert summary.per_observable["m"] == pytest.approx(30.0)
        assert summary.per_observable["w"] == pytest.approx(92.0)

    def test_excluded_observables_leave_the_aggregate(self) -> None:
        """Jet mass is separately limited by a non-universal response.

        `benchmarks/README.md` §4 measures that, so a tuning run should be able
        to score without it and report it alongside instead.
        """
        records = [
            RunRecord(name="a", config={"seed": 0}, particle={"m": 20.0, "w": 90.0}),
            RunRecord(name="b", config={"seed": 1}, particle={"m": 40.0, "w": 92.0}),
        ]

        summary = summarize_arm("baseline", records, exclude=("m",))

        assert summary.mean == pytest.approx(91.0)
        # Still reported, just not scored.
        assert summary.per_observable["m"] == pytest.approx(30.0)


class TestLoadRecords:
    def test_reads_every_run_under_the_arm_directory(self, tmp_path: Path) -> None:
        _write_run(tmp_path, "lrg1e-4_seed00", {"lr_g": 1e-4, "seed": 0}, {"m": 22.5})
        _write_run(tmp_path, "lrg3e-4_seed00", {"lr_g": 3e-4, "seed": 0}, {"m": 31.1})

        records = load_records(tmp_path)

        assert [r.name for r in records] == ["lrg1e-4_seed00", "lrg3e-4_seed00"]
        assert records[0].particle == {"m": 22.5}

    def test_skips_a_run_that_died_before_writing_metrics(self, tmp_path: Path) -> None:
        """A crashed point must not sink the collection, only be absent."""
        _write_run(tmp_path, "good_seed00", {"lr_g": 1e-4, "seed": 0}, {"m": 22.5})
        crashed: Path = tmp_path / "crashed_seed01"
        crashed.mkdir()
        _ = (crashed / "config.json").write_text(json.dumps({"lr_g": 1e-4, "seed": 1}))

        records = load_records(tmp_path)

        assert [r.name for r in records] == ["good_seed00"]

    def test_reports_when_the_directory_holds_no_runs(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no runs"):
            _ = load_records(tmp_path)


class TestObservableDeltas:
    """The aggregate can hide an effect that lives in one observable.

    Twelve observables average a +13-point move in one of them down to +1.1 in
    the mean, which is inside the noise. Scoring on the aggregate is still
    right -- it is the low-variance response -- but it must not be the only
    thing tested.
    """

    def _arm(self, label: str, values: dict[int, dict[str, float]]) -> ArmSummary:
        return summarize_arm(
            label,
            [
                RunRecord(name=f"{label}_{seed}", config={"seed": seed}, particle=obs)
                for seed, obs in values.items()
            ],
        )

    def test_tests_each_observable_separately(self) -> None:
        baseline = self._arm(
            "a", {0: {"m": 20.0, "f": 40.0}, 1: {"m": 30.0, "f": 42.0}}
        )
        arm = self._arm("b", {0: {"m": 21.0, "f": 55.0}, 1: {"m": 29.0, "f": 53.0}})

        deltas = observable_deltas(baseline, arm)

        assert deltas["f"].delta == pytest.approx(13.0)
        assert deltas["m"].delta == pytest.approx(0.0)

    def test_skips_an_observable_only_one_arm_measured(self) -> None:
        """A 6-variable arm and a 12-variable one share only the six."""
        baseline = self._arm("a", {0: {"m": 20.0}, 1: {"m": 30.0}})
        arm = self._arm("b", {0: {"m": 21.0, "q": 9.0}, 1: {"m": 29.0, "q": 8.0}})

        assert set(observable_deltas(baseline, arm)) == {"m"}


class TestReplicateAxis:
    """Both seeds are nuisance axes; either can be the one replicated.

    `CLAUDE.md`'s Seeding section defines `seed` (initialization) and
    `data_seed` (shuffle, split, batch order) as the two independent randomness
    axes. Neither is ever a hyperparameter, so neither can define an arm --
    otherwise a sweep that replicates over `data_seed` puts every run in an arm
    of its own and there is nothing to pair.
    """

    def test_neither_seed_ever_defines_an_arm(self) -> None:
        configs = [
            {"lr_g": 1e-4, "seed": 0, "data_seed": 7},
            {"lr_g": 1e-4, "seed": 1, "data_seed": 9},
        ]

        assert varying_keys(configs) == ()

    def test_summarizes_against_the_replicated_axis(self) -> None:
        records = [
            RunRecord(
                name="a", config={"seed": 0, "data_seed": 7}, particle={"m": 20.0}
            ),
            RunRecord(
                name="b", config={"seed": 0, "data_seed": 9}, particle={"m": 40.0}
            ),
        ]

        summary = summarize_arm("baseline", records, pair_on="data_seed")

        assert set(summary.per_seed) == {7, 9}
        assert summary.per_seed[7] == pytest.approx(20.0)


def test_summarize_arm_refuses_a_pairing_key_that_does_not_index_the_runs() -> None:
    """A constant pairing key silently collapsed an arm to one run.

    `--pair-on seed` against a sweep that replicated over `data_seed` keyed
    every run in the arm to the same value, so the mean was one run's while `n`
    still said eight and the SD came back `nan`. Reporting a single run as an
    eight-run mean is the exact failure this tool exists to prevent, so it has
    to raise rather than print.
    """
    records = [
        RunRecord(name="a", config={"seed": 0, "data_seed": 7}, particle={"m": 20.0}),
        RunRecord(name="b", config={"seed": 0, "data_seed": 9}, particle={"m": 40.0}),
    ]

    with pytest.raises(ValueError, match="--pair-on"):
        _ = summarize_arm("baseline", records, pair_on="seed")


class TestEffectiveSampleSize:
    """ESS is the mechanism variable for the dispersion penalty.

    `benchmarks/README.md` §2 puts the oracle at 80.1% and RAN at 73.3%, so a
    penalty coefficient is tuned by watching ESS move toward the target, not by
    guessing a scale. `history.npz` records `val_ess` per epoch over the MMD
    subsample; the number that matters is the one at the selected epoch.
    """

    def test_reads_the_ess_at_the_selected_epoch(self, tmp_path: Path) -> None:
        run: Path = tmp_path / "lam0_seed00"
        artifacts: Path = run / "artifacts"
        artifacts.mkdir(parents=True)
        _ = (run / "config.json").write_text(
            json.dumps({"lr_g": 3e-5, "seed": 0, "best_epoch": 2})
        )
        _ = (artifacts / "metrics.json").write_text(
            json.dumps({"particle_m": {"wasserstein_improvement_pct": 22.5}})
        )
        np.savez(
            artifacts / "history.npz", val_ess=np.array([100.0, 200.0, 300.0, 400.0])
        )

        assert load_records(tmp_path)[0].ess == pytest.approx(300.0)

    def test_a_run_without_history_is_still_read(self, tmp_path: Path) -> None:
        """Older runs predate the field; they score, they just report no ESS."""
        run: Path = tmp_path / "lam0_seed00"
        artifacts: Path = run / "artifacts"
        artifacts.mkdir(parents=True)
        _ = (run / "config.json").write_text(json.dumps({"lr_g": 3e-5, "seed": 0}))
        _ = (artifacts / "metrics.json").write_text(
            json.dumps({"particle_m": {"wasserstein_improvement_pct": 22.5}})
        )

        assert math.isnan(load_records(tmp_path)[0].ess)

    def test_the_arm_reports_its_mean_ess(self) -> None:
        records = [
            RunRecord(name=f"r{s}", config={"seed": s}, particle={"m": 20.0}, ess=ess)
            for s, ess in enumerate([11000.0, 13000.0])
        ]

        assert summarize_arm("baseline", records).ess == pytest.approx(12000.0)


class TestCriterionAdmissibility:
    """An arm the selection criterion rejects is not a candidate setting.

    `mmd_test` is the detector-level number the run is *selected* on, recomputed
    on a held-out subsample. The estimator's resolution floor is ~2.5e-4 at
    `MMD_SUBSAMPLE = 16384`, so an arm within that of zero is one the criterion
    cannot distinguish from the best -- and anything it can distinguish is a
    setting a truth-free pipeline would refuse to ship, however well it scores
    against truth. The first dispersion sweep made this the primary question:
    lambda=0.1 reached jet mass 78.9% while sitting 30x outside the floor.
    """

    def test_reports_the_criterion_the_run_was_selected_on(self) -> None:
        records = [
            RunRecord(
                name=f"r{s}",
                config={"seed": s, "mmd_test": mmd},
                particle={"m": 20.0},
            )
            for s, mmd in enumerate([1e-4, 3e-4])
        ]

        assert summarize_arm("baseline", records).mmd_test == pytest.approx(2e-4)

    def test_a_run_that_never_recorded_it_reports_nothing(self) -> None:
        records = [RunRecord(name="r0", config={"seed": 0}, particle={"m": 20.0})]

        assert math.isnan(summarize_arm("baseline", records).mmd_test)


class TestTiedSet:
    """Admissibility is distance from the *best arm*, in units of the floor.

    The first version of this compared `|mmd_test|` against the floor directly,
    as though the floor were a hard threshold. It is not: it is the standard
    deviation of a zero-mean estimator, so an arm one floor from zero is one
    sigma from a perfect match -- entirely consistent with it. Measuring the
    floor exposed the error, because at the true 1.1e-4 even the shipped
    lambda=0 arm sits at 0.9x and the old rule called it inadmissible.

    The right question is `averaging.py`'s: which arms does the criterion fail
    to separate from the best one? That is a difference, not a magnitude, and
    two sigma rather than one.
    """

    def test_the_best_arm_is_always_in_its_own_tied_set(self) -> None:
        assert tied_to_best(1.0e-4, best=1.0e-4, floor=1.1e-4) is True

    def test_an_arm_within_two_floors_of_the_best_is_tied(self) -> None:
        """lambda=0.01 sits 0.75 floors from the best arm and is a candidate."""
        assert tied_to_best(1.81e-4, best=9.9e-5, floor=1.095e-4) is True

    def test_an_arm_the_criterion_separates_is_not(self) -> None:
        """lambda=0.03 sits 8 floors out; lambda=0.1 sits 67."""
        assert tied_to_best(9.78e-4, best=9.9e-5, floor=1.095e-4) is False

    def test_magnitude_alone_does_not_decide_it(self) -> None:
        """Two arms equally far from zero, one of them the best there is."""
        assert tied_to_best(3.0e-4, best=2.9e-4, floor=1.1e-4) is True

    def test_the_estimator_is_signed_so_distance_is_taken_on_the_signed_value(
        self,
    ) -> None:
        """-1e-4 and +1e-4 are two floors apart, not zero apart."""
        assert tied_to_best(1.2e-4, best=-1.2e-4, floor=1.1e-4) is False
