"""Tests for the optional timing layer.

Two properties matter more than the numbers themselves. The layer must be a
true no-op when `DECONVOLVE_TIMING` is unset --- a run that is not being profiled
should not pay a `perf_counter` call, let alone a list append, at every phase
boundary --- and it must record a phase that raised, because the phase you most
want a number for is the one that just fell over.
"""

from __future__ import annotations

import json
import math
import time
from io import StringIO
from typing import TYPE_CHECKING, cast

import pytest
from deconvolve.instrumentation import timing
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import Any

    from deconvolve.coretypes import DatasetSplits
    from deconvolve.training.engine import TrainResult


def _rendered() -> str:
    """What `report` puts on a console, as text."""
    output = StringIO()
    timing.report(Console(file=output, color_system=None, width=100))
    return output.getvalue()


@pytest.fixture(autouse=True)
def _clean_recorder() -> Iterator[None]:
    """Every test starts from a disabled, empty recorder and leaves one."""
    timing.enable(False)
    yield
    timing.enable(False)


@pytest.fixture
def timing_on() -> Iterator[None]:
    """Timing on for the duration of the test, off again after."""
    timing.enable(True)
    yield
    timing.enable(False)


class TestDisabled:
    def test_is_default(self) -> None:
        assert not timing.is_enabled()

    def test_records_nothing(self) -> None:
        with timing.phase("data") as p:
            p.note("cache hit")
        assert timing.phases() == ()

    def test_yields_a_shared_singleton(self) -> None:
        """No per-phase allocation when off."""
        with timing.phase("a") as first, timing.phase("b") as second:
            assert first is second

    def test_block_returns_its_argument(self) -> None:
        sentinel = object()
        with timing.phase("train") as p:
            assert p.block(sentinel) is sentinel

    def test_report_is_silent(self) -> None:
        with timing.phase("data"):
            pass
        assert _rendered() == ""

    def test_write_makes_no_file(self, tmp_path: Path) -> None:
        with timing.phase("data"):
            pass
        timing.write(tmp_path, pass_name="train")
        assert not (tmp_path / "artifacts" / "timings.json").exists()


class TestEnabled:
    def test_records_a_phase(self) -> None:
        timing.enable(True)
        with timing.phase("data"):
            pass
        (record,) = timing.phases()
        assert record.name == "data"
        assert record.depth == 0
        assert record.seconds >= 0.0

    def test_note_sets_detail(self) -> None:
        timing.enable(True)
        with timing.phase("data") as p:
            p.note("cache hit")
        assert timing.phases()[0].detail == "cache hit"

    def test_note_reaches_the_open_phase_from_anywhere(self) -> None:
        """Loaders annotate the phase their caller opened, without owning it."""
        timing.enable(True)
        with timing.phase("data"):
            timing.note("generated")
        assert timing.phases()[0].detail == "generated"

    def test_note_targets_the_named_phase_not_the_innermost(self) -> None:
        """`evaluate_run` rebuilds the dataset, so the loaders' note would
        otherwise land on whichever phase happened to be open."""
        timing.enable(True)
        with timing.phase("data"):
            pass
        with timing.phase("evaluate"):
            timing.note("cache hit", to="data")
        assert [(p.name, p.detail) for p in timing.phases()] == [
            ("data", None),
            ("evaluate", None),
        ]

    def test_note_finds_a_named_phase_further_out(self) -> None:
        timing.enable(True)
        with timing.phase("data"), timing.phase("inner"):
            timing.note("cache hit", to="data")
        assert {p.name: p.detail for p in timing.phases()} == {
            "inner": None,
            "data": "cache hit",
        }

    def test_note_outside_a_phase_is_ignored(self) -> None:
        timing.enable(True)
        timing.note("nowhere")  # must not raise
        assert timing.phases() == ()

    def test_nesting_records_depth_and_closing_order(self) -> None:
        timing.enable(True)
        with timing.phase("train"):
            with timing.phase("transfer"):
                pass
            with timing.phase("loop"):
                pass
        assert [(p.name, p.depth) for p in timing.phases()] == [
            ("transfer", 1),
            ("loop", 1),
            ("train", 0),
        ]

    def test_a_raising_phase_is_still_recorded(self) -> None:
        timing.enable(True)
        with pytest.raises(RuntimeError), timing.phase("train"):
            raise RuntimeError("boom")
        (record,) = timing.phases()
        assert record.name == "train"
        assert record.failed

    def test_a_raising_phase_unwinds_the_depth(self) -> None:
        timing.enable(True)
        with pytest.raises(RuntimeError), timing.phase("outer"):
            raise RuntimeError("boom")
        with timing.phase("after"):
            pass
        assert timing.phases()[-1].depth == 0

    def test_enable_false_clears_what_was_recorded(self) -> None:
        timing.enable(True)
        with timing.phase("data"):
            pass
        timing.enable(False)
        assert timing.phases() == ()


class TestReport:
    def test_renders_a_row_per_phase(self) -> None:
        timing.enable(True)
        with timing.phase("data") as p:
            p.note("cache hit")
        with timing.phase("train"), timing.phase("loop"):
            pass
        rendered = _rendered()
        assert "data" in rendered
        assert "cache hit" in rendered
        assert "train" in rendered
        assert "loop" in rendered

    def test_shows_each_phase_share_of_the_total(self) -> None:
        """The point of the table: which component to go optimize."""
        timing.enable(True)
        with timing.phase("train"):
            pass
        assert "%" in _rendered()

    def test_says_nothing_when_no_phase_ran(self) -> None:
        timing.enable(True)
        assert _rendered() == ""


class TestWrite:
    def test_writes_a_flat_phase_list_parents_first(self, tmp_path: Path) -> None:
        """Flat with a depth field, but read back parent-then-children --- the
        recorder's own order closes children first, which reads backwards."""
        timing.enable(True)
        with timing.phase("train"), timing.phase("loop"):
            pass
        timing.write(tmp_path, pass_name="train")
        payload = json.loads((tmp_path / "artifacts" / "timings.json").read_text())
        assert [p["name"] for p in payload["phases"]] == ["train", "loop"]
        assert [p["depth"] for p in payload["phases"]] == [0, 1]
        assert payload["total_seconds"] >= 0.0

    def test_total_counts_only_top_level_phases(self, tmp_path: Path) -> None:
        """A nested phase is already inside its parent; adding it double-counts."""
        timing.enable(True)
        with timing.phase("train"), timing.phase("loop"):
            pass
        timing.write(tmp_path, pass_name="train")
        payload: dict[str, Any] = json.loads(
            (tmp_path / "artifacts" / "timings.json").read_text()
        )
        top: dict[str, Any] = next(p for p in payload["phases"] if p["name"] == "train")
        assert payload["total_seconds"] == pytest.approx(top["seconds"])

    def test_records_whether_the_compile_cache_was_warm(self, tmp_path: Path) -> None:
        """A warm cache makes `train.compile` read near-zero. Say so in the file,
        or the row invites exactly the wrong optimization."""
        timing.enable(True)
        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")
        payload = json.loads((tmp_path / "artifacts" / "timings.json").read_text())
        assert "compile_cache_warm" in payload

    def test_is_json_serializable_under_float32(self, tmp_path: Path) -> None:
        """np.float32 is not JSON-serializable; seconds must be plain floats."""
        timing.enable(True)
        with timing.phase("data"):
            pass
        timing.write(tmp_path, pass_name="train")
        payload = json.loads((tmp_path / "artifacts" / "timings.json").read_text())
        assert isinstance(payload["phases"][0]["seconds"], float)


@pytest.mark.usefixtures("timing_on")
class TestMerge:
    """`scripts/submit.sh` makes three passes over one run directory, and an
    overwriting writer meant the reload pass destroyed the training numbers on
    every pipeline run."""

    def test_a_reload_pass_does_not_destroy_the_training_numbers(
        self, tmp_path: Path
    ) -> None:
        """`scripts/submit.sh` makes three passes over one directory."""
        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")
        timing.reset()

        with timing.phase("plots"):
            pass
        timing.write(tmp_path, pass_name="load")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        by_name = {p["name"]: p for p in payload["phases"]}
        assert by_name.keys() == {"train", "plots"}
        assert by_name["train"]["pass"] == "train"
        assert by_name["plots"]["pass"] == "load"

    def test_a_rerun_phase_replaces_its_earlier_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two durations are pinned rather than raced.

        Two empty `with timing.phase(...)` blocks can land inside one
        `perf_counter` tick and record the identical duration, which made a
        "the second record replaced the first" assertion on seconds fail
        intermittently for a reason that has nothing to do with merging.
        """
        ticks: Iterator[float] = iter([0.0, 1.0, 10.0, 12.0])
        real: Callable[[], float] = time.perf_counter
        monkeypatch.setattr(timing.time, "perf_counter", lambda: next(ticks, real()))

        with timing.phase("plots"):
            pass
        timing.write(tmp_path, pass_name="train")
        first: dict[str, Any] = json.loads(
            (tmp_path / "artifacts/timings.json").read_text()
        )
        timing.reset()

        with timing.phase("plots"):
            pass
        timing.write(tmp_path, pass_name="load")
        second: dict[str, Any] = json.loads(
            (tmp_path / "artifacts/timings.json").read_text()
        )

        assert first["phases"][0]["seconds"] == pytest.approx(1.0)
        assert len(second["phases"]) == 1
        assert second["phases"][0]["pass"] == "load"
        assert second["phases"][0]["seconds"] == pytest.approx(2.0)

    def test_the_total_sums_the_merged_top_level_phases(self, tmp_path: Path) -> None:
        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")
        timing.reset()
        with timing.phase("plots"):
            pass
        timing.write(tmp_path, pass_name="load")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        top = [p["seconds"] for p in payload["phases"] if p["depth"] == 0]
        assert payload["total_seconds"] == pytest.approx(sum(top))

    def test_a_corrupt_timings_file_is_replaced_rather_than_raised_on(
        self, tmp_path: Path
    ) -> None:
        """The timing layer must never be what takes a run down."""
        (tmp_path / "artifacts").mkdir()
        _ = (tmp_path / "artifacts/timings.json").write_text("{not json")

        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        assert [p["name"] for p in payload["phases"]] == ["train"]

    def test_a_bare_list_payload_is_replaced_rather_than_raised_on(
        self, tmp_path: Path
    ) -> None:
        """Valid JSON, but not the object shape `_merged_phases` expects."""
        (tmp_path / "artifacts").mkdir()
        _ = (tmp_path / "artifacts/timings.json").write_text("[1, 2, 3]")

        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        assert [p["name"] for p in payload["phases"]] == ["train"]

    def test_a_phase_missing_depth_is_replaced_rather_than_raised_on(
        self, tmp_path: Path
    ) -> None:
        """A `phases` entry without `depth` would otherwise `KeyError` inside
        the total or `_merged_phases`."""
        (tmp_path / "artifacts").mkdir()
        _ = (tmp_path / "artifacts/timings.json").write_text(
            json.dumps({"phases": [{"name": "old", "seconds": 1.0}]})
        )

        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        assert [p["name"] for p in payload["phases"]] == ["train"]

    def test_a_non_numeric_seconds_is_replaced_rather_than_raised_on(
        self, tmp_path: Path
    ) -> None:
        """A non-numeric `seconds` would otherwise `TypeError` inside the sum
        that computes `total_seconds`."""
        (tmp_path / "artifacts").mkdir()
        _ = (tmp_path / "artifacts/timings.json").write_text(
            json.dumps({"phases": [{"name": "old", "seconds": "oops", "depth": 0}]})
        )

        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        assert [p["name"] for p in payload["phases"]] == ["train"]

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_a_non_finite_seconds_is_treated_as_corrupt(
        self, tmp_path: Path, literal: str
    ) -> None:
        """`NaN` is a `float` and passes isinstance, but poisons the total.

        One of them makes `total_seconds` non-finite for every phase in the
        file --- silently, and with no way to recover the numbers afterwards
        --- so the payload is discarded exactly as the other invalid shapes
        are. `json` emits and accepts these literals, so they really do reach
        `_existing` off disk.
        """
        (tmp_path / "artifacts").mkdir()
        _ = (tmp_path / "artifacts/timings.json").write_text(
            f'{{"phases": [{{"name": "old", "seconds": {literal}, "depth": 0}}]}}'
        )

        with timing.phase("train"):
            pass
        timing.write(tmp_path, pass_name="train")

        payload = json.loads((tmp_path / "artifacts/timings.json").read_text())
        assert [p["name"] for p in payload["phases"]] == ["train"]
        assert math.isfinite(cast("float", payload["total_seconds"]))


class TestEnvironment:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable(self, value: str) -> None:
        assert timing._enabled_from_env({"DECONVOLVE_TIMING": value})

    @pytest.mark.parametrize("value", ["", "0", "false", "FALSE", "no", "off"])
    def test_falsey_values_do_not(self, value: str) -> None:
        assert not timing._enabled_from_env({"DECONVOLVE_TIMING": value})

    def test_absent_does_not(self) -> None:
        assert not timing._enabled_from_env({})


@pytest.mark.slow
class TestTrainIntegration:
    """The nested phases inside `train`, and the risk that (ahead-of-time)
    compilation introduced to expose them changes the numbers.

    Deliberately not gated on a writable default cache. These call `train`,
    which points XLA's persistent cache at `COMPILE_CACHE_DIR`, but an
    unwritable one costs a recompile and a warning rather than the run --- and
    the warning is filtered in `pyproject.toml`. Skipping them would trade the
    one check that says a timed run is bit-identical to an untimed one for a
    cosmetic line.
    """

    @staticmethod
    def _splits() -> DatasetSplits:
        import numpy as np
        from deconvolve.coretypes import ZXY, Events
        from deconvolve.data import DeconvolveDataset

        rng = np.random.default_rng(21)
        n = 512
        z = rng.normal(size=(2 * n, 1)).astype(np.single)
        x = z + rng.normal(0, 0.3, size=(2 * n, 1)).astype(np.single)
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return DeconvolveDataset(batch_size=32, seed=5).splits_from_data(
            ZXY(Events(z, x), y)
        )

    @staticmethod
    def _train(splits: DatasetSplits) -> TrainResult:
        from deconvolve.training.engine import train

        return train(splits, dim=1, n_epochs=3, hidden_units=8, n_layers=1, seed=42)

    def test_timed_run_matches_an_untimed_one(self) -> None:
        """`DECONVOLVE_TIMING` must be observable in the report and nowhere else.

        Timing splits the fused path's single `jit` call into `lower().compile()`
        plus a call to the compiled object so the compile boundary is visible.
        That is the same executable either way --- and this is the test that says
        so, because a profiler that changes the result profiles nothing useful.
        """
        import numpy as np

        splits = self._splits()
        timing.enable(False)
        untimed = self._train(splits)
        timing.enable(True)
        timed = self._train(splits)

        for key in ("train_d", "train_g", "val_d"):
            np.testing.assert_array_equal(timed.history[key], untimed.history[key])
        assert timed.best_epoch == untimed.best_epoch

    def test_records_the_phases_inside_train(self) -> None:
        timing.enable(True)
        _ = self._train(self._splits())
        assert {p.name for p in timing.phases()} == {
            "transfer",
            "compile",
            "epochs",
            "select",
        }


class TestRecord:
    """`record()`: a phase measured in another process.

    `phase()` covers what this interpreter runs. The OmniFold baseline's real
    work happens under a different Python and reports its breakdown back as
    numbers, with no block here to wrap.
    """

    def test_a_recorded_phase_nests_under_the_open_one(self) -> None:
        timing.enable(True)

        with timing.phase("omnifold"):
            timing.record("unfold", 2400.0, detail="MultiFold.Unfold")

        by_name = {p.name: p for p in timing.phases()}
        assert by_name["unfold"].depth == by_name["omnifold"].depth + 1
        # Stored verbatim, not measured, so the value round-trips exactly;
        # compared loosely anyway because nothing here depends on the bits.
        assert by_name["unfold"].seconds == pytest.approx(2400.0)
        assert by_name["unfold"].detail == "MultiFold.Unfold"

    def test_a_recorded_phase_at_top_level_is_top_level(self) -> None:
        timing.enable(True)

        timing.record("standalone", 1.0)

        assert timing.phases()[-1].depth == 0

    def test_recording_is_a_no_op_when_timing_is_off(self) -> None:
        """The layer must cost nothing at all when disabled."""
        timing.enable(False)

        timing.record("ignored", 1.0)

        assert timing.phases() == ()

    def test_a_recorded_phase_does_not_inflate_the_total(self) -> None:
        """`total_seconds` sums top-level phases; a child is already inside one."""
        timing.enable(True)

        with timing.phase("omnifold"):
            timing.record("unfold", 2400.0)

        top = [p for p in timing.phases() if p.depth == 0]
        assert len(top) == 1
        assert top[0].seconds < 1.0


class TestWriteFilename:
    """A second program over the same run directory needs its own file."""

    def test_it_writes_the_named_file(self, tmp_path: Path) -> None:
        timing.enable(True)
        with timing.phase("omnifold"):
            pass

        timing.write(tmp_path, pass_name="omnifold", filename="timings_omnifold.json")

        payload = json.loads(
            (tmp_path / "artifacts" / "timings_omnifold.json").read_text()
        )
        assert [p["name"] for p in payload["phases"]] == ["omnifold"]
        assert payload["phases"][0]["pass"] == "omnifold"

    def test_it_does_not_touch_the_shared_timings_file(self, tmp_path: Path) -> None:
        """The reason the parameter exists.

        `write` merges by phase *name* alone, not by (pass, name). The baseline
        has phases called `data` and `evaluate` of its own, so writing them into
        `timings.json` would replace the training pass's rows -- the ones anyone
        actually wants -- rather than sit beside them.
        """
        timing.enable(True)
        with timing.phase("data"):
            pass
        timing.write(tmp_path, pass_name="train")
        timing.reset()

        timing.enable(True)
        with timing.phase("data"):
            pass
        timing.write(tmp_path, pass_name="omnifold", filename="timings_omnifold.json")

        shared = json.loads((tmp_path / "artifacts" / "timings.json").read_text())
        assert [p["pass"] for p in shared["phases"]] == ["train"]

    def test_the_default_is_still_the_shared_file(self, tmp_path: Path) -> None:
        timing.enable(True)
        with timing.phase("train"):
            pass

        timing.write(tmp_path, pass_name="train")

        assert (tmp_path / "artifacts" / "timings.json").is_file()
