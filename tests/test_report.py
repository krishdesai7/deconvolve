"""Tests for `deconvolve.reporting.report`'s template loading and value formatting."""

from __future__ import annotations

import math
import re
import shutil
import subprocess  # ruff: ignore[suspicious-subprocess-import]
from pathlib import Path
from typing import Any, Protocol

import pytest
from deconvolve.coretypes import SUBSTRUCTURE_VARIABLES
from deconvolve.reporting import report


def test_the_template_ships_with_the_package() -> None:
    """It is package data, not a repo-root file the wheel would drop."""
    assert "<<DETECTOR_WASSERSTEIN>>" in report.load_template()


def test_every_token_the_generator_fills_is_in_the_template() -> None:
    tokens = set(report.TEMPLATE_TOKEN.findall(report.load_template()))
    assert tokens == {
        "<<RUN_NAME>>",
        "<<CONFIG_ROWS>>",
        "<<TIMINGS_ROWS>>",
        "<<FIGURE_DIR>>",
        "<<DETECTOR_FIGURES>>",
        "<<PARTICLE_FIGURES>>",
        # One table per metric per level: sixteen columns needed `adjustbox`
        # to shrink to 7pt, which is below what anyone reads.
        "<<DETECTOR_WASSERSTEIN>>",
        "<<DETECTOR_JS>>",
        "<<DETECTOR_VLC>>",
        "<<PARTICLE_WASSERSTEIN>>",
        "<<PARTICLE_JS>>",
        "<<PARTICLE_VLC>>",
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.2089450359, "0.2089450359"),
        (8.29431403648554e-05, "0.0000829431"),
        (0.0, "0"),
        (-504.2501717, "-504.2501717"),
    ],
)
def test_numbers_are_emitted_as_plain_decimals(value: float, expected: str) -> None:
    """siunitx S columns fix a table-format and cannot absorb an exponent."""
    rendered: str = report.decimal(value)
    assert "e" not in rendered.lower()
    assert rendered == expected
    assert float(rendered) == pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_non_finite_value_renders_as_a_dash_rather_than_killing_the_compile(
    value: float,
) -> None:
    """`nan`/`inf` in a siunitx S column is a hard LaTeX error, not a bad cell.

    Reachable through `*_improvement_pct` whenever a `before` is 0.0.
    """
    assert report.decimal(value) == r"\multicolumn{1}{c}{---}"


def test_a_non_finite_config_value_replaces_its_siunitx_wrapper() -> None:
    r"""`\Raw` expands to `\num`, which rejects `nan` exactly as `S` does."""
    rows: str = report.config_rows({"lr_g": math.inf}, None)

    assert r"\Raw" not in rows
    assert r"\multicolumn{1}{c}{---}" in rows


def test_a_non_finite_improvement_leaves_the_rest_of_the_row_intact() -> None:
    """One dead cell must cost one cell, not the table."""
    entry: dict[str, float] = _entry(1.0, 0.1)
    entry["wasserstein_improvement_pct"] = math.inf
    body: str = report.metrics_table(
        "detector",
        "wasserstein",
        ("m",),
        {"detector_m": entry},
        None,
        None,
        frozenset(),
    )

    # Two absent-IBU cells, two absent-OmniFold cells, and the non-finite
    # RAN improvement.
    assert body.count(r"\multicolumn{1}{c}{---}") == 5
    # The scaled `after` still lands: 0.1 x 10^3.
    assert "100" in body


def test_underscores_in_a_value_are_escaped() -> None:
    r"""`f_ch` inside \texttt{} was a hard `Missing $ inserted` compile error."""
    assert report.latex_text("f_ch") == r"f\_ch"
    assert report.latex_text("100%") == r"100\%"


def test_the_token_guard_ignores_the_templates_own_documentation() -> None:
    """The header comment documents the contract and contains `<<...>>`."""
    assert report.TEMPLATE_TOKEN.findall("% Replace the six <<...>> tokens") == []


def test_config_rows_pair_two_entries_per_line() -> None:
    """The config table is four columns: key, value, key, value."""
    rows: str = report.config_rows({"dim": 12, "seed": 3, "n_layers": 2}, None)

    assert r"\ConfigPair{dim}{\Count{12}}{seed}{\Count{3}}" in rows
    # An odd trailing entry pads rather than shifting every later cell.
    assert r"\ConfigPair{n_layers}{\Count{2}}{}{}" in rows


def test_the_variable_list_uses_physics_symbols_in_display_order() -> None:
    """`tau21` is right for a machine interface and wrong for a table."""
    rows: str = report.config_rows({"variables": ["m", "tau21", "sdm"]}, None)

    assert r"\ln\rho" in rows
    assert r"\tau" in rows
    assert "tau21" not in rows
    # Display order puts the soft-drop mass before the N-subjettiness ratio.
    assert rows.index(r"\ln\rho") < rows.index(r"\tau")


def test_the_variable_list_spans_the_row() -> None:
    rows: str = report.config_rows({"variables": ["m", "f_ch"]}, None)
    assert r"\ConfigWide{variables}{" in rows
    assert r"\textbf{Mass and hard scale (IRC-safe kinematics):} $m$ [GeV]" in rows
    assert (
        r"\textbf{Hadronization, multiplicity and fragmentation (IRC-unsafe):} $f_{ch}$"
        in rows
    )


def test_the_mmd_sigmas_collapse_to_a_median_and_a_bracket() -> None:
    """Five bare floats tell a reader nothing; they are median x _SCALES."""
    median: float = 3.10746693611145
    sigmas: list[float] = [median * s for s in (0.5, 2**-0.5, 1.0, 2**0.5, 2.0)]

    rows: str = report.config_rows({"mmd_sigmas_detector": sigmas}, None)

    assert r"\ConfigWide{mmd_sigmas_detector}" in rows
    assert "3.107" in rows
    assert r"\sqrt2" in rows


def test_sigmas_that_are_not_the_bracket_are_printed_in_full() -> None:
    """A future change to _SCALES must not make the report quietly lie."""
    rows: str = report.config_rows({"mmd_sigmas_detector": [1.0, 2.0, 3.0]}, None)
    assert "1" in rows
    assert "2" in rows
    assert "3" in rows
    assert r"\sqrt2" not in rows


def test_compile_cache_warmth_reaches_the_config_table() -> None:
    """It is what makes the `compile` timing interpretable at all."""
    rows: str = report.config_rows({"dim": 1}, {"compile_cache_warm": True})
    assert "compile_cache_warm" in rows


def test_timing_rows_indent_nested_phases_and_blank_their_share() -> None:
    """A nested phase is already inside its parent's share."""
    payload = {
        "total_seconds": 10.0,
        "phases": [
            {
                "name": "train",
                "seconds": 8.0,
                "depth": 0,
                "detail": None,
                "failed": False,
                "pass": "train",
            },
            {
                "name": "compile",
                "seconds": 4.0,
                "depth": 1,
                "detail": None,
                "failed": False,
                "pass": "train",
            },
        ],
    }

    rows: str = report.timing_rows(payload)

    assert r"\TimingSubphase{compile}" in rows
    assert "compile} & 4 &  &" in rows.replace("\\TimingSubphase{", "")
    assert "80" in rows  # train's share
    assert r"\midrule" in rows
    assert "10" in rows  # the total row


def test_the_pass_is_folded_into_the_detail_cell() -> None:
    """The template's timing table has four columns and no pass column."""
    payload = {
        "total_seconds": 1.0,
        "phases": [
            {
                "name": "data",
                "seconds": 1.0,
                "depth": 0,
                "detail": "cache hit",
                "failed": False,
                "pass": "train",
            }
        ],
    }
    assert "cache hit -- train" in report.timing_rows(payload)


_METRIC_KEYS = ("wasserstein", "jensenshannon", "triangular")


def _entry(before: float, after: float) -> dict[str, float]:
    return {
        f"{m}_{suffix}": value
        for m in _METRIC_KEYS
        for suffix, value in (
            ("before", before),
            ("after", after),
            ("improvement_pct", (1 - after / before) * 100),
        )
    }


def test_rows_are_in_display_order() -> None:
    ran = {f"detector_{v}": _entry(1.0, 0.1) for v in SUBSTRUCTURE_VARIABLES}

    body: str = report.metrics_table(
        "detector", "wasserstein", SUBSTRUCTURE_VARIABLES, ran, None, None, frozenset()
    )

    # Group headings live in the configuration table, not the metric tables
    assert "Mass and hard scale" not in body
    assert body.index(r"$\ln\rho$") < body.index(r"$\lambda^{1}_{0.5}$")
    assert body.count(r"\midrule") == 1  # single header rule
    assert body.count(r"\\") == len(SUBSTRUCTURE_VARIABLES)


def test_a_group_with_no_variables_is_omitted_from_config_variables() -> None:
    """`--var m --var w` has nothing in the splitting group."""
    rows: str = report.config_rows({"variables": ["m", "w"]}, None)
    assert "Splitting" not in rows
    assert "Mass and hard scale" in rows
    assert "Continuous angularities" in rows


def test_a_missing_baseline_renders_dashes() -> None:
    """The template fixes the column count, so the group cannot be omitted."""
    ran = {"detector_m": _entry(1.0, 0.1)}

    body: str = report.metrics_table(
        "detector", "wasserstein", ("m",), ran, None, None, frozenset()
    )

    # Two cells per absent baseline, and both baselines are absent here.
    assert body.count(r"\multicolumn{1}{c}{---}") == 4


def test_a_skipped_variable_is_daggered_rather_than_shown_as_zero() -> None:
    """IBU returning its input unchanged is a refusal, not a measurement."""
    ran = {"detector_zg": _entry(1.0, 0.1)}
    ibu = {"detector_zg": _entry(1.0, 1.0)}

    body: str = report.metrics_table(
        "detector", "wasserstein", ("zg",), ran, ibu, None, frozenset({"zg"})
    )

    assert r"\dag" in body


def test_a_completed_variable_is_not_daggered() -> None:
    ran = {"detector_m": _entry(1.0, 0.1)}
    ibu = {"detector_m": _entry(1.0, 0.5)}

    body: str = report.metrics_table(
        "detector", "wasserstein", ("m",), ran, ibu, None, frozenset()
    )

    assert r"\dag" not in body


def test_a_gaussian_run_has_rows_but_no_groups() -> None:
    ran = {f"detector_dim_{i}": _entry(1.0, 0.1) for i in range(2)}

    body: str = report.metrics_table(
        "detector", "wasserstein", ("dim_0", "dim_1"), ran, None, None, frozenset()
    )

    assert "Mass and hard scale" not in body
    assert body.count(r"\\") == 2


def test_a_daggered_table_ends_with_a_legend_explaining_the_mark() -> None:
    """An unexplained dagger prevents no misreading."""
    ran = {"detector_zg": _entry(1.0, 0.1)}
    ibu = {"detector_zg": _entry(1.0, 1.0)}

    body: str = report.metrics_table(
        "detector", "wasserstein", ("zg",), ran, ibu, None, frozenset({"zg"})
    )

    assert body.splitlines()[-1] == report._DAGGER_LEGEND
    assert "failed to unfold" in body
    # The legend is a table row, so its braces must balance or the compile
    # dies the moment any variable is daggered.
    assert report._DAGGER_LEGEND.count("{") == report._DAGGER_LEGEND.count("}")


def test_a_table_with_nothing_skipped_carries_no_legend() -> None:
    ran = {f"detector_{v}": _entry(1.0, 0.1) for v in SUBSTRUCTURE_VARIABLES}
    ibu = {f"detector_{v}": _entry(1.0, 0.5) for v in SUBSTRUCTURE_VARIABLES}

    body: str = report.metrics_table(
        "detector", "wasserstein", SUBSTRUCTURE_VARIABLES, ran, ibu, None, frozenset()
    )

    assert report._DAGGER_LEGEND not in body
    assert "failed to unfold" not in body


def test_dagger_legend_appears_only_once_in_rendered_report(
    reference_run: Path,
) -> None:
    import json

    outcomes: list[dict[str, int | str]] = [
        {"variable_name": "m", "status": "skipped", "n_bins": 1}
    ]
    _ = (reference_run / "artifacts" / "ibu_outcomes.json").write_text(
        data=json.dumps(obj=outcomes)
    )
    source: str = report.render(reference_run)
    assert source.count("failed to unfold") == 1


def test_reading_the_skip_set_creates_nothing(tmp_path: Path) -> None:
    """`report` is a read-only consumer: it must not make `artifacts/`."""
    before: list[str] = sorted(p.name for p in tmp_path.iterdir())

    assert report.skipped_variables(tmp_path, None) == frozenset()

    assert sorted(p.name for p in tmp_path.iterdir()) == before == []


# --- The document, end to end ------------------------------------------------


class ReferenceRunBuilder(Protocol):
    """`conftest.make_reference_run`'s call signature."""

    def __call__(
        self, run_dir: Path, /, config: dict[str, object] | None = None
    ) -> Path: ...


def test_rendering_leaves_no_token_behind(reference_run: Path) -> None:
    assert report.TEMPLATE_TOKEN.findall(report.render(reference_run)) == []


def test_the_figure_paths_are_absolute(reference_run: Path) -> None:
    """`pdflatex` runs in `artifacts/`, so a relative path would not resolve."""
    source: str = report.render(reference_run)
    assert f"{(reference_run / 'artifacts').resolve()}/detector_level.pdf" in source


def test_a_run_directory_with_an_underscore_renders(
    tmp_path: Path, make_reference_run: ReferenceRunBuilder
) -> None:
    """Sweep arms are named like `hp_x/lrg1e-4_seed03`."""
    run_dir: Path = make_reference_run(tmp_path / "lrg1e-4_seed03")
    assert report.TEMPLATE_TOKEN.findall(report.render(run_dir)) == []


def test_no_cell_carries_an_exponent(reference_run: Path) -> None:
    """siunitx S columns fix a table-format and cannot absorb one."""
    body: str = report.render(reference_run)
    rows: list[str] = [
        line for line in body.splitlines() if line.rstrip().endswith(r"\\")
    ]
    assert rows
    assert not any(re.search(r"\d[eE][+-]?\d", row) for row in rows)


def test_a_missing_config_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        _ = report.render(tmp_path)


def test_rendering_creates_nothing_in_a_directory_that_is_not_a_run(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        _ = report.render(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_metrics_degrade_to_a_row_rather_than_raising(
    reference_run: Path,
) -> None:
    """A run that died before `deconvolve evaluate` still produces a report."""
    (reference_run / "artifacts" / "metrics.json").unlink()

    source: str = report.render(reference_run)

    assert source.count("metrics.json not found") == 6  # 3 metrics x 2 levels
    assert rf"\multicolumn{{{report._TABLE_COLUMNS}}}" in source


def test_missing_timings_leave_an_empty_body(reference_run: Path) -> None:
    (reference_run / "artifacts" / "timings.json").unlink()

    source: str = report.render(reference_run)

    assert report.TEMPLATE_TOKEN.findall(source) == []
    assert "compile_cache_warm" not in source


def test_a_missing_baseline_still_renders(reference_run: Path) -> None:
    (reference_run / "artifacts" / "metrics_ibu.json").unlink()

    source: str = report.render(reference_run)

    assert r"\multicolumn{1}{c}{---}" in source


def test_a_gaussian_run_renders(
    tmp_path: Path, make_reference_run: ReferenceRunBuilder
) -> None:
    """The `gaussian_params` row has no other coverage."""
    from deconvolve.data import parse_gaussian_config

    run_dir: Path = make_reference_run(
        tmp_path / "gaussian_run",
        config={
            "dataset": "gaussian",
            "dim": 2,
            "n_samples": 4096,
            "seed": 7,
            "variables": ["dim_0", "dim_1"],
            "gaussian_params": parse_gaussian_config(
                Path("params/2d_correlated.yaml")
            ).model_dump(),
        },
    )

    source: str = report.render(run_dir)

    assert report.TEMPLATE_TOKEN.findall(source) == []
    assert r"\ConfigWide{gaussian_params}" in source


def test_no_compile_stops_at_the_source(reference_run: Path) -> None:
    produced: Path = report.build_report(reference_run, compile_pdf=False)

    assert produced == reference_run / "artifacts" / "report.tex"
    assert produced.read_text(encoding="utf-8").startswith(r"\documentclass")
    assert not (reference_run / "report.pdf").exists()


def test_a_missing_pdflatex_names_the_binary_and_the_escape_hatch(
    reference_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report.shutil, "which", lambda _cmd: None)

    with pytest.raises(RuntimeError, match="pdflatex"):
        _ = report.build_report(reference_run)


def test_an_existing_report_is_not_rebuilt_without_force(
    reference_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf: Path = reference_run / "report.pdf"
    _ = pdf.write_bytes(b"stale")

    def _fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("compiled despite an existing report.pdf")

    monkeypatch.setattr(report, "_compile", _fail)

    assert report.build_report(reference_run) == pdf
    assert pdf.read_bytes() == b"stale"


def test_the_compile_invocation_is_absolute_and_non_interactive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_reference_run: ReferenceRunBuilder,
) -> None:
    r"""The argv contract, asserted without needing a TeX installation.

    Two separate bugs live in this one call, and both are invisible to every
    test that hands `build_report` an absolute `tmp_path`:

    `pdflatex` runs with `cwd=artifacts`, so a relative `-output-directory`
    resolves against `artifacts/` rather than the caller's working directory
    -- and a relative run directory is precisely what `scripts/submit.sh` and
    the documented `deconvolve report runs/<timestamp>` both pass.

    `-interaction=nonstopmode` does not cover pdflatex's pre-mode "I can't
    write on file `report.log'" prompt, so without `stdin=DEVNULL` the failure
    reads the parent's stdin and hangs instead of erroring.
    """
    run_dir: Path = make_reference_run(tmp_path / "2026-09-06T203848Z")
    captured: list[tuple[Any, dict[str, Any]]] = []

    def fake_run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append((args, kwargs))
        _ = (run_dir / "report.pdf").write_bytes(b"%PDF-1.5")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(report.shutil, "which", lambda _cmd: "/usr/bin/pdflatex")
    monkeypatch.setattr(report.subprocess, "run", fake_run)
    # A working directory that is *not* the run's parent, so a path that only
    # happens to resolve relative to the run directory still fails.
    monkeypatch.chdir(tmp_path.parent)
    relative: Path = Path(tmp_path.name) / run_dir.name
    assert not relative.is_absolute()

    _ = report.build_report(relative, force=True)

    assert captured
    for args, kwargs in captured:
        flag: str = next(a for a in args if a.startswith("-output-directory="))
        emitted = Path(flag.removeprefix("-output-directory="))
        assert emitted.is_absolute()
        assert emitted == run_dir.resolve()
        assert kwargs["stdin"] is subprocess.DEVNULL


# The five tests below need a real `pdflatex` and so do not run in CI, which
# installs no TeX -- `texlive-latex-recommended` plus `-extra` is hundreds of
# megabytes on every run, for a hand-authored template that changes rarely.
# `test_the_compile_invocation_is_absolute_and_non_interactive` above is what
# guards the invocation contract in their absence; it needs no TeX and so runs
# everywhere.
_NO_TEX = shutil.which("pdflatex") is None


@pytest.mark.slow
@pytest.mark.skipif(_NO_TEX, reason="no TeX installation")
def test_the_report_compiles(reference_run: Path) -> None:
    produced: Path = report.build_report(reference_run)

    assert produced == reference_run / "report.pdf"
    assert produced.stat().st_size > 0
    assert (reference_run / "artifacts" / "report.tex").exists()
    # The run root holds exactly report.pdf and config.json.
    assert sorted(p.name for p in reference_run.iterdir()) == [
        "artifacts",
        "config.json",
        "report.pdf",
    ]


@pytest.mark.slow
@pytest.mark.skipif(_NO_TEX, reason="no TeX installation")
def test_a_run_directory_with_an_underscore_compiles(
    tmp_path: Path, make_reference_run: ReferenceRunBuilder
) -> None:
    run_dir: Path = make_reference_run(tmp_path / "lrg1e-4_seed03")
    assert report.build_report(run_dir).stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(_NO_TEX, reason="no TeX installation")
def test_a_gaussian_run_compiles(
    tmp_path: Path, make_reference_run: ReferenceRunBuilder
) -> None:
    from deconvolve.data import parse_gaussian_config

    run_dir: Path = make_reference_run(
        tmp_path / "gaussian_run",
        config={
            "dataset": "gaussian",
            "dim": 2,
            "n_samples": 4096,
            "seed": 7,
            "variables": ["dim_0", "dim_1"],
            "gaussian_params": parse_gaussian_config(
                Path("params/2d_correlated.yaml")
            ).model_dump(),
        },
    )

    assert report.build_report(run_dir).stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(_NO_TEX, reason="no TeX installation")
def test_a_run_without_a_baseline_still_compiles(reference_run: Path) -> None:
    (reference_run / "artifacts" / "metrics_ibu.json").unlink()
    assert report.build_report(reference_run, force=True).stat().st_size > 0


@pytest.mark.slow
@pytest.mark.skipif(_NO_TEX, reason="no TeX installation")
def test_a_missing_figure_becomes_a_placeholder_rather_than_a_failure(
    reference_run: Path,
) -> None:
    r"""`\ReportGraphic` wraps `\IfFileExists`; nothing here probes the disk."""
    (reference_run / "artifacts" / "selection.pdf").unlink()

    produced: Path = report.build_report(reference_run)

    assert produced.stat().st_size > 0


class TestOmniFoldColumns:
    """The third arm's columns, and the agreement they depend on."""

    def test_the_template_column_count_matches_the_spans(self) -> None:
        r"""`_TABLE_COLUMNS` and the template's `tabular` spec must agree.

        `report.py` emits `\multicolumn{N}` spans for the group headings, the
        dagger legend and the no-metrics row; the template fixes the real
        column count. Nothing connects them but this assertion, and they had
        already drifted once -- two comments in `report.py` said "sixteen
        columns" while the spans said six.
        """
        spec = re.search(
            r"\\begin\{tabular\}\{@\{\}([^}]*?)@\{\}\}", report.load_template()
        )
        assert spec is not None
        columns = [
            token for token in str(spec.group(1)).split() if token in {"l", "D", "P"}
        ]
        assert len(columns) == report._TABLE_COLUMNS

    def test_an_omnifold_metrics_file_reaches_the_tables(self) -> None:
        """Presence of `metrics_omnifold.json` is the whole mechanism."""
        entry = _entry(1.0, 0.25)
        body = report.metrics_table(
            "detector",
            "wasserstein",
            ("m",),
            {"detector_m": entry},
            None,
            {"detector_m": _entry(1.0, 0.5)},
            frozenset(),
        )

        # IBU absent: two dashes. OmniFold present: its scaled after (0.5e3)
        # and its improvement (50%).
        assert body.count(r"\multicolumn{1}{c}{---}") == 2
        assert "500" in body
        assert "50" in body

    def test_the_methods_are_ordered_ibu_omnifold_ran(self) -> None:
        """RAN sits last, where the eye lands, behind what it is compared to.

        Ordering is positional in the row -- nothing labels the cells -- so a
        swapped pair would silently attribute each method's numbers to another.
        """
        body = report.metrics_table(
            "detector",
            "wasserstein",
            ("m",),
            {"detector_m": _entry(1.0, 0.001)},  # RAN: 1.0
            {"detector_m": _entry(1.0, 0.002)},  # IBU: 2.0
            {"detector_m": _entry(1.0, 0.003)},  # OmniFold: 3.0
            frozenset(),
        )

        cells = [c.strip() for c in body.splitlines()[-1].split("&")]
        # label, Sim, IBU, IBU%, OmniFold, OmniFold%, RAN, RAN%
        assert len(cells) == report._TABLE_COLUMNS
        assert cells[2].removeprefix(r"\bfseries ").startswith("2")
        assert cells[4].removeprefix(r"\bfseries ").startswith("3")
        assert cells[6].removeprefix(r"\bfseries ").startswith("1")
        # RAN performed best (1.0 vs 2.0 and 3.0), so its cells are bolded
        assert cells[6].startswith(r"\bfseries")
        assert cells[7].startswith(r"\bfseries")
        assert not cells[2].startswith(r"\bfseries")
        assert not cells[4].startswith(r"\bfseries")

    def test_best_performing_method_is_bolded(self) -> None:
        r"""The method achieving lowest distance is highlighted with \bfseries."""
        # OmniFold (0.001) wins over RAN (0.002) and IBU (0.003)
        body = report.metrics_table(
            "detector",
            "wasserstein",
            ("m",),
            {"detector_m": _entry(1.0, 0.002)},
            {"detector_m": _entry(1.0, 0.003)},
            {"detector_m": _entry(1.0, 0.001)},
            frozenset(),
        )
        cells = [c.strip() for c in body.splitlines()[-1].split("&")]
        assert cells[4].startswith(r"\bfseries")
        assert cells[5].startswith(r"\bfseries")
        assert not cells[2].startswith(r"\bfseries")
        assert not cells[6].startswith(r"\bfseries")

    def test_skipped_ibu_is_not_eligible_for_best_method(self) -> None:
        """A daggered IBU did not unfold, so it cannot win even with a lower value."""
        body = report.metrics_table(
            "detector",
            "wasserstein",
            ("zg",),
            {"detector_zg": _entry(1.0, 0.5)},
            {"detector_zg": _entry(1.0, 0.1)},
            {"detector_zg": _entry(1.0, 0.4)},
            frozenset({"zg"}),
        )
        # Last line is the legend, second to last is data row
        cells = [c.strip() for c in body.splitlines()[-2].split("&")]
        assert cells[4].startswith(r"\bfseries")
        assert not cells[2].startswith(r"\bfseries")

    def test_a_render_without_omnifold_still_fills_the_columns(self) -> None:
        """The template fixes the column count, so absent means dashes."""
        body = report.metrics_table(
            "detector",
            "wasserstein",
            ("m",),
            {"detector_m": _entry(1.0, 0.1)},
            {"detector_m": _entry(1.0, 0.5)},
            None,
            frozenset(),
        )

        row = body.splitlines()[-1]
        assert len(row.split("&")) == report._TABLE_COLUMNS
        assert row.count(r"\multicolumn{1}{c}{---}") == 2
