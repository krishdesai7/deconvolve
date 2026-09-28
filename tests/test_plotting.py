"""Tests for the loss-curve figure.

`losses.pdf` carried two identical dashed lines for a long time: `engine.py`
recorded the validation BCE into both the `val_d` and `val_g` history columns,
and `plot_losses` faithfully drew both. Nothing failed --- the figure just
claimed to show something it did not measure --- so these pin the shape of the
plot rather than only the shape of the data.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest
from deconvolve.coretypes import (
    JET_OBS,
    PANEL_COLUMNS,
    PANELS_PER_PAGE,
    SUBSTRUCTURE_VARIABLES,
    Events,
    Populations,
    figure_pages,
)
from deconvolve.data import ArrayDataset
from deconvolve.evaluation import plotting
from deconvolve.evaluation.plotting import (
    _DETECTOR,
    _hist_ratio_panel,
    _plot_level,
    _save_fig,
    ibu_overlay,
    omnifold_overlay,
    plot_levels,
    plot_losses,
    plot_selection,
)
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from deconvolve.coretypes import DeconvolveModel
    from numpy.typing import NDArray

type DrawnCalls = list[tuple[tuple[Any, ...], dict[str, Any]]]


def test_filled_histograms_use_one_artist_per_distribution() -> None:
    """Falling back to Matplotlib's default bar histogram creates one
    rectangle per bin and makes the large vector PDFs needlessly expensive.
    """
    figure = Figure()
    ax = figure.add_subplot(211)
    ax_r = figure.add_subplot(212)
    nature = np.array([0.1, 0.3, 0.6, 0.8], dtype=np.single)
    mc = np.array([0.2, 0.4, 0.5, 0.9], dtype=np.single)

    _hist_ratio_panel(
        ax,
        ax_r,
        nature,
        mc,
        np.ones(4, dtype=np.single),
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        nature_label="Data",
        mc_label="Sim",
        xlabel="x",
        title="Detector level",
    )

    assert len(ax.patches) == 3


def test_a_panel_is_labelled_even_without_a_baseline() -> None:
    """No `*_weights.npz` exists on the default `deconvolve train` path, so a panel
    drawn with no overlays must still get a y-label, a title and a legend --
    not only one drawn against a baseline."""
    figure = Figure()
    ax = figure.add_subplot(211)
    ax_r = figure.add_subplot(212)
    nature = np.array([0.1, 0.3, 0.6, 0.8], dtype=np.single)
    mc = np.array([0.2, 0.4, 0.5, 0.9], dtype=np.single)

    _hist_ratio_panel(
        ax,
        ax_r,
        nature,
        mc,
        np.ones(4, dtype=np.single),
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        nature_label="Data",
        mc_label="Sim",
        xlabel="x",
        title="Detector level",
    )

    assert ax.get_ylabel() == "Events"
    assert ax.get_title() == "Detector level"
    _, labels = ax.get_legend_handles_labels()
    assert labels == ["Data", "Sim", "RAN"]


def test_the_legend_lists_every_baseline_that_is_drawn() -> None:
    """The regression risk in moving the label/legend/title out of the overlay
    loop: the legend must pick up one handle per overlay, in draw order."""
    figure = Figure()
    ax = figure.add_subplot(211)
    ax_r = figure.add_subplot(212)
    nature = np.array([0.1, 0.3, 0.6, 0.8], dtype=np.single)
    mc = np.array([0.2, 0.4, 0.5, 0.9], dtype=np.single)

    _hist_ratio_panel(
        ax,
        ax_r,
        nature,
        mc,
        np.ones(4, dtype=np.single),
        bins=[0.0, 0.25, 0.5, 0.75, 1.0],
        nature_label="Data",
        mc_label="Sim",
        xlabel="x",
        title="Detector level",
        overlays=[
            ibu_overlay([np.ones(4, dtype=np.single)]).at(0),
            omnifold_overlay(np.ones(4, dtype=np.single), 1).at(0),
        ],
    )

    _, labels = ax.get_legend_handles_labels()
    assert labels == ["Data", "Sim", "RAN", "IBU", "OmniFold"]


def test_ran_is_drawn_more_prominently_than_the_baseline() -> None:
    """RAN's step line was fainter than IBU's. On the same panel."""
    assert plotting.ALPHA_RAN > plotting.ALPHA_IBU > plotting.ALPHA_FILL


def test_ran_has_a_colour_of_its_own() -> None:
    assert plotting.COLOR_RAN not in {
        plotting.COLOR_NATURE,
        plotting.COLOR_MC,
        plotting.COLOR_IBU,
        "black",
    }


def test_saving_a_figure_trims_to_its_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without `bbox_inches`, the y-labels are clipped by the page edge."""
    seen: dict[str, object] = {}
    figure = Figure()
    monkeypatch.setattr(
        target=figure, name="savefig", value=lambda **kw: seen.update(kw)
    )

    _save_fig(figure, save_path=tmp_path / "f.pdf")

    assert seen["bbox_inches"] == "tight"


def test_the_main_panel_prunes_its_lowest_tick() -> None:
    """The main axis `0` and the ratio axis `1.5` overprinted each other."""
    figure = Figure()
    figure.canvas = FigureCanvasPdf(figure)
    ax, ax_r = figure.subplots(nrows=2)
    rng = np.random.default_rng(seed=0)

    _hist_ratio_panel(
        ax,
        ax_r,
        x_nature=rng.normal(size=512).astype(np.single),
        x_mc=rng.normal(size=512).astype(np.single),
        w_ran=np.ones(512, dtype=np.single),
        bins=20,
        nature_label="Data",
        mc_label="Sim",
        xlabel="x",
        title="t",
    )
    figure.canvas.draw()

    assert isinstance(ax.yaxis.get_major_locator(), MaxNLocator)
    assert ax.get_yticks()[0] > ax.get_ylim()[0]


def test_multilevel_figure_keeps_rendered_content_inside_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`tight_layout` must contain labels and titles, not just axes rectangles."""
    captured: list[Figure] = []

    def capture(figures: Sequence[Figure], /, *, save_path: Path) -> None:
        del save_path
        captured.extend(figures)

    monkeypatch.setattr("deconvolve.evaluation.plotting._save_pages", capture)
    values = np.array(
        [[-1.0, -0.5], [0.0, 0.2], [0.5, 0.8], [1.0, 1.2]], dtype=np.single
    )
    _plot_level(
        values,
        values + np.single(0.1),
        np.ones(4, dtype=np.single),
        _DETECTOR,
        tmp_path / "levels.pdf",
        None,
        [ibu_overlay([np.ones(4, dtype=np.single), np.ones(4, dtype=np.single)])],
    )

    figure = captured[0]
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()
    page = figure.bbox
    for ax in figure.axes:
        content = ax.get_tightbbox(renderer)
        assert content is not None
        assert content.x0 >= page.x0
        assert content.y0 >= page.y0
        assert content.x1 <= page.x1
        assert content.y1 <= page.y1


# Measured inside a `pdflscape` landscape block under the template's
# `\newgeometry{margin=8mm}`: \linewidth 749.4pt, \textheight 568.8pt. A
# figure wider than this is fitted to the page width rather than its height.
_LANDSCAPE_BLOCK_ASPECT: float = 749.4 / 568.8

_LAST_PAGES: list[list[Figure]] = []


def _capture_save(figures: Sequence[Figure], /, *, save_path: Path) -> None:
    """A `_save_pages` stand-in that records the pages instead of writing them."""
    del save_path
    _LAST_PAGES.append(list(figures))


def _drawn_pages() -> list[Figure]:
    """Every page of the level figure most recently drawn."""
    return _LAST_PAGES[-1]


def _last_drawn_figure() -> Figure:
    """The first page, for assertions that do not care about pagination."""
    return _LAST_PAGES[-1][0]


def _drawn_panels() -> list[Axes]:
    """Every histogram panel across every page, in drawn order."""
    return [
        a for page in _drawn_pages() for a in page.axes if a.get_ylabel() == "Events"
    ]


def _var_info_for(variables: tuple[str, ...]) -> list[dict[str, object]]:
    """One `VarInfo`-shaped dict per column, in column (not display) order."""
    return [
        {
            "xlim": JET_OBS[name].xlim,
            "xlabel": JET_OBS[name].xlabel,
            "symbol": JET_OBS[name].symbol,
            "mu": 0.0,
            "sigma": 1.0,
        }
        for name in variables
    ]


def _plot_twelve_dim_level(save_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Draw a synthetic 12-column detector-level figure (nature/mc/weights)."""
    rng = np.random.default_rng(seed=0)
    dim = len(SUBSTRUCTURE_VARIABLES)
    nature = rng.normal(size=(64, dim)).astype(np.single)
    mc = rng.normal(size=(64, dim)).astype(np.single)
    w = np.ones(64, dtype=np.single)
    baselines = [ibu_overlay([np.ones(64, dtype=np.single) for _ in range(dim)])]
    monkeypatch.setattr("deconvolve.evaluation.plotting._save_pages", _capture_save)
    _plot_level(
        nature,
        mc,
        w,
        _DETECTOR,
        save_path,
        cast("list[Any]", _var_info_for(SUBSTRUCTURE_VARIABLES)),
        baselines,
        variables=SUBSTRUCTURE_VARIABLES,
    )


def _panel_titles_for(
    variables: tuple[str, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> list[str]:
    """Titles of a jet-level figure's panels, in drawn (display) order."""
    rng = np.random.default_rng(seed=1)
    dim = len(variables)
    nature = rng.normal(size=(64, dim)).astype(np.single)
    mc = rng.normal(size=(64, dim)).astype(np.single)
    w = np.ones(64, dtype=np.single)
    baselines = [ibu_overlay([np.ones(64, dtype=np.single) for _ in range(dim)])]
    monkeypatch.setattr("deconvolve.evaluation.plotting._save_pages", _capture_save)
    _plot_level(
        nature,
        mc,
        w,
        _DETECTOR,
        tmp_path / "levels.pdf",
        cast("list[Any]", _var_info_for(variables)),
        baselines,
        variables=variables,
    )
    return [
        title
        for page in _drawn_pages()
        for ax in page.axes
        if (title := ax.get_title())
    ]


def _one_dim_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Figure:
    """A 1D Gaussian-shaped level figure: no `var_info`, no `variables`."""
    rng = np.random.default_rng(seed=2)
    nature = rng.normal(size=(64, 1)).astype(np.single)
    mc = rng.normal(size=(64, 1)).astype(np.single)
    w = np.ones(64, dtype=np.single)
    baselines = [ibu_overlay([np.ones(64, dtype=np.single)])]
    monkeypatch.setattr("deconvolve.evaluation.plotting._save_pages", _capture_save)
    _plot_level(nature, mc, w, _DETECTOR, tmp_path / "levels.pdf", None, baselines)
    return _last_drawn_figure()


def test_twelve_observables_are_paginated_six_to_a_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""`\includegraphics` scales a figure to its text block and every font
    scales with it, so a twelve-panel 12x24in figure renders its 18pt labels
    at 5pt. A panel's width on the page is `linewidth / columns` whatever the
    figure's inch dimensions, so the column count sets it and the cell's
    absolute inches set the text size. Six 7:6.6 cells three across, at
    7x6.6in, give 3.5x3.3in panels with 9.1pt text.
    """
    save_path: Path = tmp_path / "detector.pdf"
    _plot_twelve_dim_level(save_path, monkeypatch)

    pages: list[Figure] = _drawn_pages()
    assert len(pages) == figure_pages(12) == 2
    assert len(_drawn_panels()) == 12

    for page in pages:
        hist_axes = [a for a in page.axes if a.get_ylabel() == "Events"]
        assert len(hist_axes) == PANELS_PER_PAGE
        columns = {round(a.get_position().x0, 3) for a in hist_axes}
        rows = {round(a.get_position().y0, 3) for a in hist_axes}
        assert len(columns) == PANEL_COLUMNS
        assert len(rows) == PANELS_PER_PAGE // PANEL_COLUMNS
        # A cell is wider than it is tall -- a hist over a ratio panel wants
        # roughly 7:6.6. The 4x6 cell this replaced was the same panel on its
        # end, which no amount of paginating fixes.
        cell = page.get_figwidth() / PANEL_COLUMNS
        cell_h = page.get_figheight() / (PANELS_PER_PAGE // PANEL_COLUMNS)
        assert cell / cell_h > 1.0
        # Wider than the landscape text block, so `\includegraphics`
        # fits it to the page WIDTH. Fitting to the height instead is
        # what rendered the panel text at 5pt.
        aspect = page.get_figwidth() / page.get_figheight()
        assert aspect >= _LANDSCAPE_BLOCK_ASPECT


def test_a_page_counter_appears_only_when_there_is_more_than_one_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-page figure must read exactly as it did before pagination."""
    _plot_twelve_dim_level(tmp_path / "detector.pdf", monkeypatch)
    multi = [p.get_suptitle() for p in _drawn_pages()]
    assert multi[0] == "Detector Level (1 of 2)"
    assert multi[-1] == "Detector Level (2 of 2)"

    _ = _one_dim_level(tmp_path, monkeypatch)
    assert len(_drawn_pages()) == 1
    single = _drawn_pages()[0].get_suptitle()
    assert single == "Detector Level"


def test_adjacent_panel_titles_do_not_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 4-inch panel column is much narrower than the 8-inch-wide 1x12
    strip the titles were sized for. Repeating "(detector level)" on every
    one of twelve panels made adjacent titles overlap horizontally -- up to
    109.5px in one measured case -- which no geometry-only assertion (panel
    count, row/column count) would catch.
    """
    save_path: Path = tmp_path / "detector.pdf"
    _plot_twelve_dim_level(save_path, monkeypatch)

    figure: Figure = _drawn_pages()[0]
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    renderer = canvas.get_renderer()

    hist_axes = [a for a in figure.axes if a.get_ylabel() == "Events"]
    rows: dict[float, list[Axes]] = {}
    for ax in hist_axes:
        rows.setdefault(round(ax.get_position().y0, 3), []).append(ax)

    for row_axes in rows.values():
        row_axes.sort(key=lambda a: a.get_position().x0)
        for left, right in pairwise(row_axes):
            left_box = left.title.get_window_extent(renderer)
            right_box = right.title.get_window_extent(renderer)
            assert left_box.x1 <= right_box.x0, (
                f"{left.title.get_text()!r} overlaps {right.title.get_text()!r} "
                f"by {left_box.x1 - right_box.x0:.1f}px"
            )


def test_panels_are_drawn_in_display_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Panel 0 is `m`, panel 1 is the soft-drop mass, not the multiplicity."""
    titles: list[str] = _panel_titles_for(SUBSTRUCTURE_VARIABLES, tmp_path, monkeypatch)
    assert titles[0].startswith("Jet Mass")
    assert titles[1].startswith("Soft Drop Jet Mass")


def test_a_single_dimension_still_draws_one_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 1D Gaussian config must be unaffected."""
    figure: Figure = _one_dim_level(tmp_path, monkeypatch)
    assert len([a for a in figure.axes if a.get_ylabel() == "Events"]) == 1


def test_a_variable_subset_stays_in_physics_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--var` selecting three of twelve must still lay out sensibly and keep
    physics order, not the order the flags happened to be given in."""
    subset = ("tau21", "m", "zg")
    titles: list[str] = _panel_titles_for(subset, tmp_path, monkeypatch)
    assert titles[0].startswith("Jet Mass")
    assert titles[1].startswith("Groomed Jet Momentum Fraction")
    assert titles[2].startswith(r"$N$-subjettiness")


def test_plot_levels_evaluates_generator_once_per_chunk(tmp_path: Path) -> None:
    """Chunking may call the model repeatedly, but the second figure must not
    repeat those calls for the identical generator population.
    """
    z_gen = np.linspace(-1.0, 1.0, num=10_001, dtype=np.single)[:, None]
    x_sim = z_gen + np.single(0.1)
    truth = z_gen + np.single(0.2)
    data = truth + np.single(0.1)
    populations = Populations.create(mc=Events(z_gen, x_sim), data=data, truth=truth)
    dataset = ArrayDataset(populations.interleave(), batch_size=2)
    calls = 0

    def generator(z: NDArray[np.single]) -> NDArray[np.single]:
        nonlocal calls
        calls += 1
        return np.ones((len(z), 1), dtype=np.single)

    detector = tmp_path / "detector.pdf"
    particle = tmp_path / "particle.pdf"
    plot_levels(dataset, cast("DeconvolveModel", generator), detector, particle)

    assert calls == 2
    assert detector.exists()
    assert particle.exists()


def test_plot_levels_uses_the_same_page_height_for_matching_panel_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both level figures have the same panel geometry; making the particle
    page much taller adds whitespace without adding information.
    """
    z_gen = np.array([[-1.0], [0.0], [1.0]], dtype=np.single)
    events = Events(z_gen, z_gen + np.single(0.1))
    populations = Populations.create(mc=events, data=z_gen, truth=z_gen)
    dataset = ArrayDataset(populations.interleave(), batch_size=2)
    captured: list[Figure] = []

    def capture(figures: Sequence[Figure], /, *, save_path: Path) -> None:
        del save_path
        captured.extend(figures)

    def generator(z: NDArray[np.single]) -> NDArray[np.single]:
        return np.ones((len(z), 1), dtype=np.single)

    monkeypatch.setattr("deconvolve.evaluation.plotting._save_pages", capture)
    plot_levels(
        dataset,
        cast("DeconvolveModel", generator),
        tmp_path / "detector.pdf",
        tmp_path / "particle.pdf",
    )

    assert captured[0].get_figheight() == captured[1].get_figheight()


def _history(n: int = 12) -> dict[str, list[float]]:
    """What `train` returns now: one validation column, not two."""
    rng = np.random.default_rng(0)
    val = (0.69 + rng.normal(scale=0.01, size=n)).tolist()
    return {
        "train_d": (0.68 + rng.normal(scale=0.01, size=n)).tolist(),
        "train_g": (-0.68 + rng.normal(scale=0.01, size=n)).tolist(),
        "val_d": val,
    }


@pytest.fixture
def drawn(monkeypatch: pytest.MonkeyPatch) -> DrawnCalls:
    """Record every `ax.plot` call, then let it draw for real.

    `plot_losses` builds its own Figure and returns nothing, so intercepting
    the Axes is the only way to assert on what ends up in the legend.

    The recorder delegates rather than swallowing the call. `plot_losses` ends
    with `ax.legend()`, and an Axes holding no labelled artist makes matplotlib
    warn "No artists with labels found to put in legend" on every suite run.
    That warning is about the fixture, not about the code under test, so it is
    silenced here at its source rather than by a global filter.
    """
    calls: DrawnCalls = []
    real = Axes.plot

    def record(ax: Axes, *args: Any, **kwargs: Any) -> list[Any]:
        calls.append((args, kwargs))
        return cast("list[Any]", real(ax, *args, **kwargs))

    monkeypatch.setattr(Axes, "plot", record)
    return calls


class TestLossCurves:
    def test_validation_is_drawn_once(self, drawn: DrawnCalls, tmp_path: Path) -> None:
        plot_losses(_history(), save_path=tmp_path / "losses.pdf")

        labels = [k.get("label", "") for _, k in drawn]
        assert sum(label.startswith("Val") for label in labels) == 1
        assert labels == ["Train D", "Train G", "Val D"]

    def test_no_two_curves_carry_the_same_data(
        self, drawn: DrawnCalls, tmp_path: Path
    ) -> None:
        """The actual regression: `val_g` was a literal copy of `val_d`."""
        plot_losses(_history(), save_path=tmp_path / "losses.pdf")

        series = [
            np.asarray(a=cast("list[float]", args[1]), dtype=np.double)
            for args, _ in drawn
        ]
        for i, first in enumerate(series):
            for second in series[i + 1 :]:
                assert not np.array_equal(first, second)

    def test_a_legacy_four_column_history_still_plots(
        self, drawn: DrawnCalls, tmp_path: Path
    ) -> None:
        """Runs saved before the merge carry a `val_g` key holding a copy of
        `val_d`. `--load-run` replots them, so reading it must stay optional ---
        and it must stay unread, or the duplicate comes back."""
        legacy = _history()
        legacy["val_g"] = list(legacy["val_d"])

        plot_losses(legacy, save_path=tmp_path / "losses.pdf")

        assert [k.get("label", "") for _, k in drawn] == ["Train D", "Train G", "Val D"]


def _drawn_losses(history: dict[str, list[float]], tmp_path: Path) -> Figure:
    """Render `plot_losses` for real and return the Figure it built.

    `plot_losses` returns nothing, so capturing the Figure means intercepting
    `Figure.add_subplot` the way `captured_axes` does for `plot_selection` ---
    this lets matplotlib actually compute ticks and limits rather than mocking
    the draw away, which is where the fixed-axis regression lives.
    """
    captured: list[Axes] = []
    original_add_subplot = Figure.add_subplot

    def record_and_call(self: Figure, *args: Any, **kwargs: Any) -> Axes:
        ax = original_add_subplot(self, *args, **kwargs)
        captured.append(ax)
        return ax

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "add_subplot", record_and_call)
        plot_losses(history, save_path=tmp_path / "losses.pdf")

    return cast("Figure", captured[0].figure)


class TestLossAxis:
    def test_the_loss_axis_is_fixed_around_log_two(self, tmp_path: Path) -> None:
        """Autoscaling turned a 0.4% band into an apparent divergence."""
        history = {
            "train_d": [0.689, 0.688],
            "train_g": [0.688, 0.687],
            "val_d": [0.690, 0.693],
        }
        figure: Figure = _drawn_losses(history, tmp_path)
        ax = figure.axes[0]

        low, high = ax.get_ylim()
        assert low == pytest.approx(math.log(2) * (1 - 2**-4))
        assert high == pytest.approx(math.log(2) * (1 + 2**-4))

    def test_log_two_is_a_tick_and_not_a_legend_entry(self, tmp_path: Path) -> None:
        history = {"train_d": [0.689], "train_g": [0.688], "val_d": [0.690]}
        ax = _drawn_losses(history, tmp_path).axes[0]

        legend = ax.get_legend()
        assert legend is not None
        assert "log(2)" not in [t.get_text() for t in legend.get_texts()]
        assert any(t == pytest.approx(math.log(2)) for t in ax.get_yticks())

    def test_the_y_label_has_a_space_in_it(self, tmp_path: Path) -> None:
        history = {"train_d": [0.689], "train_g": [0.688], "val_d": [0.690]}
        assert _drawn_losses(history, tmp_path).axes[0].get_ylabel() == "Weighted BCE"


@pytest.fixture
def captured_axes(monkeypatch: pytest.MonkeyPatch) -> list[Axes]:
    """Capture every Axes `plot_selection` builds via `Figure.add_subplot`,
    without mocking away the real rendering -- unlike `drawn`, this fixture
    lets matplotlib actually compute scales, transforms and view limits, which
    is what the symlog/log regression lives in.
    """
    captured: list[Axes] = []
    original_add_subplot = Figure.add_subplot

    def record_and_call(self: Figure, *args: Any, **kwargs: Any) -> Axes:
        ax = original_add_subplot(self, *args, **kwargs)
        captured.append(ax)
        return ax

    monkeypatch.setattr(Figure, "add_subplot", record_and_call)
    return captured


def _noisy_history(n_epochs: int = 40) -> dict[str, list[float]]:
    """A synthetic history with the oscillation the real criterion has:
    detector MMD^2 falling on average but noisy by a wide factor epoch to
    epoch, a correlated particle-level curve, and a decaying ESS."""
    rng = np.random.default_rng(0)
    epochs = np.arange(n_epochs, dtype=np.double)
    trend = 0.05 * np.exp(-epochs / 15) + 1e-4
    detector = trend * rng.uniform(0.3, 1.7, size=n_epochs)
    particle = trend * rng.uniform(0.3, 1.7, size=n_epochs) + 0.02
    ess = 900.0 * np.exp(-epochs / 60)
    return {
        "train_d": [0.69] * n_epochs,
        "train_g": [0.69] * n_epochs,
        "val_d": [0.69] * n_epochs,
        "val_mmd": detector.tolist(),
        "val_mmd_particle": particle.tolist(),
        "val_ess": ess.tolist(),
    }


def _drawn_selection(
    history: dict[str, list[float]], best_epoch: int, path: Path
) -> Figure:
    """Render `plot_selection` for real and return the Figure it built, by
    intercepting `Figure.add_subplot` the way `captured_axes` does -- this
    lets matplotlib actually compute positions and extents rather than
    mocking the draw away."""
    captured: list[Axes] = []
    original_add_subplot = Figure.add_subplot

    def record_and_call(self: Figure, *args: Any, **kwargs: Any) -> Axes:
        ax = original_add_subplot(self, *args, **kwargs)
        captured.append(ax)
        return ax

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Figure, "add_subplot", record_and_call)
        plot_selection(history, best_epoch=best_epoch, save_path=path)

    figure = cast("Figure", captured[0].figure)
    figure.canvas = FigureCanvasAgg(figure)
    figure.canvas.draw()
    return figure


class TestSelectionPlot:
    def test_selection_plot_survives_a_missing_particle_curve(
        self, tmp_path: Path
    ) -> None:
        """A real measurement has no truth, so the particle curve is optional."""
        history = {
            "train_d": [0.69] * 5,
            "train_g": [0.69] * 5,
            "val_d": [0.69] * 5,
            "val_mmd": [0.05, 0.03, 0.01, 0.02, 0.04],
            "val_ess": [900.0, 850.0, 800.0, 700.0, 600.0],
        }
        out = tmp_path / "selection.pdf"
        plot_selection(history, best_epoch=2, save_path=out)
        assert out.exists()
        assert out.stat().st_size > 0

        history["val_mmd_particle"] = [0.09, 0.07, 0.06, 0.06, 0.07]
        plot_selection(history, best_epoch=2, save_path=out)
        assert out.exists()

    def test_negative_and_zero_mmd_survive_a_log_style_axis(
        self, captured_axes: list[Axes], tmp_path: Path
    ) -> None:
        """`weighted_mmd` is the unbiased estimator, negative roughly half the
        time once the distributions actually match, since MMD^2 is 0 at
        P = Q. A plain log axis silently masks non-positive values -- the
        line vanishes and the view autoscales away from them -- exactly in
        the neighbourhood of a converged run's minimum. This pins both the
        axis scale and the actual view limits, since the masking happens at
        render/autoscale time, not by mutating the stored data: a check on
        `Line2D.get_ydata()` alone would pass under the old `set_yscale("log")`
        code too.
        """
        history = {
            "train_d": [0.69] * 6,
            "train_g": [0.69] * 6,
            "val_d": [0.69] * 6,
            "val_mmd": [6.6e-4, -9.3e-4, -1.0e-4, 2.0e-4, -5.0e-4, 0.0],
            "val_ess": [900.0] * 6,
        }
        plot_selection(history, best_epoch=2, save_path=tmp_path / "selection.pdf")

        ax = captured_axes[0]
        assert ax.get_yscale() != "log"

        mmd_line = ax.get_lines()[0]
        ydata = np.asarray(mmd_line.get_ydata(), dtype=np.double)
        assert (ydata < 0).any()

        ax.figure.canvas.draw()
        ylim = ax.get_ylim()
        assert ylim[0] < 0, (
            "y-limits must reach the negative data; a log axis clips the view "
            "to the smallest positive value and silently drops the rest"
        )

    def test_selection_splits_mmd_and_ess_into_two_panels(self, tmp_path: Path) -> None:
        """Three noisy series on one axis with a twin scale read as
        seismographs."""
        figure = _drawn_selection(
            _noisy_history(), best_epoch=38, path=tmp_path / "selection.pdf"
        )
        panels = [a for a in figure.axes if a.get_xlabel() or a.get_ylabel()]

        assert any("MMD" in a.get_ylabel() for a in panels)
        assert any("Effective sample size" in a.get_ylabel() for a in panels)
        # The ESS panel is its own axes, not a twin of the MMD one.
        ess = next(a for a in panels if "Effective sample size" in a.get_ylabel())
        mmd = next(a for a in panels if "MMD" in a.get_ylabel())
        assert ess.get_position().y1 <= mmd.get_position().y0 + 1e-6

    def test_the_correlation_inset_appears_only_with_truth(
        self, tmp_path: Path
    ) -> None:
        """A real measurement has no particle-level curve to scatter
        against."""
        with_truth = _drawn_selection(_noisy_history(), 38, tmp_path / "a.pdf")
        history = _noisy_history()
        del history["val_mmd_particle"]
        without = _drawn_selection(history, 38, tmp_path / "b.pdf")

        # The scatter is its own top-level axes (in the right column) rather
        # than an `inset_axes` over the MMD panel -- see the "hides the data
        # it sits on" fix below -- so its presence shows up directly in
        # `Figure.axes`.
        assert len(with_truth.axes) == len(without.axes) + 1

    def test_the_legend_is_outside_the_axes(self, tmp_path: Path) -> None:
        """It used to cover the bottom third of the plot."""
        figure = _drawn_selection(_noisy_history(), 38, tmp_path / "selection.pdf")
        canvas = cast("FigureCanvasAgg", figure.canvas)
        mmd = next(a for a in figure.axes if a.get_ylabel() == r"MMD$^2$")
        legend_ax = next(a for a in figure.axes if a.get_legend() is not None)
        legend = legend_ax.get_legend()
        assert legend is not None
        renderer = canvas.get_renderer()
        legend_box = legend.get_window_extent(renderer)
        axes_box = mmd.get_window_extent(renderer)
        assert not legend_box.overlaps(axes_box)

    def test_the_scatter_does_not_hide_criterion_points(self, tmp_path: Path) -> None:
        """The original complaint was a legend covering the bottom third of
        the plot; an opaque inset scatter sitting on top of the MMD curves
        it explains is the same defect wearing a different shape."""
        figure = _drawn_selection(_noisy_history(), 38, tmp_path / "selection.pdf")
        canvas = cast("FigureCanvasAgg", figure.canvas)
        renderer = canvas.get_renderer()
        mmd = next(a for a in figure.axes if a.get_ylabel() == r"MMD$^2$")
        scatter_ax = next(
            a for a in figure.axes if a.get_xlabel() == "Detector MMD$^2$"
        )
        scatter_box = scatter_ax.get_window_extent(renderer)

        for line in mmd.get_lines():
            points = mmd.transData.transform(
                np.column_stack([line.get_xdata(), line.get_ydata()])
            )
            inside = (
                (points[:, 0] >= scatter_box.x0)
                & (points[:, 0] <= scatter_box.x1)
                & (points[:, 1] >= scatter_box.y0)
                & (points[:, 1] <= scatter_box.y1)
            )
            assert not inside.any()

    def test_the_mmd_panel_is_not_mostly_empty_floor(self, tmp_path: Path) -> None:
        """The resolution floor used to set the view's lower bound at zero
        regardless of the data, leaving most of the panel empty and the
        curves crushed into the top third."""
        figure = _drawn_selection(_noisy_history(), 38, tmp_path / "selection.pdf")
        canvas = cast("FigureCanvasAgg", figure.canvas)
        renderer = canvas.get_renderer()
        mmd = next(a for a in figure.axes if a.get_ylabel() == r"MMD$^2$")
        axes_box = mmd.get_window_extent(renderer)

        detector = np.asarray(_noisy_history()["val_mmd"], dtype=np.double)
        min_point_y = mmd.transData.transform((0.0, detector.min()))[1]
        # Display y grows upward from the axes' bottom edge (`y0`); the
        # fraction of the panel *below* the lowest data point is the gap
        # between that edge and the point's pixel row.
        below_fraction: float = (min_point_y - axes_box.y0) / axes_box.height
        assert below_fraction < 0.25

    def test_negative_best_epoch_skips_the_selection_marker(
        self, captured_axes: list[Axes], tmp_path: Path
    ) -> None:
        """`best_epoch` and `val_mmd` are read from different files
        (`config.json` vs `history.npz`) and can diverge; `best_epoch`
        defaults to -1. `best_epoch + 1` in the label would then read
        "selected (epoch 0)" while the axvline lands off the left edge."""
        history = {
            "train_d": [0.69] * 5,
            "train_g": [0.69] * 5,
            "val_d": [0.69] * 5,
            "val_mmd": [0.05, 0.03, 0.01, 0.02, 0.04],
            "val_ess": [900.0, 850.0, 800.0, 700.0, 600.0],
        }
        plot_selection(history, best_epoch=-1, save_path=tmp_path / "selection.pdf")

        ax = captured_axes[0]
        _, labels = ax.get_legend_handles_labels()
        assert not any(label.startswith("selected") for label in labels)


class TestBaselineOverlays:
    """Two baselines on one panel, and the paint order between them.

    The overlay list replaced a single `ibu_weights` parameter. What that
    change can break is not whether a curve appears --- an assertion on the
    legend catches that --- but which curve ends up *visible* where they agree,
    which no test was previously in a position to notice.
    """

    def test_omnifold_repeats_one_vector_across_dimensions(self) -> None:
        """OmniFold reweights events, not observables.

        IBU's weights genuinely differ per observable and are passed as a list;
        OmniFold's single vector has to reach every panel, and `from_shared`
        is the only place that is said.
        """
        weights = np.asarray([0.5, 1.5], dtype=np.single)
        overlay = omnifold_overlay(weights, 4)

        assert len(overlay.weights) == 4
        for i in range(4):
            assert np.array_equal(overlay.at(i).weights, weights)

    def test_the_two_baselines_differ_in_linestyle_as_well_as_colour(self) -> None:
        """Panels are printed in greyscale, where colour alone says nothing."""
        ibu = ibu_overlay([np.ones(2, dtype=np.single)])
        omnifold = omnifold_overlay(np.ones(2, dtype=np.single), 1)

        assert ibu.color != omnifold.color
        assert ibu.linestyle != omnifold.linestyle
        assert ibu.marker != omnifold.marker

    def test_ran_is_painted_above_every_baseline(self) -> None:
        """RAN is the method being showcased; it must not sit under a baseline.

        The overlays are drawn *after* RAN so they read last in the legend, and
        at linewidth 4 the last one drawn would otherwise bury RAN wherever the
        curves agree --- which on a converged run is everywhere. Caught by
        rendering the figure and looking at it, not by a green suite.
        """
        figure = Figure()
        ax = figure.add_subplot(211)
        ax_r = figure.add_subplot(212)
        nature = np.array([0.1, 0.3, 0.6, 0.8], dtype=np.single)
        mc = np.array([0.2, 0.4, 0.5, 0.9], dtype=np.single)
        ones = np.ones(4, dtype=np.single)

        _hist_ratio_panel(
            ax,
            ax_r,
            nature,
            mc,
            ones,
            bins=[0.0, 0.25, 0.5, 0.75, 1.0],
            nature_label="Data",
            mc_label="Sim",
            xlabel="x",
            title="Detector level",
            overlays=[
                ibu_overlay([ones]).at(0),
                omnifold_overlay(ones, 1).at(0),
            ],
        )

        by_label = {
            label: artist
            for artist, label in zip(*ax.get_legend_handles_labels(), strict=True)
        }
        ran_z = by_label["RAN"].get_zorder()
        assert ran_z > by_label["IBU"].get_zorder()
        assert ran_z > by_label["OmniFold"].get_zorder()

    def test_every_panel_of_a_page_carries_both_baselines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The overlays are indexed per panel; an off-by-one would drop one."""
        monkeypatch.setattr("deconvolve.evaluation.plotting._save_pages", _capture_save)
        rng = np.random.default_rng(0)
        dim = 3
        nature = rng.normal(size=(64, dim)).astype(np.single)
        mc = rng.normal(size=(64, dim)).astype(np.single)
        w = np.ones(64, dtype=np.single)

        _plot_level(
            nature,
            mc,
            w,
            _DETECTOR,
            tmp_path / "levels.pdf",
            None,
            [
                ibu_overlay([np.ones(64, dtype=np.single) for _ in range(dim)]),
                omnifold_overlay(np.ones(64, dtype=np.single), dim),
            ],
        )

        panels = [ax for ax in _drawn_pages()[0].get_axes() if ax.get_title()]
        assert len(panels) == dim
        for panel in panels:
            _, labels = panel.get_legend_handles_labels()
            assert labels == ["Data", "Sim", "RAN", "IBU", "OmniFold"]
