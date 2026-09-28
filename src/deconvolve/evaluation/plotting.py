from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, cast

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf, PdfPages
from matplotlib.figure import Figure
from matplotlib.font_manager import fontManager
from matplotlib.ticker import MaxNLocator

from ..coretypes import (
    LOG2,
    PANEL_COLUMNS,
    PANEL_WIDTH_INCHES,
    PANELS_PER_PAGE,
    display_order,
)
from .evaluate import _get_weights

if TYPE_CHECKING:
    from logging import Logger
    from typing import Final

    from matplotlib.axes import Axes
    from matplotlib.container import BarContainer
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec, SubplotSpec
    from matplotlib.patches import Polygon
    from numpy.typing import NDArray

    from ..coretypes import DeconvolveModel, EventArray, Populations, VarInfo
    from ..data import ArrayDataset

type AxesHist = tuple[
    NDArray[np.double],
    NDArray[np.double],
    BarContainer | Polygon | list[BarContainer | Polygon],
]

logger: Logger = logging.getLogger(name=__name__)

mpl.rcParams["font.family"] = "serif"
available_fonts: set[str] = {f.name for f in fontManager.ttflist}
if "Cochineal" in available_fonts:
    mpl.rcParams["font.serif"] = ["Cochineal"]
mpl.rcParams["font.size"] = 18
mpl.rcParams["text.usetex"] = False
# TrueType rather than matplotlib's default Type 3. Both are vector, but Type 3
# embeds glyphs as bare charprocs with no dependable encoding, so text in the
# saved PDF cannot be searched, copied or re-edited as text -- and arXiv and
# several publishers reject it outright. Read at save time, so this one setting
# covers `_save_fig` and `_save_pages` alike.
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "0.85"
mpl.rcParams["grid.linewidth"] = 0.5
mpl.rcParams["grid.alpha"] = 0.6
mpl.rcParams["grid.linestyle"] = "--"
mpl.rcParams["lines.markerfacecolor"] = "none"


# One place for the figure's visual hierarchy, rather than seven literals
# scattered through `_hist_ratio_panel`: RAN's curve must read as the method
# under test, not a baseline drawn at equal or greater prominence.
COLOR_NATURE: Final[str] = "C0"  # Data / Truth
COLOR_MC: Final[str] = "C1"  # Sim / Gen
COLOR_IBU: Final[str] = "green"
COLOR_OMNIFOLD: Final[str] = "#E31A1C"  # crimson; the third baseline curve
COLOR_RAN: Final[str] = "#6A3D9A"  # deep violet; greyscales to a dark mid-tone

ALPHA_FILL: Final[float] = 0.35  # the two filled background histograms
ALPHA_IBU: Final[float] = 0.75
ALPHA_OMNIFOLD: Final[float] = 0.75
ALPHA_RAN: Final[float] = 0.90

# Paint order, which is not legend order: baselines are created after RAN so
# they read last in the legend, but must not paint over it. Matplotlib's
# default for lines is 2.
Z_BASELINE: Final[int] = 2
Z_RAN: Final[int] = 3


# `weighted_mmd` is the unbiased U-statistic estimator, which is negative
# roughly half the time once the two distributions actually match (MMD^2 is 0
# when P = Q, so an unbiased estimator of it must cross zero). A converged
# run's criterion curve therefore has values around and below zero right where
# selection lands. A plain log axis silently masks non-positive values, which
# hides exactly the epochs a converged run cares about; `symlog` renders those
# linearly while keeping the log compression that makes the early, large
# epochs readable. `SELECTION_MMD_LINTHRESH` sets where that linear region
# starts -- near the estimator's resolution floor (~5e-4 at m=8192, see
# `train.MMD_SUBSAMPLE`), so the linear region roughly matches the noise band
# rather than being an arbitrary cutoff.
SELECTION_MMD_LINTHRESH: Final[float] = 5e-4

# Width of the centred rolling median drawn over the raw MMD traces in
# `plot_selection`. The raw series stays visible underneath at low alpha --
# this is a legibility aid, not a smoothing of the reported criterion.
SELECTION_SMOOTHING_WINDOW: Final[int] = 5

# The equilibrium band. Every series a converged run produces sits within a
# fraction of a percent of `ln 2`, and autoscaling that band to the height of
# the axes makes a 0.4% drift look like a divergence. Fixed limits also make
# two runs' loss plots directly comparable.
LOSS_YLIM_FRACTION: Final[float] = 2.0**-4


class _PanelSpec(NamedTuple):
    """Everything that varies between the panels of one figure."""

    nature: EventArray
    mc: EventArray
    bins: NDArray[np.double]
    xlabel: str
    title: str


class _LevelStyle(NamedTuple):
    """Everything that differs between the detector-level and particle-level figures."""

    level: str  # "detector" / "particle", used in axis labels
    symbol: str  # "x" / "z"
    title_prefix: str  # "Detector Level" / "Particle Level"
    nature_label: str  # legend entry for the reference sample
    mc_label: str  # legend entry for the simulated sample
    height_per_dim: float  # panel height in inches; see PANEL_WIDTH_INCHES
    bins_span_both: bool  # default binning covers both samples, not just nature


class _PanelOverlay(NamedTuple):
    """One baseline's curve on one panel: weights already picked per dimension."""

    label: str
    weights: EventArray
    color: str
    linestyle: str
    marker: str
    alpha: float


class BaselineOverlay(NamedTuple):
    """A comparison baseline's weights, and how its curve is drawn.

    A list of these threads through `plot_levels` and `_hist_ratio_panel`,
    so a third baseline costs one constructor rather than a new parameter in
    every signature between them.

    `weights` holds one full-length weight vector **per dimension**, because
    IBU unfolds each observable separately and its weights genuinely differ
    between them. A method producing one vector for every observable, as
    OmniFold and RAN do, repeats it; `from_shared` is that, said once.
    """

    label: str
    weights: list[EventArray]
    color: str
    linestyle: str
    marker: str
    alpha: float

    @classmethod
    def from_shared(
        cls,
        label: str,
        weights: EventArray,
        dim: int,
        color: str,
        linestyle: str,
        marker: str,
        alpha: float,
    ) -> BaselineOverlay:
        """A baseline whose single weight vector applies to every dimension."""
        return cls(label, [weights] * dim, color, linestyle, marker, alpha)

    def at(self, i: int, /) -> _PanelOverlay:
        return _PanelOverlay(
            self.label,
            self.weights[i],
            self.color,
            self.linestyle,
            self.marker,
            self.alpha,
        )


def ibu_overlay(weights: list[EventArray]) -> BaselineOverlay:
    """IBU: dotted, green, square markers. One weight vector per observable."""
    return BaselineOverlay(
        label="IBU",
        weights=weights,
        color=COLOR_IBU,
        linestyle=":",
        marker="s",
        alpha=ALPHA_IBU,
    )


def omnifold_overlay(weights: EventArray, dim: int) -> BaselineOverlay:
    """OmniFold: dash-dot, crimson, triangles.

    One weight vector covers every observable --- OmniFold reweights events,
    not observables --- so it is repeated across the dimensions rather than
    indexed. Distinguished from IBU by linestyle as well as colour, so the
    panels survive being printed in greyscale.
    """
    return BaselineOverlay.from_shared(
        label="OmniFold",
        weights=weights,
        dim=dim,
        color=COLOR_OMNIFOLD,
        linestyle="-.",
        marker="^",
        alpha=ALPHA_OMNIFOLD,
    )


def _collect_data(dataset: ArrayDataset) -> Populations:
    """Return the split as the four physics populations, each (n, dim)."""
    return dataset.as_arrays().partition()


def _hist_ratio_panel(
    ax: Axes,
    ax_r: Axes,
    x_nature: EventArray,
    x_mc: EventArray,
    w_ran: EventArray,
    bins: Sequence[float] | int,
    nature_label: str,
    mc_label: str,
    xlabel: str,
    title: str,
    overlays: Sequence[_PanelOverlay] = (),
) -> None:
    h_nature: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_nature,
            bins=bins,
            histtype="stepfilled",
            alpha=ALPHA_FILL,
            color=COLOR_NATURE,
            label=nature_label,
        ),
    )
    h_mc: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_mc,
            bins=cast(typ=Sequence[float], val=h_nature[1]),
            histtype="stepfilled",
            alpha=ALPHA_FILL,
            color=COLOR_MC,
            label=mc_label,
        ),
    )
    h_ran: AxesHist = cast(
        typ=AxesHist,
        val=ax.hist(
            x_mc,
            bins=cast(typ=Sequence[float], val=h_nature[1]),
            weights=w_ran,
            histtype="step",
            color=COLOR_RAN,
            linestyle="-",
            linewidth=4,
            alpha=ALPHA_RAN,
            label="RAN",
            # Above every baseline. Overlays draw after this call, putting
            # them last in the legend, but at linewidth 4 the last curve
            # drawn would otherwise bury RAN wherever they agree -- which on
            # a converged run is everywhere. `zorder` keeps paint order
            # separate from legend order.
            zorder=Z_RAN,
        ),
    )

    bin_edges: NDArray[np.double] = h_nature[1]
    centres: NDArray[np.double] = (bin_edges[:-1] + bin_edges[1:]) / 2
    safe: NDArray[np.bool] = h_nature[0] > 0
    ratio_mc: NDArray[np.double] = np.full_like(
        a=h_nature[0],
        fill_value=np.nan,
        dtype=np.double,
    )
    ratio_ran: NDArray[np.double] = np.full_like(
        a=h_ran[0],
        fill_value=np.nan,
        dtype=np.double,
    )
    ratio_mc[safe] = h_mc[0][safe] / h_nature[0][safe]
    ratio_ran[safe] = h_ran[0][safe] / h_nature[0][safe]

    _ = ax_r.plot(
        centres,
        ratio_mc,
        color=COLOR_MC,
        marker="d",
        linestyle="--",
        alpha=ALPHA_FILL,
    )
    _ = ax_r.plot(
        centres,
        ratio_ran,
        color=COLOR_RAN,
        marker="o",
        linestyle="--",
        alpha=ALPHA_RAN,
        zorder=Z_RAN,
    )

    for overlay in overlays:
        h_baseline: AxesHist = cast(
            typ=AxesHist,
            val=ax.hist(
                x_mc,
                bins=cast(typ=Sequence[float], val=h_nature[1]),
                weights=overlay.weights,
                histtype="step",
                color=overlay.color,
                linestyle=overlay.linestyle,
                linewidth=4,
                alpha=overlay.alpha,
                label=overlay.label,
                zorder=Z_BASELINE,
            ),
        )
        ratio_baseline: NDArray[np.double] = np.full_like(
            a=h_baseline[0], fill_value=np.nan, dtype=np.double
        )
        ratio_baseline[safe] = h_baseline[0][safe] / h_nature[0][safe]
        _ = ax_r.plot(
            centres,
            ratio_baseline,
            color=overlay.color,
            marker=overlay.marker,
            linestyle="--",
            alpha=overlay.alpha,
            zorder=Z_BASELINE,
        )
    # Every panel gets a label, a title and a legend, whether or not a
    # baseline overlay was drawn: `ax.legend()` runs once here, after the
    # overlay loop, and picks up whichever handles that loop created.
    _ = ax.set_ylabel(ylabel="Events")
    _ = ax.set_title(label=title)
    _ = ax.legend()
    _ = ax_r.axhline(y=1, color="gray", linewidth=0.5, alpha=0.75)
    width: float = 0.5
    _ = ax_r.set_ylim(bottom=1 - width, top=1 + width)
    _ = ax_r.set_ylabel(ylabel=f"Ratio to\n{nature_label}")
    # The main panel's bottom tick and the ratio panel's top tick land at the
    # same height where the two axes meet and overprint each other. `prune`
    # drops the lowest label only when it sits at the axis edge, which is
    # exactly the collision and nothing else.
    ax.yaxis.set_major_locator(locator=MaxNLocator(prune="lower"))
    # Event counts run to five and six digits, and a column of "64000" labels
    # costs more panel width than the numbers are worth. Scientific notation
    # factors the magnitude out into a single `x10^4` above the axis and
    # leaves one- or two-digit ticks. Counts axis only: the ratio panel sits
    # around 1, where an offset would be absurd.
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 3), useMathText=True)
    _ = ax_r.set_xlabel(xlabel)


def _save_fig(figure: Figure, save_path: Path) -> None:
    """Save `figure`, trimmed to its rendered contents.

    Without `bbox_inches="tight"` the y-labels are clipped by the page edge:
    wide tick labels (e.g. five-digit event counts) can push a y-label
    further left than a fixed `GridSpec` margin reserves for it. Every save
    path goes through here rather than calling `figure.savefig` directly, so
    the bbox handling cannot drift out of sync between figures.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(fname=save_path, bbox_inches="tight")
    logger.info("Saved %s", save_path)


def _save_pages(figures: Sequence[Figure], /, *, save_path: Path) -> None:
    r"""Save `figures` as the successive pages of one PDF.

    One file rather than `detector_level_1.pdf`, `_2.pdf`, ...: the run
    directory keeps a single artifact per level, `\\includegraphics[page=k]`
    selects a page, and `rantypes.figure_pages` tells the report how many
    there are without opening the file. Each page is trimmed exactly as
    `_save_fig` trims a single figure -- see that docstring for why the tight
    bbox is not optional.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(filename=save_path) as pdf:
        for figure in figures:
            pdf.savefig(figure=figure, bbox_inches="tight")
    logger.info("Saved %s (%d page(s))", save_path, len(figures))


_DETECTOR = _LevelStyle(
    level="detector",
    symbol="x",
    title_prefix="Detector Level",
    nature_label="Data",
    mc_label="Sim",
    height_per_dim=6.6,
    bins_span_both=False,
)
_PARTICLE = _LevelStyle(
    level="particle",
    symbol="z",
    title_prefix="Particle Level",
    nature_label="Truth",
    mc_label="Gen.",
    height_per_dim=6.6,
    bins_span_both=True,
)


def _panel_spec(
    i: int,
    dim: int,
    nature: EventArray,
    mc: EventArray,
    var_info: list[VarInfo] | None,
    style: _LevelStyle,
) -> _PanelSpec:
    """Decide what dimension `i` shows: the arrays, binning, and labels."""
    if var_info is not None:
        cfg: VarInfo = var_info[i]
        mu: float = cfg["mu"]
        sigma: float = cfg["sigma"]
        return _PanelSpec(
            nature=nature[:, i] * sigma + mu,
            mc=mc[:, i] * sigma + mu,
            bins=np.linspace(start=cfg["xlim"][0], stop=cfg["xlim"][1], num=21),
            xlabel=cfg["symbol"],
            # No "(detector/particle level)" suffix here: at a 4-inch panel
            # width in the grid `_plot_level` lays out, repeating it on all
            # twelve panels made adjacent titles overlap (worst case 109.5px).
            # `_plot_level`'s `figure.suptitle` states the level once for the
            # whole figure instead.
            title=cfg["xlabel"],
        )

    nature_i: EventArray = nature[:, i]
    mc_i: EventArray = mc[:, i]
    lo: np.single = (
        min(nature_i.min(), mc_i.min()) if style.bins_span_both else nature_i.min()
    )
    hi: np.single = (
        max(nature_i.max(), mc_i.max()) if style.bins_span_both else nature_i.max()
    )
    return _PanelSpec(
        nature=nature_i,
        mc=mc_i,
        bins=np.linspace(start=lo, stop=hi, num=51),
        xlabel=(
            f"${style.symbol}_{{{i}}}$ ({style.level} level)"
            if dim > 1
            else f"{style.symbol} ({style.level} level)"
        ),
        # `_plot_level`'s `figure.suptitle` already states `style.title_prefix`
        # once for the whole figure, so a panel title would duplicate it.
        # `dim > 1` still names which dimension a panel is; `dim == 1` has
        # nothing left to say.
        title=(f"Dim {i}" if dim > 1 else ""),
    )


def _draw_panel(
    figure: Figure,
    cell: SubplotSpec,
    i: int,
    dim: int,
    nature: EventArray,
    mc: EventArray,
    w: EventArray,
    var_info: list[VarInfo] | None,
    style: _LevelStyle,
    baselines: Sequence[BaselineOverlay],
) -> None:
    """Draw dimension `i`'s stacked hist+ratio panel into `cell`."""
    inner_grid: GridSpecFromSubplotSpec = cell.subgridspec(
        nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.0
    )
    ax: Axes = figure.add_subplot(inner_grid[0])
    ax_r: Axes = figure.add_subplot(inner_grid[1], sharex=ax)
    ax.tick_params(labelbottom=False)

    panel: _PanelSpec = _panel_spec(i, dim, nature, mc, var_info, style)
    _hist_ratio_panel(
        ax,
        ax_r,
        x_nature=panel.nature,
        x_mc=panel.mc,
        w_ran=w,
        bins=panel.bins.tolist(),
        nature_label=style.nature_label,
        mc_label=style.mc_label,
        xlabel=panel.xlabel,
        title=panel.title,
        overlays=[baseline.at(i) for baseline in baselines],
    )


def _page_figure(
    indices: Sequence[int],
    page: int,
    pages: int,
    dim: int,
    nature: EventArray,
    mc: EventArray,
    w: EventArray,
    var_info: list[VarInfo] | None,
    style: _LevelStyle,
    baselines: Sequence[BaselineOverlay],
) -> Figure:
    """One page of the level figure: up to `PANELS_PER_PAGE` panels."""
    ncols: int = min(PANEL_COLUMNS, len(indices))
    nrows = int(np.ceil(len(indices) / ncols))
    figure = Figure(figsize=(PANEL_WIDTH_INCHES * ncols, style.height_per_dim * nrows))
    figure.canvas = FigureCanvasPdf(figure)
    # Absolute margins in inches do not survive a figure whose height varies
    # with `nrows`; `tight_layout` at the end replaces them. Row/column spacing
    # goes through `tight_layout`'s own `h_pad` below rather than an `hspace=`
    # here: passing `hspace` marks this `GridSpec` as "locally modified"
    # (`GridSpec.locally_modified_subplot_params`), which makes `tight_layout`
    # treat every nested Axes as unrecognized and silently fall back to
    # Matplotlib's default (too-small) margins instead of computed ones --
    # visible as axis labels rendered off the left edge of the page.
    outer_grid: GridSpec = figure.add_gridspec(nrows=nrows, ncols=ncols)
    # States the level once per page instead of on every panel title -- see
    # `_panel_spec`. The page counter only appears when there is more than
    # one page.
    title: str = (
        style.title_prefix
        if pages == 1
        else f"{style.title_prefix} ({page} of {pages})"
    )
    _ = figure.suptitle(t=title, fontsize="x-large", y=0.995)

    for position, i in enumerate(iterable=indices):
        _draw_panel(
            figure,
            outer_grid[position // ncols, position % ncols],
            i,
            dim,
            nature,
            mc,
            w,
            var_info,
            style,
            baselines,
        )
    # `rect`'s top leaves a fixed-fraction band for the suptitle that
    # `tight_layout`'s own margin computation does not know to reserve --
    # verified (see the test below) not to collide with the top row's panel
    # titles across 1-, 2- and 12-panel grids.
    figure.tight_layout(h_pad=2.0, rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _plot_level(
    nature: EventArray,
    mc: EventArray,
    w: EventArray,
    style: _LevelStyle,
    save_path: str | Path,
    var_info: list[VarInfo] | None,
    baselines: Sequence[BaselineOverlay],
    variables: tuple[str, ...] | None = None,
) -> None:
    r"""Draw one stacked hist+ratio panel per dimension, paginated.

    Panels are ordered by `display_order` on `variables` (or the `dim_i`
    identity for a non-jet run) rather than by raw column index, and split
    `PANELS_PER_PAGE` to a page across the pages of ONE multi-page PDF.

    Pagination keeps the panels legible: `\includegraphics` scales a figure
    to fit its text block and every font scales with it, so a twelve-panel
    12x24in figure would render its 18pt labels at 5pt. `figure_pages` gives
    the same per-page count `report.py` needs to know how many
    `\includegraphics[page=...]` blocks to emit, without opening the file.
    """
    dim: int = nature.shape[1]
    order: Sequence[int] = display_order(
        variables if variables is not None else [f"dim_{i}" for i in range(dim)]
    )
    chunks: list[Sequence[int]] = [
        order[i : i + PANELS_PER_PAGE] for i in range(0, len(order), PANELS_PER_PAGE)
    ]
    _save_pages(
        [
            _page_figure(
                chunk,
                page,
                len(chunks),
                dim,
                nature,
                mc,
                w,
                var_info,
                style,
                baselines,
            )
            for page, chunk in enumerate(iterable=chunks, start=1)
        ],
        save_path=Path(save_path),
    )


def plot_detector_level(
    test_dataset: ArrayDataset,
    g: DeconvolveModel,
    save_path: Path = Path("plots/detector_level.pdf"),
    var_info: list[VarInfo] | None = None,
    baselines: Sequence[BaselineOverlay] = (),
    variables: tuple[str, ...] | None = None,
) -> None:
    """Detector level: data against simulation, before and after reweighting."""
    test: Populations = _collect_data(test_dataset)

    _plot_level(
        nature=test.data,
        mc=test.mc.x,
        w=_get_weights(g, z_gen=test.mc.z),
        style=_DETECTOR,
        save_path=save_path,
        var_info=var_info,
        baselines=baselines,
        variables=variables,
    )


def plot_particle_level(
    test_dataset: ArrayDataset,
    g: DeconvolveModel,
    save_path: Path = Path("plots/particle_level.pdf"),
    var_info: list[VarInfo] | None = None,
    baselines: Sequence[BaselineOverlay] = (),
    variables: tuple[str, ...] | None = None,
) -> None:
    """Particle level: truth against generation, before and after reweighting."""
    test: Populations = _collect_data(test_dataset)

    _plot_level(
        nature=test.require_truth(),
        mc=test.mc.z,
        w=_get_weights(g, z_gen=test.mc.z),
        style=_PARTICLE,
        save_path=save_path,
        var_info=var_info,
        baselines=baselines,
        variables=variables,
    )


def plot_levels(
    test_dataset: ArrayDataset,
    g: DeconvolveModel,
    detector_path: Path = Path("plots/detector_level.pdf"),
    particle_path: Path = Path("plots/particle_level.pdf"),
    var_info: list[VarInfo] | None = None,
    baselines: Sequence[BaselineOverlay] = (),
    variables: tuple[str, ...] | None = None,
) -> None:
    """Draw both physics levels from one partition and generator evaluation."""
    test: Populations = _collect_data(test_dataset)
    weights: EventArray = _get_weights(g, z_gen=test.mc.z)
    _plot_level(
        nature=test.data,
        mc=test.mc.x,
        w=weights,
        style=_DETECTOR,
        save_path=detector_path,
        var_info=var_info,
        baselines=baselines,
        variables=variables,
    )
    _plot_level(
        nature=test.require_truth(),
        mc=test.mc.z,
        w=weights,
        style=_PARTICLE,
        save_path=particle_path,
        var_info=var_info,
        baselines=baselines,
        variables=variables,
    )


def plot_losses(
    history: dict[str, list[float]],
    save_path: Path = Path("plots/losses.pdf"),
) -> None:
    epochs: NDArray[np.uintc] = np.arange(len(history["train_d"]), dtype=np.uintc)

    figure: Figure = Figure(figsize=(8, 5))
    figure.canvas = FigureCanvasPdf(figure)
    # argument must be positional only.
    ax: Axes = figure.add_subplot(111)
    train_d: NDArray[np.double] = np.array(
        object=history["train_d"],
        dtype=np.double,
    )
    val_d: NDArray[np.double] = np.array(object=history["val_d"], dtype=np.double)
    train_g: NDArray[np.double] = np.array(object=history["train_g"], dtype=np.double)
    _ = ax.plot(epochs, train_d, label="Train D", color="C0", ls=":", lw=1)
    _ = ax.plot(epochs, train_g, label="Train G", color="C1", ls=":", lw=1)
    # One validation curve, because there is one validation number: `eval_step`
    # scores both networks with a single weighted BCE, so a "Val G" line would
    # be this one drawn twice. Older runs carry a `val_g` key holding exactly
    # that copy --- it is deliberately not read.
    _ = ax.plot(epochs, val_d, label="Val D", color="C0", ls="--", lw=3, alpha=0.5)
    _ = ax.axhline(y=LOG2, color="gray", lw=1)  # no `label`: it is a tick, not a series

    _ = ax.set_ylim(
        bottom=LOG2 * (1 - LOSS_YLIM_FRACTION), top=LOG2 * (1 + LOSS_YLIM_FRACTION)
    )
    # Invariant: `offsets` is symmetric and odd-length, which is the only
    # reason `len(offsets) // 2` is the index of the zero offset -- i.e. the
    # only reason the $\ln 2$ label below lands on the $\ln 2$ tick.
    offsets: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
    ticks: list[float] = [LOG2 * (1 + k * 2.0**-5) for k in offsets]
    _ = ax.set_yticks(ticks=ticks)
    _ = ax.set_yticklabels(
        labels=[
            r"$\ln 2$" if i == len(offsets) // 2 else f"{t:.4f}"
            for i, t in enumerate(iterable=ticks)
        ]
    )

    # The same positions as a percentage deviation, so a reader sees "within
    # 1% of equilibrium" without doing the arithmetic.
    deviation: Axes = ax.twinx()
    _ = deviation.set_ylim(*ax.get_ylim())
    _ = deviation.set_yticks(ticks=ticks)
    _ = deviation.set_yticklabels(
        labels=[f"{k * 2.0**-5 * 100:+.1f}%" for k in offsets]
    )
    _ = deviation.set_ylabel(ylabel=r"Deviation from $\ln 2$")

    _ = ax.set_xlabel(xlabel="Epoch")
    _ = ax.set_ylabel(ylabel="Weighted BCE")
    _ = ax.set_title(label="Training History")
    _ = ax.legend()

    figure.tight_layout()
    _save_fig(figure, save_path=Path(save_path))


def _rolling_median(values: NDArray[np.double], window: int, /) -> NDArray[np.double]:
    """Centred rolling median, edges held at the nearest full window.

    Invariant: `window` must be ODD. An even one pads `window // 2` on both
    sides and so shifts the output half a sample rather than centring it.
    The only caller passes `SELECTION_SMOOTHING_WINDOW`, which is 5.
    """
    pad: int = window // 2
    padded: NDArray[np.double] = np.pad(array=values, pad_width=pad, mode="edge")
    return np.array(
        object=[np.median(a=padded[i : i + window]) for i in range(values.size)],
        dtype=np.double,
    )


def _mmd_series(
    ax: Axes,
    epochs: NDArray[np.uintc],
    values: NDArray[np.double],
    *,
    color: str,
    ls: str,
    label: str,
) -> None:
    """Raw trace at low alpha, rolling median on top carrying the label."""
    _ = ax.plot(epochs, values, color=color, ls=ls, lw=1, alpha=0.3)
    smoothed: NDArray[np.double] = _rolling_median(values, SELECTION_SMOOTHING_WINDOW)
    _ = ax.plot(epochs, smoothed, color=color, ls=ls, lw=2, label=label)


def _mmd_scatter(ax: Axes, history: dict[str, list[float]], best_epoch: int) -> None:
    """Detector-vs-particle scatter: the correlation two overlaid noisy time
    series cannot show. Only drawn when a particle-level curve exists.

    Lives in its own axes in the figure's right column rather than as an
    `inset_axes` over the MMD panel: an opaque box sitting on top of the
    curves it is meant to explain would hide exactly the criterion points it
    is there to relate.
    """
    detector = np.array(history["val_mmd"], dtype=np.double)
    particle = np.array(history["val_mmd_particle"], dtype=np.double)
    _ = ax.scatter(detector, particle, s=8, alpha=0.6, color=COLOR_MC)
    if 0 <= best_epoch < detector.size:
        _ = ax.scatter(
            x=detector[best_epoch], y=particle[best_epoch], s=40, color="k", marker="x"
        )
    _ = ax.set_xlabel(xlabel="Detector MMD$^2$", fontsize="x-small")
    _ = ax.set_ylabel(ylabel="Particle MMD$^2$", fontsize="x-small")
    ax.tick_params(labelsize="x-small")


def _mmd_values(history: dict[str, list[float]]) -> NDArray[np.double]:
    """Every plotted raw MMD value, detector and particle (when present)."""
    detector = np.array(history["val_mmd"], dtype=np.double)
    if "val_mmd_particle" in history:
        particle = np.array(history["val_mmd_particle"], dtype=np.double)
        return np.concatenate([detector, particle])
    return detector


def _mmd_ylim(history: dict[str, list[float]]) -> tuple[float, float]:
    """Y-limits sized to the plotted data, not to the resolution floor.

    Fixing the view's lower bound to the floor's `ymin=0` would let an empty
    floor band dominate the panel on a run whose data sits well above it, and
    crush every curve into a fraction of the height. Padding 20% past the
    data's own min/max instead lets the floor be clipped by the view -- still
    drawn, just not the majority of the panel. Padding is taken as a fraction
    of `abs(value)` rather than a flat multiply, so it still widens (not
    narrows) the view when the unbiased MMD estimator's noise puts the
    extreme value below zero.
    """
    values: NDArray[np.double] = _mmd_values(history)
    data_min: np.double = values.min()
    data_max: np.double = values.max()
    bottom: float = data_min - 0.2 * np.abs(data_min)
    top: float = data_max + 0.2 * np.abs(data_max)
    return bottom, top


def _mmd_panel(ax: Axes, history: dict[str, list[float]], best_epoch: int) -> None:
    """Top panel: detector (criterion) and particle (diagnostic) MMD^2, each
    as a raw trace plus a rolling median, the resolution floor shaded, and the
    selected epoch marked. Y-limits are sized to the data (see `_mmd_ylim`);
    the legend is drawn separately, in the figure's right column."""
    epochs: NDArray[np.uintc] = np.arange(len(history["val_mmd"]), dtype=np.uintc)

    _mmd_series(
        ax,
        epochs,
        values=np.array(object=history["val_mmd"], dtype=np.double),
        color=COLOR_NATURE,
        ls="-",
        label="Detector MMD$^2$ (criterion)",
    )
    if "val_mmd_particle" in history:
        _mmd_series(
            ax,
            epochs,
            values=np.array(object=history["val_mmd_particle"], dtype=np.double),
            color=COLOR_IBU,
            ls="--",
            label="Particle MMD$^2$ (diagnostic)",
        )

    _ = ax.axhspan(
        ymin=0,
        ymax=SELECTION_MMD_LINTHRESH,
        color="0.85",
        zorder=0,
        label="estimator resolution floor",
    )
    if best_epoch >= 0:
        _ = ax.axvline(
            x=best_epoch,
            color="k",
            ls=":",
            lw=1,
            label=f"selected (epoch {best_epoch + 1})",
        )
    ax.set_yscale(value="symlog", linthresh=SELECTION_MMD_LINTHRESH)
    _ = ax.set_ylabel(ylabel=r"MMD$^2$")
    ax.tick_params(axis="x", labelbottom=False)
    _ = ax.set_ylim(*_mmd_ylim(history))
    _clip_ticks_to_view(ax)


def _clip_ticks_to_view(ax: Axes) -> None:
    """Drop major y-ticks the locator placed outside the current view.

    `SymmetricalLogLocator` generates a fixed decade ladder around
    `linthresh` regardless of how narrow the data range actually is -- a
    real run's detector MMD^2 can sit entirely within one decade, and the
    unfiltered ladder then draws tick labels far above the axes, off the top
    of the figure. `get_yticks()` already evaluates the locator against the
    current view; this just keeps the ones inside it.
    """
    low, high = ax.get_ylim()
    ticks: NDArray[np.double] = np.asarray(ax.get_yticks(), dtype=np.double)
    in_view: NDArray[np.double] = ticks[(ticks >= low) & (ticks <= high)]
    if in_view.size:
        _ = ax.set_yticks(ticks=in_view)


def _ess_panel(ax: Axes, history: dict[str, list[float]]) -> None:
    """Bottom panel: effective sample size as a percentage of epoch 0. A
    falling MMD bought by a collapsing ESS is not an improvement."""
    epochs: NDArray[np.uintc] = np.arange(len(history["val_ess"]), dtype=np.uintc)
    ess: NDArray[np.double] = np.array(object=history["val_ess"], dtype=np.double)
    ess_pct: NDArray[np.double] = 100 * ess / ess[0]

    _ = ax.plot(epochs, ess_pct, color=COLOR_MC, lw=1.5)
    _ = ax.set_xlabel(xlabel="Epoch")
    # Rotated 90 deg, a two-line label's rendered height is set by its
    # *longest line's width*, not by the font size alone -- at the module's
    # default 18pt, "Effective sample size" alone is taller than this short
    # (height-ratio 3) panel, and spills into the x-tick labels below it.
    # Shrinking just this label keeps the panel proportions the brief calls
    # for instead of stealing height from it.
    _ = ax.set_ylabel(ylabel="Effective sample size\n(% of epoch 0)", fontsize=10)
    low: float = min(ess_pct.min(), 100.0)
    high: float = max(ess_pct.max(), 100.0)
    pad: float = 0.05 * (high - low if high > low else 1.0)
    _ = ax.set_ylim(bottom=low - pad, top=high + pad)
    _clip_ticks_to_view(ax)


def _selection_legend(legend_ax: Axes, mmd_ax: Axes) -> None:
    """Draw the MMD panel's legend into its own axes instead of `mmd_ax`.

    `legend_ax` carries no data of its own -- it exists to host the legend at
    a fixed spot in the figure's right column, entirely outside both plotted
    panels, so it is turned off rather than left with empty spines and ticks.
    """
    handles, labels = mmd_ax.get_legend_handles_labels()
    _ = legend_ax.axis("off")
    _ = legend_ax.legend(handles=handles, labels=labels, loc="center", frameon=True)


def plot_selection(
    history: dict[str, list[float]],
    best_epoch: int,
    save_path: Path = Path("plots/selection.pdf"),
) -> None:
    """Two panels answering three separate questions: which epoch was
    selected and does the criterion justify it (top left); does the
    truth-free detector-level criterion track the particle-level one, shown
    as a correlation scatter rather than two overlaid noisy time series
    (bottom right, when truth is available); and did the effective sample
    size collapse while MMD fell (bottom left).

    Detector-level MMD is the criterion; particle-level is the diagnostic.
    The particle curve is absent for a real measurement, which has no truth
    to score against, so it -- and the scatter it feeds -- are optional.

    The legend lives in its own axes in the top right, rather than inside
    or beside the MMD axes, so it neither covers the data nor competes with
    the scatter panel for the same space.
    """
    figure: Figure = Figure(figsize=(9, 6))
    figure.canvas = FigureCanvasPdf(figure)
    has_particle: bool = "val_mmd_particle" in history

    # Neither the outer 1x2 split nor either nested column passes an
    # explicit `wspace`/`hspace` to `add_gridspec` itself -- only to a
    # `SubplotSpec.subgridspec` nested inside a cell. `tight_layout` marks a
    # `GridSpec` "locally modified" (falling back to undersized margins)
    # exactly when spacing is set on the gridspec it inspects directly; a
    # nested subgridspec's own spacing does not trip that check. `_draw_panel`
    # above uses the same trick for the same reason.
    outer: GridSpec = figure.add_gridspec(nrows=1, ncols=2, width_ratios=[7, 4])
    left: GridSpecFromSubplotSpec = outer[0].subgridspec(
        nrows=2, ncols=1, height_ratios=[7, 3], hspace=0.08
    )
    # The first argument of `add_subplot` must be positional only.
    mmd_ax: Axes = figure.add_subplot(left[0])
    ess_ax: Axes = figure.add_subplot(left[1], sharex=mmd_ax)

    right_rows: int = 2 if has_particle else 1
    right: GridSpecFromSubplotSpec = outer[1].subgridspec(
        nrows=right_rows,
        ncols=1,
        height_ratios=[1, 1] if has_particle else [1],
        hspace=0.35,
    )
    # The first argument of `add_subplot` must be positional only.
    legend_ax: Axes = figure.add_subplot(right[0])

    _mmd_panel(mmd_ax, history, best_epoch)
    _ess_panel(ess_ax, history)
    _selection_legend(legend_ax, mmd_ax)
    if has_particle:
        # The first argument of `add_subplot` must be positional only.
        scatter_ax: Axes = figure.add_subplot(right[1])
        _mmd_scatter(scatter_ax, history, best_epoch)

    figure.tight_layout()
    _save_fig(figure, save_path=Path(save_path))
