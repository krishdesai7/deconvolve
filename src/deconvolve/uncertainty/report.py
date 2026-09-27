"""Turn a finished design into the two numbers a paper needs.

The **summary table** decomposes the unfolded mean of each observable into its
three sources and shows what the naive quadrature sum of two one-dimensional
sweeps would have claimed instead. The mean is used rather than a bin content
because it depends on no binning choice, and a decomposition whose answer
moves with the binning invites exactly the objection it exists to settle.

The **covariance** is the other half, and the one the speed argument pays for.
Unbinned unfolding weights are routinely propagated as if their bin-to-bin
correlations were zero. They are not, and measuring them takes a hundred
retrainings --- which at OmniFold's cost is not an analysis anyone runs, and
at RAN's is twenty minutes on a node. `variance.npz` carries the full `K x K`
matrix for every observable.

One caveat is stated rather than buried, because a referee will raise it:
RAN's weights preserve the total count by construction, so a spectrum's bins
sum to a fixed number and its covariance is singular with rank `K - 1`: every
row sums to zero. For bins of equal variance that fixes the *average*
correlation of a bin with the others at `-1 / (K - 1)`, the multinomial value
for equal-occupancy bins. It fixes nothing else --- individual entries are
free, and can lie on either side of it --- so it is a reference, not a floor.
It is written into the output next to the measurement. Structure around that
average --- neighbouring bins correlating more strongly than distant ones,
say --- is the part normalization cannot explain.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, cast

import matplotlib as mpl
import numpy as np
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.figure import Figure
from rich.console import Console
from rich.table import Table

from ..coretypes import DatasetName
from ..data import gaussian_config_from_run_config
from .design import DesignSpec, base_populations, load_cells
from .variance import (
    binned_spectra,
    component_covariances,
    correlation,
    decompose,
    evaluation_covariance,
    evaluation_variance,
    quantile_edges,
    weighted_means,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from logging import Logger
    from pathlib import Path
    from typing import Any

    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage
    from numpy.typing import NDArray

    from ..coretypes import EventArray, Populations
    from .design import Design, EvaluationSet
    from .variance import Covariances, VarianceComponents

logger: Logger = logging.getLogger(name=__name__)
mpl.use(backend="Agg")

# The residual is the interaction plus everything else that varies from cell to
# cell and with neither axis: the split, the batch order and the selection
# subsamples (`design.split_seed`), and any hardware non-determinism.
SOURCES: tuple[str, str, str] = (
    "data (bootstrap)",
    "initialization",
    "interaction, split and order",
)


def multinomial_off_diagonal(n_bins: int, /) -> float:
    """The average off-diagonal correlation a fixed total implies.

    A fixed total makes every covariance row sum to zero, which for bins of
    equal variance puts the average correlation of a bin with the others at
    `-1 / (K - 1)`; a multinomial with `p_i = 1 / K` has exactly that value
    in every entry. The constraint fixes only the average, so a measured
    matrix is read against this as a reference: its structure around the
    average is what normalization does not explain.
    """
    if n_bins < 2:
        raise ValueError(f"need at least 2 bins, got {n_bins}")
    return -1.0 / (n_bins - 1)


def _warn_if_rank_deficient(n_datasets: int, n_bins: int, /) -> None:
    """A `K x K` covariance from `B` replicates has rank at most `B - 1`.

    Below that the matrix is singular and every correlation saturates at
    +-1 --- a heatmap of solid red and blue that looks like a very strong
    result and is an artifact of the sample size. Said out loud, because the
    figure gives no other sign of it.
    """
    if n_datasets <= n_bins:
        logger.warning(
            "%d bootstrap datasets against %d bins: the covariance has rank at "
            "most %d and the correlations are saturated by construction. Run "
            "B well above K (B=100 for K=20) before reading the off-diagonals.",
            n_datasets,
            n_bins,
            n_datasets - 1,
        )


def _evaluation_events(design: Design, /) -> EventArray:
    """Rebuild the common evaluation set the cells were read on.

    Regenerated rather than stored: it is identical across every cell, so
    writing it into each one would cost hundreds of megabytes for a copy of
    something two recorded seeds already determine --- and having exactly one
    copy makes it impossible for the cells to disagree about it.
    """
    from .design import reserve_evaluation_set

    meta: dict[str, Any] = design.meta
    raw: dict[str, Any] | None = meta.get("gaussian_params")
    pops: Populations = base_populations(
        DatasetName(value=meta["dataset"]),
        n_samples=meta["n_samples"],
        batch_size=meta["batch_size"],
        data_seed=meta["data_seed"],
        variables=tuple(meta["variables"]),
        params=(
            gaussian_config_from_run_config(params=raw, dim=meta["dim"])
            if raw is not None
            else None
        ),
    )[0]
    evaluation: EvaluationSet = reserve_evaluation_set(
        pops, n_eval=meta["n_eval"], seed=meta["data_seed"]
    )
    if evaluation.z.shape[0] != design.weights.shape[2]:
        raise ValueError(
            f"regenerated {evaluation.z.shape[0]} evaluation events but the "
            f"cells carry {design.weights.shape[2]} weights each; the recorded "
            "metadata does not reproduce the design"
        )
    return evaluation.z


def _variable_names(design: Design, /) -> list[str]:
    meta: dict[str, Any] = design.meta
    if meta["dataset"] == DatasetName.jets.value:
        return list(meta["variables"])
    return [f"dim_{i}" for i in range(meta["dim"])]


def _summary_row(
    components: VarianceComponents, evaluation: float, /
) -> dict[str, float]:
    """One observable's decomposition, as standard deviations and shares.

    `var_total` is the run-to-run total the grid measures. `var_evaluation` is
    the evaluation set's own sampling variance, which cancels from every
    component and is therefore outside that total; a quoted uncertainty adds
    it back.
    """
    total: float = float(components.total)
    return {
        "sd_total": float(np.sqrt(max(total, 0.0))),
        "sd_data": float(np.sqrt(max(float(components.data), 0.0))),
        "sd_init": float(np.sqrt(max(float(components.init), 0.0))),
        "sd_interaction": float(np.sqrt(max(float(components.interaction), 0.0))),
        "var_total": total,
        "var_data": float(components.data),
        "var_init": float(components.init),
        "var_interaction": float(components.interaction),
        "sd_naive_quadrature": float(
            np.sqrt(max(float(components.naive_quadrature), 0.0))
        ),
        "var_evaluation": evaluation,
        "sd_evaluation": float(np.sqrt(max(evaluation, 0.0))),
    }


def _render(summary: dict[str, dict[str, float]], /, console: Console | None) -> None:
    active: Console = console or Console()
    table = Table(title="Variance of the unfolded mean, by source")
    table.add_column(header="Variable")
    table.add_column(header="Source")
    table.add_column(header="SD", justify="right")
    table.add_column(header="% of total variance", justify="right")
    for name, row in summary.items():
        total: float = row["var_total"]
        for label, key in zip(SOURCES, ("data", "init", "interaction"), strict=True):
            share: float = (
                100.0 * row[f"var_{key}"] / total if total > 0 else float("nan")
            )
            table.add_row(name, label, f"{row[f'sd_{key}']:.5f}", f"{share:+.1f}%")
        table.add_row(name, "[bold]total[/bold]", f"{row['sd_total']:.5f}", "100.0%")
        inflation: float = (
            100.0 * (row["sd_naive_quadrature"] / row["sd_total"] - 1.0)
            if row["sd_total"] > 0
            else float("nan")
        )
        table.add_row(
            "",
            "naive quadrature",
            f"{row['sd_naive_quadrature']:.5f}",
            f"(SD {inflation:+.1f}%)",
        )
        table.add_row(
            "",
            "evaluation sample",
            f"{row['sd_evaluation']:.5f}",
            "(add to total)",
        )
        table.add_section()
    active.print(table)


def _plot_correlations(
    path: Path,
    names: list[str],
    matrices: dict[str, NDArray[np.double]],
    null: dict[str, float],
    /,
) -> None:
    """One heatmap per observable of the bootstrap bin-to-bin correlation."""
    figure = Figure(figsize=(4.0 * min(len(names), 3), 3.6 * ((len(names) + 2) // 3)))
    figure.canvas = FigureCanvasPdf(figure)
    axes: NDArray[Any] = figure.subplots(
        nrows=(len(names) + 2) // 3, ncols=min(len(names), 3), squeeze=False
    )
    flat_axes: Sequence[Axes] = cast(typ="Sequence[Axes]", val=list(axes.ravel()))
    for ax, name in zip(flat_axes, names, strict=False):
        image: AxesImage = ax.imshow(
            X=matrices[name], cmap="RdBu_r", vmin=-1.0, vmax=1.0, origin="lower"
        )
        _ = ax.set_title(label=f"{name}  (closure mean {null[name]:+.3f})", fontsize=9)
        _ = ax.set_xlabel(xlabel="bin")
        _ = ax.set_ylabel(ylabel="bin")
        _ = figure.colorbar(mappable=image, ax=ax, fraction=0.046)
    for ax in flat_axes[len(names) :]:
        ax.set_axis_off()
    _ = figure.suptitle(t="Bootstrap correlation of the unfolded spectrum")
    figure.tight_layout()
    figure.savefig(fname=path)


def collect(
    design_dir: Path,
    spec: DesignSpec,
    /,
    *,
    n_bins: int = 20,
    console: Console | None = None,
) -> dict[str, dict[str, float]]:
    """Decompose a finished design and write its table, npz and figure."""
    design: Design = load_cells(design_dir, spec)
    _warn_if_rank_deficient(spec.n_datasets, n_bins)
    z_eval: EventArray = _evaluation_events(design)
    names: list[str] = _variable_names(design)

    summary: dict[str, dict[str, float]] = {}
    saved: dict[str, NDArray[np.double]] = {}
    correlations: dict[str, NDArray[np.double]] = {}
    null: dict[str, float] = {}

    for i, name in enumerate(iterable=names):
        column: EventArray = z_eval[:, i]
        # Averaged over cells: the evaluation variance depends on each run's
        # weights, and the grid samples the distribution of those runs.
        summary[name] = _summary_row(
            decompose(weighted_means(column, design.weights)),
            float(evaluation_variance(column, design.weights).mean()),
        )

        edges: NDArray[np.double] = quantile_edges(column, n_bins=n_bins)
        spectra: NDArray[np.double] = binned_spectra(
            column, design.weights, edges=edges
        )
        covariances: Covariances = component_covariances(spectra)
        correlations[name] = correlation(covariances.data)
        null[name] = multinomial_off_diagonal(edges.size - 1)
        saved |= {
            f"{name}_edges": edges,
            f"{name}_spectrum": spectra.mean(axis=(0, 1)),
            f"{name}_cov_data": covariances.data,
            f"{name}_cov_init": covariances.init,
            f"{name}_cov_interaction": covariances.interaction,
            f"{name}_cov_evaluation": evaluation_covariance(
                column, design.weights, edges=edges
            ).mean(axis=(0, 1)),
            f"{name}_corr_data": correlations[name],
        }

    # See deconvolve.baselines.ibu: unpacking a str-keyed dict into savez means a key
    # could in principle be "allow_pickle", which is declared bool.
    np.savez(file=design_dir / "variance.npz", **saved)  # pyrefly: ignore[bad-argument-type]  # ty:ignore[invalid-argument-type]
    _ = (design_dir / "variance.json").write_text(
        data=json.dumps(
            obj={
                "design": spec._asdict(),
                "n_eval": design.weights.shape[2],
                "n_bins": n_bins,
                "multinomial_off_diagonal": null,
                "summary": summary,
            },
            indent=2,
        )
    )
    _plot_correlations(design_dir / "correlation.pdf", names, correlations, null)
    _render(summary, console)
    logger.info(
        "Wrote %s, %s and %s",
        design_dir / "variance.npz",
        design_dir / "variance.json",
        design_dir / "correlation.pdf",
    )
    return summary
