from __future__ import annotations

from typing import TYPE_CHECKING

from . import evaluate, plotting
from .evaluate import (
    SLICED_PROJECTIONS,
    SLICED_SEED,
    MetricSet,
    apply_to_runs,
    evaluate_run,
    evaluate_runs,
    joint_entry,
    render_metrics,
    warn_if_no_joint,
)
from .plotting import (
    ALPHA_DECONVOLVE,
    ALPHA_FILL,
    ALPHA_IBU,
    ALPHA_OMNIFOLD,
    COLOR_DECONVOLVE,
    COLOR_IBU,
    COLOR_MC,
    COLOR_NATURE,
    COLOR_OMNIFOLD,
    LOSS_YLIM_FRACTION,
    SELECTION_MMD_LINTHRESH,
    SELECTION_SMOOTHING_WINDOW,
    Z_BASELINE,
    Z_DECONVOLVE,
    AxesHist,
    BaselineOverlay,
    ibu_overlay,
    omnifold_overlay,
    plot_detector_level,
    plot_levels,
    plot_losses,
    plot_particle_level,
    plot_selection,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "ALPHA_DECONVOLVE",
    "ALPHA_FILL",
    "ALPHA_IBU",
    "ALPHA_OMNIFOLD",
    "COLOR_DECONVOLVE",
    "COLOR_IBU",
    "COLOR_MC",
    "COLOR_NATURE",
    "COLOR_OMNIFOLD",
    "LOSS_YLIM_FRACTION",
    "SELECTION_MMD_LINTHRESH",
    "SELECTION_SMOOTHING_WINDOW",
    "SLICED_PROJECTIONS",
    "SLICED_SEED",
    "Z_BASELINE",
    "Z_DECONVOLVE",
    "AxesHist",
    "BaselineOverlay",
    "MetricSet",
    "apply_to_runs",
    "evaluate",
    "evaluate_run",
    "evaluate_runs",
    "ibu_overlay",
    "joint_entry",
    "omnifold_overlay",
    "plot_detector_level",
    "plot_levels",
    "plot_losses",
    "plot_particle_level",
    "plot_selection",
    "plotting",
    "render_metrics",
    "warn_if_no_joint",
)
