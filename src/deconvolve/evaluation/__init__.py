from __future__ import annotations

from typing import TYPE_CHECKING

from . import evaluate, plotting
from .evaluate import (
    MetricSet,
    apply_to_runs,
    evaluate_run,
    evaluate_runs,
    render_metrics,
)
from .plotting import (
    ALPHA_FILL,
    ALPHA_IBU,
    ALPHA_OMNIFOLD,
    ALPHA_RAN,
    COLOR_IBU,
    COLOR_MC,
    COLOR_NATURE,
    COLOR_OMNIFOLD,
    COLOR_RAN,
    LOSS_YLIM_FRACTION,
    SELECTION_MMD_LINTHRESH,
    SELECTION_SMOOTHING_WINDOW,
    Z_BASELINE,
    Z_RAN,
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
    "ALPHA_FILL",
    "ALPHA_IBU",
    "ALPHA_OMNIFOLD",
    "ALPHA_RAN",
    "COLOR_IBU",
    "COLOR_MC",
    "COLOR_NATURE",
    "COLOR_OMNIFOLD",
    "COLOR_RAN",
    "LOSS_YLIM_FRACTION",
    "SELECTION_MMD_LINTHRESH",
    "SELECTION_SMOOTHING_WINDOW",
    "Z_BASELINE",
    "Z_RAN",
    "AxesHist",
    "BaselineOverlay",
    "MetricSet",
    "apply_to_runs",
    "evaluate",
    "evaluate_run",
    "evaluate_runs",
    "ibu_overlay",
    "omnifold_overlay",
    "plot_detector_level",
    "plot_levels",
    "plot_losses",
    "plot_particle_level",
    "plot_selection",
    "plotting",
    "render_metrics",
)
