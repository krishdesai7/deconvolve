"""Redraw a finished run's figures: `deconvolve plot RUN_DIR`.

The figures are drawn once at the end of `deconvolve train`, which is before
either baseline can have run. The baselines write their weights
(`ibu_weights.npz`, `omnifold_weights.npz`) next to the run's other artifacts
and stop there, so this is the step that puts their overlays on the plots --
and the way to draw a run trained with `--no-plots` at all.

Nothing is trained and nothing is scored. Everything comes from the run
directory: `config.json` rebuilds the same dataset (same `data_seed`, so the
same test split), and the saved generator and history are reloaded.

```shell
deconvolve plot runs/2026-10-04T022859Z   # one run
deconvolve plot runs                      # every run under runs/
```
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ..baselines._shared import parse_run_config
from ..coretypes import DatasetName
from ..data import gaussian_config_from_run_config
from ..evaluation import apply_to_runs
from .train import (
    _display_variables,
    _draw_figures,
    _load_artifacts,
    _prepare_gaussian,
    _prepare_jets,
)

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path

    from ..coretypes import DatasetSplits, RunConfig, VarInfo

logger: Logger = logging.getLogger(name=__name__)


def plot_run(run_dir: Path, /) -> None:
    """Redraw one run's figures, with whichever baselines have run by now."""
    config: RunConfig = parse_run_config(
        raw=json.loads(s=(run_dir / "config.json").read_text())
    )
    var_info: list[VarInfo] | None = None
    if config.dataset == DatasetName.jets:
        splits: DatasetSplits
        splits, _, var_info = _prepare_jets(
            config.n_samples, config.batch_size, config.variable_names, config.data_seed
        )
    else:
        splits, _, _ = _prepare_gaussian(
            None,
            gaussian_config_from_run_config(
                config.source["gaussian_params"], config.dim
            ),
            config.batch_size,
            config.n_samples,
            config.data_seed,
        )

    g, history = _load_artifacts(run_dir)
    _draw_figures(
        run_dir,
        splits,
        g,
        history,
        config.dim,
        var_info,
        # Absent on a run saved before selection was recorded: the same
        # fallback `train --load-run` uses.
        config.source.get("best_epoch", -1),
        _display_variables(config.dataset, config.variable_names),
        plots=True,
    )


def plot_runs(run_dir: Path, /) -> None:
    """Redraw a single run's figures, or those of every run inside a directory."""
    apply_to_runs(run_dir, evaluate_one=plot_run, description="plot", log=logger)
