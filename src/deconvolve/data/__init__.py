"""The dataset, from its sources to the accelerator.

The same events are described in three forms along the way. Sources
(`jets`, and the Gaussian draw in `datasets`) produce
`deconvolve.coretypes.Populations`, the physics form; `Populations.interleave`
turns it into a `ZXY`, the labelled transport form that `datasets` shuffles
and splits; and `device` moves those splits onto the accelerator as the
training form. `config` parses Gaussian dataset configs, and `download`
fetches and caches the jet data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import config, datasets, device, download, jets
from .config import (
    gaussian_config_from_run_config,
    parse_gaussian_config,
    sigma_to_covariance,
)
from .datasets import ArrayDataset, DeconvolveDataset
from .device import (
    DEFAULT_EVAL_BATCH_SIZE,
    DeviceSplits,
    EvalSplit,
    TrainSplit,
    gather,
    grouping,
    train_indices,
)
from .download import PID_CHARGE, download_jet_data
from .jets import load_jet_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "DEFAULT_EVAL_BATCH_SIZE",
    "PID_CHARGE",
    "ArrayDataset",
    "DeconvolveDataset",
    "DeviceSplits",
    "EvalSplit",
    "TrainSplit",
    "config",
    "datasets",
    "device",
    "download",
    "download_jet_data",
    "gather",
    "gaussian_config_from_run_config",
    "grouping",
    "jets",
    "load_jet_dataset",
    "parse_gaussian_config",
    "sigma_to_covariance",
    "train_indices",
)
