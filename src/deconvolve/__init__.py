# pyrefly: ignore-errors[unused-call-result]
"""Deconvolve: Reweighting Adversarial Networks (RANs) for unfolding.

A library for training and evaluating reweighting adversarial networks.

Importing anything under `deconvolve` first pins the Keras 3 backend to JAX
and disables JAX's 64-bit mode. Both settings are read once, when
`jax`/`keras` are first imported, so they must be in place before any
submodule imports either. `setdefault` throughout, so that the environment
can be explicitly overridden.

The package is float32 end to end. The pin is `EVENT_DTYPE` in
`deconvolve.coretypes.constants`, with its annotation twin `EventArray` in
`deconvolve.coretypes.types`; `JAX_ENABLE_X64=0` and the `dtype=` arguments in
`deconvolve.training.models` follow from it.

Import the submodule needed (`from deconvolve.workflows import run`); the CLI
re-exports below are the sole exception.

Only `cli.py` sits beside `__init__.py` and `__main__.py`; everything else
lives in a subpackage: `config` (layered CLI configuration: discovery and
merge, the command spec, `config show`), and by pipeline stage `data`,
`training` (models, the fused loop, MMD), `evaluation` (metrics and plots),
`baselines`, `uncertainty`, `workflows` (the orchestration behind
`deconvolve train` and `deconvolve leakage-check`), `reporting` (the LaTeX
dossier and its template) and `instrumentation` (timing and logging setup).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

os.environ.setdefault(key="KERAS_BACKEND", value="jax")
os.environ.setdefault(key="JAX_ENABLE_X64", value="0")

from . import cli
from .cli import (
    app,
    configure,
    evaluate_command,
    ibu_command,
    leakage_check_command,
    train_command,
)

__all__: Final[Sequence[str]] = (
    "app",
    "cli",
    "configure",
    "evaluate_command",
    "ibu_command",
    "leakage_check_command",
    "train_command",
)
