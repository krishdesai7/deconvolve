"""Comparison baselines for RAN: IBU and OmniFold.

```python
from deconvolve.baselines import ibu_evaluate_runs, omnifold_evaluate_runs
```

IBU runs in this process. OmniFold cannot: it needs TensorFlow, which has no
wheels for this project's Python floor and could not share a Keras backend
with JAX even if it did. `omnifold` is the host half and `_omnifold_worker.py`
-- a PEP 723 script, never imported -- is the other, with one `.npz` file
between them. `notes/agent/omnifold.md` carries the whole argument, including
the `module load cudatoolkit/12.9` that Perlmutter needs and the silent CPU
fallback it prevents.

Both are scored by the same code: `_shared` reads the run's config, rebuilds
its populations, and computes the same metrics RAN is scored by, so a
comparison is a comparison of unfolding methods and nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import ibu, omnifold
from .ibu import VariableUnfolding, unfold_variable
from .ibu import evaluate_runs as ibu_evaluate_runs
from .ibu import evaluate_single as ibu_evaluate_single
from .omnifold import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_N_EPOCHS,
    DEFAULT_N_ITERATIONS,
    WORKER_TIMEOUT_SECONDS,
    worker_script,
)
from .omnifold import evaluate_runs as omnifold_evaluate_runs
from .omnifold import evaluate_single as omnifold_evaluate_single
from .omnifold import unfold as omnifold_unfold

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

__all__: Final[Sequence[str]] = (
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_N_EPOCHS",
    "DEFAULT_N_ITERATIONS",
    "WORKER_TIMEOUT_SECONDS",
    "VariableUnfolding",
    "ibu",
    "ibu_evaluate_runs",
    "ibu_evaluate_single",
    "omnifold",
    "omnifold_evaluate_runs",
    "omnifold_evaluate_single",
    "omnifold_unfold",
    "unfold_variable",
    "worker_script",
)
