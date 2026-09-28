# Tech Stack

- Python >= 3.12, managed with `uv` (no pip). 3.12 is the floor because
  `evaluation/plotting.py`, `coretypes/types.py` and `instrumentation/timing.py`
  use PEP 695 (`type X =`, `def block[T]`), which 3.11 cannot parse. The floor
  was `>=3.14` until the pre-release compatibility pass; what held it there was
  PEP 758's unparenthesized `except OSError, ValueError:` in
  `reporting/report.py` and `instrumentation/timing.py` (now parenthesized) and
  one `hashlib.sha256(data=...)` keyword in `data/datasets.py` (now positional,
  hashing identical bytes, so existing caches stay valid). **Do not reintroduce
  either**: with ruff's target inferred from `requires-python` the formatter no
  longer canonicalises to PEP 758, but hand-written 3.13+ syntax would silently
  raise the floor. Development still happens on 3.14 (`.python-version`); the
  suite is run on 3.12, 3.13 and 3.14
- Keras 3 on the **JAX** backend for training; `jax[cuda13]` on x86_64 Linux
- Typer for the CLI, Rich for logging and metrics tables
- Matplotlib for publication-quality plots; `pdflatex` (TeX Live, with
  siunitx, booktabs and pdflscape) for `deconvolve report`, and only for that
- scipy for evaluation metrics (Wasserstein distance, Jensen-Shannon divergence)
- jaxtyping + beartype for shape/dtype checking on the training loop's array seams
- ruff (lint + format), pyrefly (types, `--min-severity info`) + `uv check --locked`, complexipy (max 10)
- TensorFlow is not a dependency, direct or transitive. JAX is the only array
  backend in the build, so nothing here has to negotiate for the GPU.

## Backend

JAX is the only backend in the build; there is no second framework competing
for the Keras backend slot or for the GPU.

`src/deconvolve/__init__.py` sets `KERAS_BACKEND=jax` and `JAX_ENABLE_X64=0`. Keras 3
defaults to TensorFlow when that variable is unset, and TensorFlow is not
installed, so the pin makes `import keras` work at all. It must land before
the first keras import, which is why it lives in the package `__init__`;
`src/deconvolve/training/engine.py` keeps a cheap guard that raises a readable error if someone
sets `KERAS_BACKEND` to something else by hand.

## Gaussian Config Format

YAML files in `params/` use keys: `mu_gen`, `mu_true`, `sigma_gen`,
`sigma_true`, `sigma_detector`. Sigma values are promoted via
`sigma_to_covariance`: scalar → σ²I, vector → diag(σ²), matrix → used as-is.
