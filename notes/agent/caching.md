# Caching

Everything RAN can regenerate lives under one root, `.cache/` by default:
generated Gaussian datasets, the per-variable jet `.npz` files pulled from
Zenodo, and the XLA compilation cache. **`DECONVOLVE_CACHE_DIR` moves the whole tree**,
which is what a cluster needs — on Perlmutter `$HOME` is small, quota'd and
shared across nodes:

```bash
export DECONVOLVE_CACHE_DIR="$SCRATCH/ran-cache"
```

It is deliberately its own variable rather than a read of `XDG_CACHE_HOME`. That
one is already set (or defaults to `~/.cache`) on most Linux systems, so
deriving from it would silently move every existing checkout's cache and orphan
the jet data already on disk. `~` is expanded, and an empty value falls back to
the default rather than meaning the current directory — a SLURM `--export` that
forwards an unset variable delivers `""`, not absence.

`CACHE_ENV_VAR` and `CACHE_DIR` are resolved once, at import of
`coretypes/constants.py`, because the `cache_dir=` defaults throughout
`deconvolve.data` bind to `CACHE_DIR` at import either way.

This same fact is why `DECONVOLVE_CACHE_DIR` did not join the layered config stack
added in `notes/agent/configuration.md`: that stack is resolved lazily, once
per invocation, inside the `configure()` Typer callback, specifically so a
setting is not fixed before the CLI has even parsed its arguments. `CACHE_DIR`
is the case the callback's docstring points to as the failure mode being
avoided — it is read at `coretypes/constants.py:35` and then bound into default
arguments at `data/jets.py:62`, `data/download.py:272` and
`data/datasets.py:153`, all at **import** time, well before any callback runs.
`DECONVOLVE_TIMING` has the identical shape: the recorder is constructed at import of
`instrumentation/timing.py:122`. Folding either into the stack would mean
converting these module-level constants into lazy accessors and unbinding
three default arguments in code that decides where datasets are cached — a
change with its own reproducibility risk, not a documentation-sized one. Both
therefore remain environment-only, and `deconvolve config show` lists them in a
separate "Environment-only" section rather than pretending they participate in
precedence they don't.

## Dev-tool caches

`.cache/` also holds the dev-tool caches — `.cache/pytest`, `.cache/ruff`,
`.cache/complexipy` and `.cache/pycache`. These are a separate concern from
the regenerable-data cache above: they exist to declutter the root, not to be
relocated off `$HOME` on a cluster, so unlike `CACHE_DIR` they are **not**
chained to `DECONVOLVE_CACHE_DIR` — moving a few kilobytes of lint/test cache buys
nothing on a quota, and chaining it would make `DECONVOLVE_CACHE_DIR` mean two
different things.

Each tool gets there by whatever mechanism it supports:

- `pytest` and `complexipy` read `cache_dir`/`cache-dir` from `pyproject.toml`
  (`[tool.pytest.ini_options]`, `[tool.complexipy]`), because both support a
  config key but neither reads an environment variable for it.
- `ruff` and CPython's own bytecode cache take theirs from `RUFF_CACHE_DIR` and
  `PYTHONPYCACHEPREFIX`, set in `.env`, because a `pyproject.toml` key is not
  the only thing that works and the project's `.env` is already the place
  environment-only settings live (see `DECONVOLVE_CACHE_DIR` above, and
  `JAX_PLATFORMS`). `.env` must actually be sourced into the shell for these
  to take effect — it is not read by `uv run` or any script here.

## Compilation cache

`engine.train()` calls `_use_compilation_cache()`, which points XLA's persistent cache
at `CACHE_DIR / "jax"`. This is worth doing because **compile is the largest
single term in a short run**: `benchmarks/boundary.py` on an A100 measures 4.60s
of XLA against 0.034s per epoch, so a 100-epoch run spends half its wall clock
compiling. The cache keys on lowered HLO rather than Python identity — the fresh
`jax.jit(lambda ...)` in `_run` hits it regardless — and it lives on disk, so an
ensemble of N interpreters compiling one architecture pays the cost once instead
of N times. Measured locally, 1.41s cold → 0.36s warm across processes.

Two settings, not one. JAX's default `jax_persistent_cache_min_compile_time_secs`
of 1.0s leaves RAN's cache **entirely empty**: a run compiles a few dozen
executables totalling ~4.6s and no single one of them clears a second.
`_use_compilation_cache` drops it to zero.

Anything the caller configured wins: `JAX_COMPILATION_CACHE_DIR` and
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` still override, and an unwritable
directory costs a warning from JAX rather than the run.
