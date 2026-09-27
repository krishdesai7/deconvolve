# Deconvolve

Deconvolve is an adversarial machine learning package that uses adversarial learning to unfold (deconvolve) reco-level data to nominal truth. The package lives under
`src/deconvolve/`; see [notes/agent/structure.md](notes/agent/structure.md) for the full tree.

## Core Algorithm

- `(z_gen, x_sim)` are naturally paired MC events. `(z_true, x_data)` are naturally paired data events.
- **Generator** `g(z)` takes nominal-level events as input and produces per-event weights `w = g(z)`. For data events (y=1), weights are fixed to 1. Weights are normalized to preserve total counts per class.
- These weights are applied to reco-level _distributions_ (not the events themselves). `x_data` is reweighted with w=1 (unchanged), `x_sim` is reweighted with `w = g(z_gen)`.
- **Discriminator** `d(x)` operates at reco level to distinguish these two reweighted distributions.
- **Loss**: weighted BCE — `w_i * y_i * log(d(x_i)) + (1 - y_i) * w_i * log(1 - d(x_i))`.
- **Training**: min-max game. `d` minimizes BCE (correctly distinguish data from reweighted sim). `g` maximizes BCE (generate weights that confound `d`).

## Critical Constraint

**No network should ever have access to `z_true`.** This is the unfolded truth that we do not know in principle. `g` only sees `z` (nominal-level features), never the true particle-level values from data. The event types in [notes/agent/data-model.md](notes/agent/data-model.md) enforce this in the type system, not just by convention.

## Tooling Preferences

- Prefer `fd` over `find`, `rg` over `grep`, and `fzf` for fuzzy finding. `find`/`grep` are fine as fallbacks.
- When working with Python, you may invoke the relevant `/astral:<skill>` for uv, ty, and ruff.

## Reference Docs

Everything below is detail this file used to carry inline. Read the linked file when a task touches that area; don't load all of them speculatively.

| Doc                                                              | Covers                                                                   |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [notes/agent/data-model.md](notes/agent/data-model.md)           | `Populations`/`ZXY`/device-resident split types; jet column ordering     |
| [notes/agent/structure.md](notes/agent/structure.md)             | Full source tree, `runs/` and `.cache/` layout                           |
| [notes/agent/cli-and-running.md](notes/agent/cli-and-running.md) | CLI reference, `just` recipes, test markers, `scripts/submit.zsh`        |
| [notes/agent/training-loop.md](notes/agent/training-loop.md)     | `engine.py`'s fused `lax.scan` program and host-side checkpoint selection |
| [notes/agent/seeding.md](notes/agent/seeding.md)                 | `data_seed` vs `seed`, batch order, leakage-check                        |
| [notes/agent/precision.md](notes/agent/precision.md)             | The float32 pin, where it doesn't reach, and reproducibility gotchas     |
| [notes/agent/tech-stack.md](notes/agent/tech-stack.md)           | Dependencies, Keras/JAX backend pin, Gaussian config YAML format         |
| [notes/agent/caching.md](notes/agent/caching.md)                 | `DECONVOLVE_CACHE_DIR`, dev-tool caches, the XLA compilation cache              |
| [notes/agent/reporting.md](notes/agent/reporting.md)             | `deconvolve report`, the LaTeX template, figure pagination                      |
| [notes/agent/timing.md](notes/agent/timing.md)                   | `DECONVOLVE_TIMING=1`, `timings.json` phase structure                           |
| [notes/agent/omnifold.md](notes/agent/omnifold.md)               | The OmniFold baseline's subprocess boundary and SLURM job                |
| [notes/agent/uncertainty.md](notes/agent/uncertainty.md)         | The bootstrap x seed variance design                                     |
| [notes/agent/releasing.md](notes/agent/releasing.md)             | The Actions release workflow and versioning policy                       |
| [notes/agent/configuration.md](notes/agent/configuration.md)     | The five config layers, discovery order, `deconvolve config show`, the design freeze |

Module-level detail also lives in `README.md` files next to the code: `src/deconvolve/README.md`, `src/deconvolve/coretypes/README.md`, `src/deconvolve/data/README.md`, `src/deconvolve/baselines/README.md`, `src/deconvolve/uncertainty/README.md`, `benchmarks/README.md`, `scripts/README.md`.

### Where to look from a source file

Source files carry no pointers into `notes/`; this is the map. Line numbers drift, so the symbol is the durable half.

| Source                                                            | Notes                                                    |
| ----------------------------------------------------------------- | -------------------------------------------------------- |
| `coretypes/constants.py:24` `EVENT_DTYPE`                          | [precision.md](notes/agent/precision.md)                 |
| `coretypes/constants.py:33` `CACHE_ENV_VAR`, `CACHE_DIR`           | [caching.md](notes/agent/caching.md)                     |
| `coretypes/constants.py:71` `SUBSTRUCTURE_VARIABLES`, `:179` `JET_DISPLAY_ORDER` | [data-model.md](notes/agent/data-model.md) |
| `evaluation/evaluate.py:103` recorded `variables`                 | [data-model.md](notes/agent/data-model.md)               |
| `evaluation/evaluate.py:212` `_counts` (scatter-add)              | [precision.md](notes/agent/precision.md)                 |
| `instrumentation/timing.py:12` `compile_cache_warm`               | [timing.md](notes/agent/timing.md), [caching.md](notes/agent/caching.md) |
| `instrumentation/timing.py:356` `write`, `:252` `_ordered`        | [timing.md](notes/agent/timing.md)                       |
| `baselines/omnifold.py:1` module docstring                        | [omnifold.md](notes/agent/omnifold.md)                   |
| `baselines/omnifold.py:229` `_record_iteration_timings`, `:334`   | [timing.md](notes/agent/timing.md)                       |
| `config/show.py:1` module docstring                               | [configuration.md](notes/agent/configuration.md)         |

## Maintaining This File

Keep this file to roughly 100 lines. It is the index, not the encyclopedia: new subsystem detail, war stories and rationale belong in a file under `notes/agent/` (add a new one if none fits) or a module's own `README.md`, linked from the Reference Docs table above rather than inlined here. If a change needs more than a couple of sentences here, it almost certainly belongs in one of those instead.

`src/deconvolve/coretypes/`, `src/deconvolve/data/`, `src/deconvolve/baselines/` and
`src/deconvolve/uncertainty/` each carry their own `README.md`.

`docs/` is the published mkdocs site and holds only reader-facing pages. Everything internal lives in `notes/`: `notes/agent/` (the files above), `notes/superpowers/{plans,specs}/` (write new plans and specs there, not under `docs/`) and `notes/working-notes/`.

## Running

`uv run deconvolve <subcommand>` is the entry point (`train`, `evaluate`, `report`,
`leakage-check`, `baseline {ibu,omnifold}`, `uncertainty {run,collect,freeze}`,
`config show`); `--log-level` is global and goes before the subcommand. Dev
recipes go through `just` (`just validate`, `just test-fast`). Everything else
(flags, SLURM scripts, test markers) is in
[notes/agent/cli-and-running.md](notes/agent/cli-and-running.md).
