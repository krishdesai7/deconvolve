# Project Structure

The package lives under `src/` and is importable as `ran`.

```text
src/deconvolve/                      Python package
├── __init__.py               Pins KERAS_BACKEND=jax, JAX_ENABLE_X64=0 (see tech-stack.md)
├── __main__.py                Fallback entry point (python -m ran)
├── cli.py                    Unified Typer command tree; `ran` script targets cli:app
├── py.typed                  PEP 561 marker
├── config/                    Layered CLI configuration (see configuration.md)
│   ├── layers.py               Discovery and merge of TOML layers; stdlib-only leaf
│   ├── spec.py                 Command tree introspected from the live Typer app
│   └── show.py                 `deconvolve config show` rendering (Rich)
├── coretypes/
│   ├── events.py              Split, Events, ZXY, Populations, DatasetSplits
│   ├── configs.py             GaussianConfig, RunConfig, REQUIRED_KEYS
│   ├── results.py             UnfoldingPopulations, VariableOutcome, IBUResult
│   ├── constants.py           Zenodo record, cache layout, JET_OBS
│   ├── enums.py                LogLevel, DatasetName (CLI choice enums)
│   └── types.py                TypedDicts and array aliases (annotation-space only)
├── data/
│   ├── config.py               YAML config parsing, sigma promotion, gaussian_config_from_run_config
│   ├── datasets.py             ArrayDataset (host container), DatasetSplits, DeconvolveDataset, caching
│   ├── jets.py                 Jet substructure loading, standardization (JET_OBS, load_jet_dataset)
│   ├── device.py                Device-resident training form (TrainSplit/EvalSplit, batch order)
│   └── download.py              One-time Zenodo download
├── baselines/
│   ├── _shared.py              Run config + populations a baseline needs, minus the unfolder
│   ├── ibu.py                   IBU (Iterative Bayesian Unfolding) baseline
│   ├── omnifold.py              OmniFold baseline, host half (see omnifold.md)
│   └── _omnifold_worker.py     PEP 723 script; 3.13 + TensorFlow, never imported
├── uncertainty/
│   ├── design.py                Bootstrap x seed grid: resampling, one cell, loading
│   ├── variance.py              Two-way ANOVA components, covariances, quantile binning
│   └── report.py                Decomposition table, variance.npz, correlation.pdf
├── training/
│   ├── models.py               Generator and discriminator architectures
│   ├── engine.py               Fused JAX training program (owns TrainResult/TrainState/RunCarry)
│   └── mmd.py                  Weighted MMD, the checkpoint-selection signal
├── evaluation/
│   ├── evaluate.py             Post-hoc distance metrics (Wasserstein, JS, triangular discriminator)
│   └── plotting.py             Detector-level, particle-level, and loss curve plots
├── workflows/
│   ├── train.py                Training and reload workflow behind `deconvolve train`
│   └── leakage.py              Data-poisoning leakage check behind `deconvolve leakage-check`
├── reporting/
│   ├── report.py               PDF dossier behind `deconvolve report` (see reporting.md)
│   └── templates/
│       └── report.tex          LaTeX skeleton `report.py` fills in; see reporting.md
└── instrumentation/
    ├── timing.py               Optional per-phase wall clock (see timing.md)
    └── logging_config.py       Rich structured application logging

params/                        Gaussian config YAML files
├── 1d_default.yaml
├── 2d_correlated.yaml
├── 4d_correlated.yaml
└── 6d_correlated.yaml

scripts/
├── submit.zsh                  SLURM submission script
├── submit_omnifold.zsh          OmniFold against an existing run dir (see omnifold.md)
├── submit_hparam.zsh            Packed hyperparameter arm sweep (paired on seed)
├── submit_precision.zsh         float32 vs float64 paired ensemble
└── submit_uncertainty.zsh       Packed bootstrap x seed grid (see uncertainty.md)

tests/                         pytest tests (601 cases; `just test`, or `just test-fast`)
Justfile                       Dev recipes: just validate / lint-fix / test / type-check / ci
.github/workflows/ci.yml       Same suite on push
runs/<timestamp>Z/             One run. Two files at the top, the rest below:
├── config.json                Every knob that produced the run
├── report.pdf                  `deconvolve report` output; the thing a human reads
└── artifacts/                  Everything else, flat -- see reporting.md
    ├── generator.keras, discriminator.keras, params.npz, history.npz
    ├── metrics.json, metrics_ibu.json, ibu_outcomes.json, ibu_weights.npz
    ├── metrics_omnifold.json, omnifold_weights.npz, timings_omnifold.json
    ├── timings.json            Merged across passes (see timing.md)
    ├── detector_level.pdf, particle_level.pdf, losses.pdf, selection.pdf
    └── report.tex               The filled-in template, kept for debugging
.cache/                        Regenerable cache; relocatable via DECONVOLVE_CACHE_DIR (see caching.md)
├── gaussian_*.npz              Generated Gaussian datasets, keyed on the promoted covariances
├── mass.npz, mult.npz, ...      Per-variable jet caches from the Zenodo download
└── jax/                         XLA persistent compilation cache
```

Module-level detail lives in docstrings, rendered under `docs/api/` by
`scripts/gen_api_docs.py`.

The cubic-response sweep (`deconvolve sweep`, `src/deconvolve/experiments/`,
`scripts/submit_sweep.zsh`) has been retired and sits under `legacy/`, which is
a holding pen and not a supported path: it is not importable as `ran`, not
covered by `just test`, and slated for deletion. Nothing in the package
references it.
