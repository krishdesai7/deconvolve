# Deconvolve: Reweighting Adversarial Networks

An adversarial neural network that learns per-event weights to correct simulated (Monte Carlo) distributions so they match observed data. Built with Keras 3 on the JAX backend.

## Motivation

In particle physics, Monte Carlo (MC) simulations are used to model detector responses and physical processes. These simulations never perfectly reproduce real data. There are always residual mismodelling effects. Traditional reweighting uses hand-tuned correction factors binned in one or two variables, which scales poorly to high-dimensional feature spaces.

Deconvolve replaces this with a learned reweighting: a **generator** network predicts a continuous per-event weight from particle-level (truth) features, while an **adversarial discriminator** tries to distinguish the reweighted simulation from real data. At convergence the discriminator can no longer tell them apart, and the generator's weights constitute an optimal correction.

## Model

The system is a two-player adversarial game over event weights:

| Component                | Input                      | Output                                         | Role                                          |
| ------------------------ | -------------------------- | ---------------------------------------------- | --------------------------------------------- |
| **Generator** $g(z)$     | Particle-level feature $z$ | Per-event weight (`logplus` = $\log(1+\exp)$ ) | Predict weights that make MC look like nature |
| **Discriminator** $d(x)$ | Detector-level feature $x$ | Data vs MC probability (`sigmoid`)             | Distinguish real data from reweighted MC      |

### Training loop

1. **Discriminator step**: freeze $g$, update $d$ to maximize weighted binary cross-entropy (classify Data vs reweighted Simulation)
2. **Generator step**: freeze $d$, update $g$ to minimize the same loss (fool the discriminator).
3. Repeat with 5:1 D:G update ratio

Weight normalization ensures the total MC yield is preserved:

$$w_i = \frac{g(z_i)}{\text{mean}(g(z))}$$

In equilibrium, both losses converge to $\log(2)$ and the reweighted MC matches data.

## Installation

Requires Python >= 3.14. Uses [`uv`](https://docs.astral.sh/uv/) for dependency management. One way to install it is with `pip install uv`; for alternatives see the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

```shell
git clone https://github.com/krishdesai7/deconvolve.git
cd Deconvolve
uv sync
```

This installs the `deconvolve` console script into `.venv/bin`. Commands below are written as `deconvolve ...`; from a checkout without an activated virtualenv, prefix them with `uv run` (`uv run deconvolve train --config params/1d_default.yaml`). Tab completion for subcommands, flags and enum values is available with `deconvolve --install-completion`.

### GPU Support

The JAX dependency is platform-resolved:

#### Linux, x86_64

Built against `jax[cuda13]` on x86_64 Linux, compiled against CUDA version 13.0. For NVIDIA GPUs, the CUDA 13 runtime libraries are available as pypi wheels that the JAX binary is built against, so only a compatible NVIDIA driver is needed.

#### macOS, arm64 (Apple Silicon)

The official macOS arm64 wheels for JAX do not provide GPU acceleration. Therefore JAX and consequentially deconvolve only offer CPU support on Apple Silicon. Experimental alternatives, such as `jax-mps` or `IREE`-based workflows, may enable Metal acceleration, but these configurations are not tested or supported by deconvolve. Users should independently validate their correctness and performance.

## Usage

### Gaussian Datasets

Gaussian datasets are configured via YAML files. Examples are provided in `params/`:

```shell
# 1D uncorrelated Gaussian
deconvolve train --config params/1d_default.yaml

# 2D with correlated covariance
deconvolve train --config params/2d_correlated.yaml

# 4D and 6D correlated
deconvolve train --config params/4d_correlated.yaml
deconvolve train --config params/6d_correlated.yaml

# Customize network and training
deconvolve train --config params/1d_default.yaml -u128 -l3 -e200
```

YAML config format (see `params/` for examples):

```yaml
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9 # scalar, vector, or full covariance matrix
sigma_true: 1.0
sigma_detector: 0.5
```

Sigma values are promoted to covariance matrices:

- scalar $\to \sigma^2 I$
- vector $\to \text{diag}(\sigma^2)$
- matrix $\to$ as-is

### Jet Substructure

```shell
# All 12 jet variables
deconvolve train -Djets

# Specific variables: mass and width
deconvolve train -Djets -vm -vw
```

### Other Options

```shell
# Reload an existing run (regenerate plots/metrics)
deconvolve train -r runs/2026-03-14T061023Z

# Enable debug logging for any command
deconvolve -L DEBUG train --config params/1d_default.yaml

# SLURM submission
sbatch scripts/submit.zsh --config params/2d_correlated.yaml
sbatch scripts/submit.zsh -Djets
```

| Long option           | Short option | Type          | Default        | Description                                                                         |
| --------------------- | ------------ | ------------- | -------------- | ----------------------------------------------------------------------------------- |
| `--batch-size`        | `-b`         | `int`         | `1024`         | Training batch size                                                                 |
| `--config`            |              | `Path`        | `None`         | Path to Gaussian YAML config                                                        |
| `--data-seed`         | `-d`         | `int`         | `42`           | Data generation, shuffle, split and batch order                                     |
| `--dataset`           | `-D`         | `DatasetName` | `gaussian`     | Dataset type: `gaussian` or `jets`                                                  |
| `--hidden-units`      | `-u`         | `int`         | `64`           | Units per hidden layer                                                              |
| `--lambda-dispersion` |              | `float`       | `0.015`        | Penalty on the variance of `g`'s weights                                            |
| `--load-run`          | `-r`         | `Path`        | `None`         | Path to an existing run directory to reload                                         |
| `--log-every`         |              | `int`         | `1`            | Log training progress every N epochs                                                |
| `--lr-d`              |              | `float`       | `1e-4`         | Discriminator learning rate (Adam)                                                  |
| `--lr-g`              |              | `float`       | `3e-5`         | Generator learning rate (Adam)                                                      |
| `--n-disc-steps`      | `-k`         | `int`         | `5`            | Discriminator updates per generator update                                          |
| `--n-epochs`          | `-e`         | `int`         | `100`          | Epochs to train (best checkpoint is always restored)                                |
| `--n-layers`          | `-l`         | `int`         | `2`            | Number of hidden layers                                                             |
| `--n-samples`         | `-n`         | `int`         | `500_000`      | Number of events per class (data + MC)                                              |
| `--plots/--no-plots`  |              | `bool`        | `True`         | Useful to disable for hyperparameter sweeps, bootstrapping, etc. Metrics still run. |
| `--run-dir`           |              | `Path`        | `None`         | Where to save this run. Default: timestamp under runs/.                             |
| `--seed`              | `-s`         | `int`         | system entropy | Weight-initialization seed (see [Seeding](#seeding))                                |
| `--var`               | `-v`         | `list[str]`   | all 12         | Repeat once for each jet substructure variable to use                               |

The pipeline will:

1. Generate (or load from cache) the dataset
2. Split into train / validation / test sets (70 / 10 / 20%)
3. Train the network for a fixed number of epochs, then restore the checkpoint minimizing detector-level MMD
4. Save models, training history, and plots to `runs/<UTC-timestamp>/`
5. Compute distance metrics on the test set

### Evaluation

Distance metrics can be computed independently on existing runs:

```shell
# Evaluate all runs
deconvolve evaluate

# Evaluate a single run
deconvolve evaluate runs/2026-03-14T061023Z

# Recompute even if metrics.json exists
deconvolve evaluate --force
```

This computes per-dimension 1D Wasserstein distances, Jensen-Shannon divergences, and triangular discriminator (Vincze-LeCam divergence) \[$\times10^3$\] at both detector and particle level, before and after reweighting. Results are saved to `metrics.json` in each run directory.

### Reports

One PDF dossier per run built from the JSON a run already writes. Contains configuration, timing, metrics tables and figures.

```shell
# Compile runs/<timestamp>/report.pdf
deconvolve report runs/2026-03-14T061023Z

# Rebuild one that already exists
deconvolve report runs/2026-03-14T061023Z --force

# Emit artifacts/report.tex alone, without a TeX installation
deconvolve report runs/2026-03-14T061023Z --no-compile
```

`report.tex` is written into `artifacts/`; `report.pdf` lands at the run root beside `config.json`. Compilation needs `pdflatex` on `PATH`. A run missing its
baseline, its timings or even its metrics still reports: the affected cells degrade to dashes or a labelled row rather than failing. `scripts/submit.zsh` ends with `deconvolve report`.

### Baseline Comparisons

#### IBU

Run IBU (Iterative Bayesian Unfolding) on the same datasets for head-to-head comparison:

```shell
# IBU — single run
deconvolve baseline ibu runs/2026-03-14T061023Z

# IBU — all runs
deconvolve baseline ibu
```

Results are saved to `metrics_ibu.json` in each run directory using the same metric format as deconvolve.

#### OmniFold

OmniFold is the second baseline this project implements. It runs in its own PEP 723-managed subprocess (Python 3.13 + TensorFlow) rather than in this project's environment, since TensorFlow has no wheels for this project's Python floor and cannot share a Keras backend with JAX:

```shell
# OmniFold — single run (writes metrics_omnifold.json)
deconvolve baseline omnifold runs/2026-03-14T061023Z

# Reload to redraw the figures with the OmniFold overlay
deconvolve train -r runs/2026-03-14T061023Z
```

`uv` must be on `PATH`, and its worker environment should be warmed on a login node before running on a cluster with no outbound network:

```shell
uv run --no-project src/deconvolve/baselines/_omnifold_worker.py
```

### Leakage Verification

A core correctness requirement is that the generator $g(z)$ never receives $z_\text{true}$, the particle-level values of measured data events, which are unknowable in a real experiment. The `leakage-check` command verifies this empirically via a **data poisoning test**:

```shell
# Clean run with z_true drawn from N(0, 1) as normal
deconvolve leakage-check --clean

# Poisoned run with z_true overwritten with sentinel value (default: -999) after x_data is generated
deconvolve leakage-check --poison [-S <sentinel>]
```

The poisoned run corrupts every data particle-level value to a nonsense sentinel while leaving $x_\text{data}$ (the reco-level observations the discriminator actually sees) unchanged. If $g$ had any access to $z_\text{true}$, the poisoned run would produce degraded weights. Both runs should report statistically identical Wasserstein and triangular discriminator improvements. Matching results confirm that no leakage path exists.

Both arms must share `--seed`, or initialization variance causes the arms to differ even with no leakage. With it fixed, detector-level results are
bit-identical between the clean and poisoned arms.

## Backend

This build relies on the JAX backend.

`src/deconvolve/__init__.py` sets `KERAS_BACKEND=jax` and `JAX_ENABLE_X64=0`. If using deconvolve as a library module rather than a command-line tool, ensure that **any `deconvolve.*` import must come before `import keras`**. `src/deconvolve/training/engine.py` raises a clear error if the backend has been initialized to something else.

### Precision

The project runs in single precision end to end. The pin is a single constant, `EVENT_DTYPE` in `src/deconvolve/coretypes/constants.py`, with the annotation alias `EventArray` alongside it; `JAX_ENABLE_X64=0` and the `dtype=` arguments in `src/deconvolve/training/models.py` follow from it.

Every jet observable is float32-clean. In particular, `mass` and `mult` survive a float32 round trip bit-exactly, and the others lose exactly half a ULP, the least a cast can cost. Across 320 paired seeds, single and double precision are indistinguishable on unfolding improvement to within 3.5 sigma, while the seed-to-seed spread within either precision is larger than the gap between them. `benchmarks/precision.py` reproduces the comparison and `benchmarks/compare_precision.py` runs the statistics.

`deconvolve.data.download` computes jet observables in double precision, because the ε protecting degenerate jets is below the smallest single precision denormal.

`src/deconvolve/training/engine.py` is a hand-rolled loop, since the two-optimizer min-max game does not fit a standard `keras.Model.fit`. It does, however, follow the standard Keras 3 + JAX pattern:

- Model state lives in JAX pytrees (`TrainState`) for the duration of training
- Updates are applied through `stateless_call`/`stateless_apply`
- Each step is a single jitted function
- Values are written back into the Keras models at the end, so the returned objects are ordinary saveable `keras.Model`s
- A whole run compiles to one XLA program: one epoch is a `lax.scan` over grouped batches, and the epoch loop is a `lax.scan` with a fixed trip count. Every epoch's parameters are retained, and the checkpoint minimizing detector-level MMD against a validation subsample is restored on the host once training finishes.
- Loss math is plain `jnp`. `stateless_call`/`stateless_apply` are the only Keras calls inside the trace; `lax.scan` and `jax.random` are native JAX.

`keras.ops.mean` is not used in this project because for double precision inputs it picks a single precision compute dtype internally and returns a result carrying ~1e-8 relative error. Reductions use `jnp.sum(...) / n` instead, which `tests/test_train.py` pins.

## Seeding

Two independent randomness axes, deliberately kept separate:

| Seed          | Controls                                               |
| ------------- | ------------------------------------------------------ |
| `--data-seed` | Generation, shuffle, train/val/test split, batch order |
| `--seed`      | Weight initialization only                             |

`--seed` defaults to a draw from system entropy, and the value used is recorded in `config.json`, so a run stays reproducible after the fact. To estimate model uncertainty, ensemble, i.e. rerun on the same inputs with fresh initializations and take the variance as the model uncertainty, is a loop over `--seed` at fixed `--data-seed`.

Because the networks are Dense-only (no dropout or batch norm) and Adam is deterministic, the two seeds together fully determine a run, up to non-deterministic GPU reductions. Force bitwise reproducibility by exporting `XLA_FLAGS=--xla_gpu_deterministic_ops=true`. This costs throughput and is not needed for variance estimates.

## Project Structure

```txt
Deconvolve/
├── src/deconvolve/                      Python package
│   ├── __init__.py               Pins KERAS_BACKEND=jax and JAX_ENABLE_X64=0
│   ├── __main__.py               Fallback entry point (python -m deconvolve)
│   ├── cli.py                    Unified Typer command tree; target of the `deconvolve` script
│   ├── py.typed                  PEP 561 typing marker
│   ├── coretypes/
│   │   ├── events.py             Split, Events, ZXY, Populations, DatasetSplits
│   │   ├── configs.py            GaussianConfig, RunConfig
│   │   ├── results.py            UnfoldingPopulations, VariableOutcome, IBUResult
│   │   ├── constants.py          Zenodo record, cache layout, jet plot metadata
│   │   ├── enums.py              CLI choice enums
│   │   └── types.py              TypedDicts and array aliases
│   ├── data/
│   │   ├── config.py             YAML config parsing, sigma promotion
│   │   ├── datasets.py           DatasetSplits, DeconvolveDataset, caching
│   │   ├── jets.py               Jet substructure loading and standardization
│   │   ├── device.py             Device-resident training form (TrainSplit/EvalSplit)
│   │   └── download.py           One-time Zenodo data download
│   ├── baselines/
│   │   ├── _shared.py            Run config and populations a baseline needs, minus the unfolder
│   │   ├── ibu.py                IBU (Iterative Bayesian Unfolding) baseline
│   │   ├── omnifold.py           OmniFold baseline, host half
│   │   └── _omnifold_worker.py   PEP 723 subprocess script (Python 3.13 + TensorFlow)
│   ├── uncertainty/
│   │   ├── design.py             Bootstrap x seed grid: resampling, one cell, loading
│   │   ├── variance.py           Two-way ANOVA components, covariances
│   │   └── report.py             Decomposition table, variance.npz, correlation.pdf
│   ├── training/
│   │   ├── models.py             Generator and discriminator architectures
│   │   ├── engine.py             Fused JAX training program
│   │   └── mmd.py                Weighted MMD for checkpoint selection
│   ├── evaluation/
│   │   ├── evaluate.py           Post-hoc distance metrics (Wasserstein, JS, triangular)
│   │   └── plotting.py           Detector-level, particle-level, and loss curve plots
│   ├── workflows/
│   │   ├── train.py              Training and reload workflow behind `deconvolve train`
│   │   └── leakage.py            Data-poisoning leakage check
│   ├── reporting/
│   │   ├── report.py             PDF dossier behind `deconvolve report`
│   │   └── templates/report.tex  LaTeX skeleton `report.py` fills in
│   └── instrumentation/
│       ├── timing.py             Optional per-phase wall-clock reporting
│       └── logging_config.py     Structured application logging
├── params/                       Gaussian config YAML files
│   ├── 1d_default.yaml
│   ├── 2d_correlated.yaml
│   ├── 4d_correlated.yaml
│   └── 6d_correlated.yaml
├── scripts/                      SLURM submission scripts
│   ├── submit.zsh                 End-to-end jet run: train, IBU, report
│   ├── submit_omnifold.zsh        OmniFold against an existing run directory
│   ├── submit_hparam.zsh          Hyperparameter arm sweep
│   ├── submit_precision.zsh       float32 vs float64 paired ensemble
│   └── submit_uncertainty.zsh     Bootstrap x seed variance grid
├── tests/                        pytest tests
├── .github/workflows/ci.yml      Lint, format, types, complexity, tests
├── Justfile                      Development recipes (just validate, just lint-fix, ...)
├── pyproject.toml                Project metadata and dependencies
├── runs/                         Output directory (timestamped subdirectories)
└── .cache/                       Cached datasets and XLA compilation cache
```

Module-level detail lives in the docstrings, rendered as the API Reference of the documentation site.

## Datasets

### Gaussian (Synthetic)

Configurable multivariate Gaussian distributions with correlated covariance matrices. Supports arbitrary dimensionality and correlation structure via YAML config files. Both truth and MC samples are smeared by additive Gaussian noise to simulate detector resolution, producing paired particle-level ($z$) and detector-level ($x$) features.

### Jet Substructure (Physics)

`Herwig` (data) vs `Pythia26` (MC) $Z+$ jets at high $p_T$ (200 GeV), with [`Delphes`](https://github.com/delphes/delphes) detector simulation. Automatically downloaded from [Zenodo record 3548091](https://zenodo.org/record/3548091) if not already present in `.cache/`.

| Variable | Symbol                  | Description                      |
| -------- | ----------------------- | -------------------------------- |
| `m`      | $m$ \[GeV\]             | Jet mass                         |
| `sdm`    | $\ln\rho$               | Log soft-drop jet mass           |
| `lha`    | $\lambda^{1}_{0.5}$     | Les Houches angularity           |
| `w`      | $w$                     | Jet width                        |
| `ang2`   | $\lambda^{1}_{2}$       | Jet angularity                   |
| `zg`     | $z_g$                   | Groomed jet momentum fraction    |
| `tau21`  | $\tau_{21}^{(\beta=1)}$ | N-subjettiness ratio             |
| `M`      | $M$                     | Jet constituent multiplicity     |
| `n_ch`   | $n_{ch}$                | Charged constituent multiplicity |
| `f_ch`   | $f_{ch}$                | Jet charge fraction              |
| `ptd`    | $p_T^D$                 | Transverse momentum dispersion   |
| `q`      | $q$                     | Jet charge                       |

`--var (-v)`, repeatable, selects any subset; `-vm -vM -vw -vtau21 -vzg -vsdm` reproduces the original six-observable OmniFold configuration. All variables are z-score standardized using MC gen-level statistics only (no information leakage).

## Output

Each run produces a timestamped directory under `runs/`. The root holds only human-readable `config.json` (run configuration, for reproducibility) and `report.pdf`. All other supporting material lives in `runs/artifacts/`:

```text
runs/<timestamp>/
├── report.pdf
├── config.json
└── artifacts/   figures, metrics/timings JSON, checkpoints, arrays
```

- **`generator.keras`**/**`discriminator.keras`** -- Saved model checkpoints
- **`history.npz`** -- Training loss history
- **`detector_level.pdf`** -- Histogram comparing data, MC, and reweighted MC at detector level with ratio panel
- **`particle_level.pdf`** -- Same comparison at particle level
- **`losses.pdf`** -- Training curves with log(2) equilibrium target
- **`selection.pdf`** -- Per-epoch MMD curves and the epoch model selection restored
- **`metrics.json`** -- Wasserstein, JS divergence, and triangular discriminator (before/after)
- **`metrics_ibu.json`**/**`ibu_weights.npz`** -- Same metrics and per-event weights from the IBU baseline (if run)
- **`metrics_omnifold.json`**/**`omnifold_weights.npz`** -- Same, from the OmniFold baseline (if run)
- **`timings.json`** -- Per-phase wall clock, when the run was made under `DECONVOLVE_TIMING=1`
- **`timings_omnifold.json`** -- OmniFold's own per-phase wall clock (if run)
- **`report.tex`** -- The LaTeX source `deconvolve report` compiles into the run root's `report.pdf`

## Training Hyperparameters

These are internal training defaults in `src/deconvolve/training/engine.py`; the CLI-exposed training options are listed above.

| Parameter           | Default | Description                                 |
| ------------------- | ------- | ------------------------------------------- |
| `n_epochs`          | 100     | Training epochs — a fixed `scan` trip count |
| `n_disc_steps`      | 5       | Discriminator updates per generator update  |
| `lr_g`              | 3e-5    | Generator learning rate (Adam)              |
| `lr_d`              | 1e-4    | Discriminator learning rate (Adam)          |
| `lambda_dispersion` | 0.015   | Penalty on the variance of `g`'s weights    |
| `hidden_units`      | 64      | Units per hidden layer                      |
| `n_layers`          | 2       | Number of hidden layers                     |

`lr_g` and `lambda_dispersion` are both measured rather than chosen, and they act on the same axis: the dispersion of `g`'s normalized MC weights. See "What
tuning actually found" and "The dispersion penalty: the trade made explicit" in `benchmarks/README.md`. The penalty is **on** by default, so a run left at these defaults is the configuration any comparison should be made against.

`n_epochs` is not a maximum in the early-stopping sense. `scan` needs a fixed trip count, so every run executes all of them; the best epoch is then restored
on the host by the detector-level MMD argmin.

## Development

Dev work on this repo requires a `uv sync`'ed checkout: `just`, `ruff`,
`pyrefly` and the rest are managed dependencies, not global installs.

```shell
uv run just validate    # all local, read-only validation (format, lint, typecheck, complexity, tests)
uv run just ci          # validate, then audit locked dependencies for known vulnerabilities
uv run just test        # pytest, forwards extra args
uv run just test-fast   # pytest, minus tests marked slow
uv run just typecheck   # type checking using pyrefly and ty
uv run just lint-fix    # apply safe lint fixes, then format
uv run just             # list every recipe
```

GitHub Actions runs the same suite on push.

- [`uv`](https://docs.astral.sh/uv/) is used to manage dependencies and run the recipes.
- [`ruff`](https://docs.astral.sh/ruff/) is used for linting and formatting.
- [`pyrefly`](https://pyrefly.org/) at `--min-severity info` plus `uv check --locked` for type checking.
- [`complexipy`](https://github.com/rohaquinlop/complexipy) enforces a maximum cognitive complexity of 10.

## Dependencies

- [`Beartype`](https://beartype.readthedocs.io/en/latest/) >= 0.22.9
- [`JAX`](https://docs.jax.dev/) >= 0.11.1 ( `jax[cuda13]` on x86_64 Linux)
- [`Jaxtyping`](https://jaxtyping.readthedocs.io/en/latest/) >= 0.3.11
- [`Keras`](https://keras.io/) >= 3.15.1
- [`NumPy`](https://numpy.org/) >= 2.5.3
- [`Matplotlib`](https://matplotlib.org/) >= 3.11.2
- [`Typer`](https://typer.tiangolo.com/) >= 0.27.2
- [`PyYAML`](https://pyyaml.org/) >= 6.0.3
