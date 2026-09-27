<!-- markdownlint-disable ul-indent no-inline-html -->
# CLI Reference

The `deconvolve` CLI is a single Typer command tree with the following subcommands:

- `train`
- `evaluate`
- `report`
- `leakage-check`
- `baseline`
    - `ibu`
    - `omnifold`
- `uncertainty`
    - `freeze`
    - `run`
    - `collect`
- `config show`

Most options can also be set in a `deconvolve.toml`, a `[tool.deconvolve]` table in `pyproject.toml`, or a `DECONVOLVE_*` environment variable, e.g. `DECONVOLVE_TRAIN_N_EPOCHS=500`. A flag on the command line always wins. See [Configuration](configuration.md) for where the files are found, the precedence order, and which options can't be configured.

---

## Global Options

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--log-level` | `-L` | `LogLevel` | `info` | Application log level. Options: `debug`, `info`, `warning`, `error`, `critical`. |
| `--install-completion` | | `bool` | | Install shell autocompletion. |
| `--show-completion` | | `bool` | | Print the completion script. |
| `--help` | | `bool` | | |

!!! note
    `--log-level` (`-L`) can precede a subcommand, and must be placed **before** the subcommand. For example:

    ```shell
    deconvolve -Ldebug train --config params/1d_default.yaml
    ```
---

## `deconvolve train`

Usage:

```shell
deconvolve train [-D{gaussian|jets}]
                 [--config <path>]
                 [-v<str>]...
                 [-n<int>]
                 [--data-seed <int>]
                 [-r<path>]
                 [-u<int>]
                 [-l<int>]
                 [-e<int>]
                 [-b<int>]
                 [-k<int>]
                 [--lr-g <float>]
                 [--lr-d <float>]
                 [--lambda-dispersion <float>]
                 [--seed <int>]
                 [--run-dir <path>]
                 [--plots | --no-plots]
                 [--log-every <int>]
```

### Dataset

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--dataset` | `-D` | `Dataset` | `gaussian` | Dataset to train on. Options: `gaussian`, `jets`. |
| `--config` | | `Path` | `None` | YAML config file (Gaussian datasets only). |
| `--n-samples` | `-n` | `int` | `500000` | Number of events to generate/load. |
| `--var` | `-v` | `str`, repeatable | all | Jet substructure variable(s) to train on, e.g. `-vm -vw`. Ignored for `gaussian`. |
| `--data-seed` | | `int` | `42` | Seed for dataset sampling and the train/val/test split. |
| `--load-run` | `-r` | `Path` | `None` | Reload a previously saved run directory instead of starting fresh. |

### Architecture & Optimization

| Long option | Short option | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--hidden-units` | `-u` | `int` | `64` | Hidden units per dense layer, generator and discriminator. |
| `--n-layers` | `-l` | `int` | `2` | Number of hidden dense layers, generator and discriminator. |
| `--n-epochs` | `-e` | `int` | `100` | Number of training epochs. |
| `--batch-size` | `-b` | `int` | `1024` | Batch size per training step. |
| `--n-disc-steps` | `-k` | `int` | `5` | Discriminator updates per generator update. |
| `--lr-g` | | `float` | `3e-5` | Generator learning rate (<span style="font-variant: small-caps;">Adam</span>). Tuned; see `benchmarks/README.md`. |
| `--lr-d` | | `float` | `1e-4` | Discriminator learning rate (<span style="font-variant: small-caps;">Adam</span>). |
| `--lambda-dispersion` | | `float` | `0.015` | Penalty on the variance of the generator's normalized weights. `0` disables it. |
| `--seed` | | `int` | `None` | Seed for model initialization and shuffle order. Defaults to a random value. |

### Output

| Long option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--run-dir` | `Path` | `None` | Where to save this run. Defaults to UTC timestamp under `runs/` |
| `--plots` / `--no-plots` | `bool` | `True` | Draw diagnostic figures. Metrics are computed either way. |
| `--log-every` | `int` | `1` | Log every N epochs. |

---

## `deconvolve evaluate`

Compute distance metrics for one run, or every run under a parent directory (see [Evaluation & Metrics](evaluation.md)).

Usage:

```shell
deconvolve evaluate RUN_DIR [--force]
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `RUN_DIR` | `Path` | Run directory to evaluate. |

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--force` | `bool` | `False` | Recompute even if `metrics.json` already exists. |

---

## `deconvolve report`

Compile a run directory into one PDF dossier (see [Reporting & Artifacts](reporting.md)).

Usage:

```shell
deconvolve report RUN_DIR [--force] [--no-compile]
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `RUN_DIR` | `Path` | Run directory to evaluate. |

| Option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--force` | `bool` | `False` | Rebuild an existing `report.pdf`. |
| `--compile`/`--no-compile` | `bool` | `True` | Compile the LaTeX, or stop at `artifacts/report.tex`. |

---

## `deconvolve baseline`

Run comparison baselines against the same run directory a `deconvolve train` call produced (see [Comparison Baselines](baselines.md)).

### `deconvolve baseline ibu`

Usage:

```shell
deconvolve baseline ibu RUN_DIR [--force] [-i<int>] [--purity-threshold <float>]
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `RUN_DIR` | `Path` | Run directory to add the IBU baseline to. |

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--force` | | `bool` | `False` | Recompute even if already evaluated. |
| `--niter` | `-i` | `int` | `10` | Number of IBU iterations. |
| `--purity-threshold` | | `float` | `√0.5 ≈ 0.7071` | Purity threshold used by the response matrix. |

### `deconvolve baseline omnifold`

Runs in a Python 3.13 subprocess; needs `uv` on `PATH`, and CUDA 12 (without it, TensorFlow silently falls back to CPU). On many HPC systems, CUDA 12.9 can be loaded with `module load cudatoolkit/12`.

```shell
deconvolve baseline omnifold RUN_DIR [--force] [-i<int>] [-e<int>] [-b<int>]
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `RUN_DIR` | `Path` | Run directory to add the OmniFold baseline to. |

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--force` | | `bool` | `False` | Recompute even if already evaluated. |
| `--niter` | `-i` | `int` | `3` | Number of OmniFold iterations. |
| `--n-epochs` | `-e` | `int` | `50` | Number of epochs per iteration. |
| `--batch-size` | `-b` | `int` | `512` | Batch size. |

---

## `deconvolve leakage-check`

Verifies that `z_true` never reaches a network.

Usage:

```shell
deconvolve leakage-check [--poison | --clean]
                         [-S<float>]
                         [--seed <int>]
                         [--init-seed <int>]
```

| Long option | Short | Type | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--poison` / `--clean` | `-X` / | `bool` | `--clean` | Poison mode injects a sentinel to confirm the check would actually catch a leak. |
| `--sentinel` | `-S` | `float` | `-999.0` | Sentinel value used in `--poison` mode. |
| `--seed` | | `int` | `42` | Model initialization seed. |
| `--init-seed` | | `int` | `0` | Bootstrap/init seed. |

---

## `deconvolve uncertainty`

Bootstrap × seed variance decomposition, run as a `freeze` once, followed by many `run`
cells (typically a SLURM array), then a final `collect`.

### `deconvolve uncertainty freeze`

Usage:

```shell
deconvolve uncertainty freeze DESIGN_DIR
                                [--force]
                                [-B<int>]
                                [-S<int>]
                                [--n-eval <int>]
                                [-D{gaussian|jets}]
                                [-v<str>]...
                                [--config <path>]
                                [-b<int>]
                                [-n<int>]
                                [-u<int>]
                                [-l<int>]
                                [-k<int>]
                                [--lr-g <float>]
                                [--lr-d <float>]
                                [--lambda-dispersion <float>]
                                [--data-seed <int>]
                                [--init-seed <int>]
                                [--resample {both|data|mc}]

```

Resolves the full config stack once and writes `DESIGN_DIR/design.json`, which every `uncertainty run` cell then reads instead of the ordinary config layers.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `DESIGN_DIR` | `Path` | Design directory to freeze. |

| Long option | Short | Type | Default |
| :--- | :--- | :--- | :--- |
| `--force` | | `bool` | `False` |
| `--n-datasets` | `-B` | `int` | `8` |
| `--n-seeds` | `-S` | `int` | `8` |
| `--n-eval` | | `int` | `100000` |
| `--dataset` | `-D` | `Dataset` | `jets` |
| `--var` | `-v` | `str`, repeatable | all |
| `--config` | | `Path` | `None` |
| `--batch-size` | `-b` | `int` | `1024` |
| `--n-samples` | `-n` | `int` | `500000` |
| `--hidden-units` | `-u` | `int` | `64` |
| `--n-layers` | `-l` | `int` | `2` |
| `--n-epochs` | `-e` | `int` | `100` |
| `--n-disc-steps` | `-k` | `int` | `5` |
| `--lr-g` | | `float` | `3e-5` |
| `--lr-d` | | `float` | `1e-4` |
| `--lambda-dispersion` | | `float` | `0.015` |
| `--data-seed` | | `int` | `42` |
| `--init-seed` | | `int` | `0` |
| `--resample` | | `Resample` | `both` |

### `deconvolve uncertainty run`

Usage:

```shell
deconvolve uncertainty run CELL DESIGN_DIR
                                [--force]
                                [-B<int>]
                                [-S<int>]
                                [--n-eval <int>]
                                [-D{gaussian|jets}]
                                [-v<str>]...
                                [--config <path>]
                                [-b<int>]
                                [-n<int>]
                                [-u<int>]
                                [-l<int>]
                                [-k<int>]
                                [--lr-g <float>]
                                [--lr-d <float>]
                                [--lambda-dispersion <float>]
                                [--data-seed <int>]
                                [--init-seed <int>]
                                [--resample {both|data|mc}]

```

Trains one `(bootstrap dataset, init seed)` cell of the design. Takes the same argument and options as `freeze` plus a required CELL argument, but reads their values from the frozen `DESIGN_DIR/design.json` rather than the config stack. An explicit flag on the command line still overrides the frozen value.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `CELL` | `int` | Cell number to train. |
| `DESIGN_DIR` | `Path` | Design directory to read options from. |

| Long option | Short | Type | Default |
| :--- | :--- | :--- | :--- |
| `--force` | | `bool` | `False` |
| `--n-datasets` | `-B` | `int` | Value from `design.json`. |
| `--n-seeds` | `-S` | `int` | Value from `design.json`. |
| `--n-eval` | | `int` | Value from `design.json`. |
| `--dataset` | `-D` | `Dataset` | Value from `design.json`. |
| `--var` | `-v` | `str`, repeatable | Value from `design.json`. |
| `--config` | | `Path` | Value from `design.json`. |
| `--batch-size` | `-b` | `int` | Value from `design.json`. |
| `--n-samples` | `-n` | `int` | Value from `design.json`. |
| `--hidden-units` | `-u` | `int` | Value from `design.json`. |
| `--n-layers` | `-l` | `int` | Value from `design.json`. |
| `--n-epochs` | `-e` | `int` | Value from `design.json`. |
| `--n-disc-steps` | `-k` | `int` | Value from `design.json`. |
| `--lr-g` | | `float` | Value from `design.json`. |
| `--lr-d` | | `float` | Value from `design.json`. |
| `--lambda-dispersion` | | `float` | Value from `design.json`. |
| `--data-seed` | | `int` | Value from `design.json`. |
| `--init-seed` | | `int` | Value from `design.json`. |
| `--resample` | | `Resample` | Value from `design.json`. |

### `deconvolve uncertainty collect`

Usage:

```shell
deconvolve uncertainty collect DESIGN_DIR
                                [-B<int>]
                                [-S<int>]
                                [--n-bins <int>]
                                [--data-seed <int>]
                                [--init-seed <int>]
```

Decomposes a finished design and writes an uncertainty table, `.npz`, and figure for it.

| Argument | Type | Description |
| :--- | :--- | :--- |
| `DESIGN_DIR` | `Path` | Design directory to collect. |

| Long option | Short | Type | Default |
| :--- | :--- | :--- | :--- |
| `--n-datasets` | `-B` | `int` | Value from `design.json`. |
| `--n-seeds` | `-S` | `int` | Value from `design.json`. |
| `--n-bins` | | `int` | `20` (not frozen; layerable) |
| `--data-seed` | | `int` | Value from `design.json`. |
| `--init-seed` | | `int` | Value from `design.json`. |

---

## `deconvolve config show`

Prints the resolved value of every setting and its source.

```shell
deconvolve config show [COMMAND] [--json]
```

| Argument | Type | Description |
| :--- | :--- | :--- |
| `COMMAND` | `str` | Scope the listing to one command, e.g. `deconvolve config show train`. |

| Long option | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--json` | `bool` | `False` | Emit the same content as machine-readable JSON. |
