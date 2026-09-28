<!-- markdownlint-disable no-inline-html -->
# Configuration

Every CLI option that describes a preference, like `--n-epochs` or `--batch-size`, can be set once in a configuration file or an environment variable instead of being typed on every invocation.

!!! note
    This page is about default values for CLI options. The Gaussian YAML file passed with `train --config` describes a dataset, and is covered in [Datasets](datasets.md).

Tables mirror the command tree: `[train]` sets options for `deconvolve train`, `[baseline.omnifold]` for `deconvolve baseline omnifold`, and so on. In `pyproject.toml`, the same tables should be declared under `tool.deconvolve`.

Configuration files are validated before any command runs. An unknown table or key is an error, with a suggestion when there's a close match:

```text
deconvolve.toml [train]: unknown option `n-epoch`, did you mean `n-epochs`?
```

Values are also checked against the option's type and range, so `n-epochs = -5` is rejected.

<span style="font-variant: small-caps;">Deconvolve</span> supports persistent configuration files at both the project- and user-level.

## Project-level configuration

<span style="font-variant: small-caps;">Deconvolve</span> looks for a `pyproject.toml` with a `[tool.deconvolve]` table, or a `deconvolve.toml` file, in the current directory and then in each parent directory. The search stops at the first directory that has either one, but never crosses beyond the repository root.

!!! important
    - If there is no `[tool.deconvolve]` table in `pyproject.toml`, the file will be ignored, and <span style="font-variant: small-caps;">Deconvolve</span> will continue searching in the directory hierarchy.
    - If both files exist in the same directory, configuration will be read from `deconvolve.toml`. The `[tool.deconvolve]` table in `pyproject.toml` will be ignored entirely. The two are *not* merged.

### User-level configuration

<span style="font-variant: small-caps;">Deconvolve</span> also supports a user-level `deconvolve.toml` that applies to every project. It should be placed at `$XDG_CONFIG_HOME/deconvolve/deconvolve.toml`, or `~/.config/deconvolve/deconvolve.toml` when `XDG_CONFIG_HOME` is not set. It uses the same format as a project `deconvolve.toml`. User-level configuration must use the `deconvolve.toml` format, rather than the `pyproject.toml` format, as a `pyproject.toml` is intended to define a Python project.

When a user-level file and a project-level file are both found, they are merged key by key, and the project-level value overrides the user-level value for any key that both files set. Array values such as `variable` are replaced, not concatenated.

### Configuration keys

Keys typically resemble long option names. Both snake_case (`n_epochs`) and kebab-case (`n-epochs`) are accepted. Each command's page in the [CLI Reference](cli.md) lists its options.

#### Global

| Key | Value |
| :--- | :--- |
| `log-level` | `"debug"`, `"info"`, `"warning"`, `"error"` or `"critical"` |

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve]
    log-level = "debug"
    ```

=== "deconvolve.toml"

    ```toml
    log-level = "debug"
    ```

#### `[train]`

<div class="grid cards" markdown="1">

| Key | Value |
| :--- | :--- |
| `dataset` | `"gaussian"` or `"jets"` |
| `config` | path |
| `n-samples` | integer ≥ 1 |
| `variable` | array of strings |
| `data-seed` | integer |
| `batch-size` | integer ≥ 1 |
| `hidden-units` | integer ≥ 1 |
| `n-layers` | integer ≥ 1 |
| `n-epochs` | integer ≥ 1 |

| Key | Value |
| :--- | :--- |
| `n-disc-steps` | integer ≥ 1 |
| `lr-g` | float ≥ 0 |
| `lr-d` | float ≥ 0 |
| `lambda-dispersion` | float ≥ 0 |
| `seed` | integer |
| `run-dir` | path |
| `plots` | boolean |
| `log-every` | integer ≥ 1 |

</div>

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.train]
    dataset = "gaussian"
    config = None
    n-samples = 500000
    variable = None # all observables
    data-seed = 42
    batch-size = 1024
    hidden-units = 64
    n-layers = 2
    n-epochs = 100
    n-disc-steps = 5
    lr-g = 3e-5
    lr-d = 1e-4
    lambda-dispersion = 0.015
    seed = None # random
    run-dir = None # a UTC timestamp under `runs/`
    plots = True
    log-every = 1
    ```

=== "deconvolve.toml"

    ```toml
    [train]
    dataset = "gaussian"
    config = None
    n-samples = 500000
    variable = None # all observables
    data-seed = 42
    batch-size = 1024
    hidden-units = 64
    n-layers = 2
    n-epochs = 100
    n-disc-steps = 5
    lr-g = 3e-5
    lr-d = 1e-4
    lambda-dispersion = 0.015
    seed = None # random
    run-dir = None # a UTC timestamp under `runs/`
    plots = True
    log-every = 1
    ```
    

#### `[report]`

| Key | Value |
| :--- | :--- |
| `compile-pdf` | boolean |

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.report]
    compile-pdf = True
    ```
    
=== "deconvolve.toml"

    ```toml
    [report]
    compile-pdf = True
    ```
    

#### `[baseline.ibu]`

| Key | Value |
| :--- | :--- |
| `n-iterations` | integer ≥ 1 |
| `purity-threshold` | float |

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.baseline.ibu]
    n-iterations = 10
    purity-threshold = 0.7071 # sqrt(0.5)
    ```
    
=== "deconvolve.toml"

    ```toml
    [baseline.ibu]
    n-iterations = 10
    purity-threshold = 0.7071 # sqrt(0.5)
    ```

#### `[baseline.omnifold]`

| Key | Value |
| :--- | :--- |
| `n-iterations` | integer ≥ 1 |
| `n-epochs` | integer ≥ 1 |
| `batch-size` | integer ≥ 1 |

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.baseline.omnifold]
    n-iterations = 3
    n-epochs = 50
    batch-size = 512
    ```
    
=== "deconvolve.toml"

    ```toml
    [baseline.omnifold]
    n-iterations = 3
    n-epochs = 50
    batch-size = 512
    ```
    

#### `[uncertainty.freeze]`

These are the settings every cell of the design trains with; see [Uncertainty designs](#uncertainty-designs).

<div class="grid cards" markdown="1">

| Key | Value |
| :--- | :--- |
| `n-datasets` | integer ≥ 2 |
| `n-seeds` | integer ≥ 2 |
| `n-eval` | integer ≥ 1 |
| `dataset` | `"gaussian"` or `"jets"` |
| `config` | path |
| `variable` | array of strings |
| `batch-size` | integer ≥ 1 |
| `n-samples` | integer ≥ 1 |
| `hidden-units` | integer ≥ 1 |

| Key | Value |
| :--- | :--- |
| `n-layers` | integer ≥ 1 |
| `n-epochs` | integer ≥ 1 |
| `n-disc-steps` | integer ≥ 1 |
| `lr-g` | float ≥ 0 |
| `lr-d` | float ≥ 0 |
| `lambda-dispersion` | float ≥ 0 |
| `data-seed` | integer |
| `init-seed` | integer |
| `resample` | `"both"`, `"data"` or `"mc"` |

</div>

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.uncertainty.freeze]
    n-datasets = 8
    n-seeds = 8
    n-eval = 100000
    dataset = "jets"
    config = None
    variable = None # all observables
    batch-size = 1024
    n-samples = 500000
    hidden-units = 64
    n-layers = 2
    n-epochs = 100
    n-disc-steps = 5
    lr-g = 3e-5
    lr-d = 1e-4
    lambda-dispersion = 0.015
    data-seed = 42
    init-seed = 0
    resample = "both"
    ```
    
=== "deconvolve.toml"

    ```toml
    [uncertainty.freeze]
    n-datasets = 8
    n-seeds = 8
    n-eval = 100000
    dataset = "jets"
    config = None
    variable = None # all observables
    batch-size = 1024
    n-samples = 500000
    hidden-units = 64
    n-layers = 2
    n-epochs = 100
    n-disc-steps = 5
    lr-g = 3e-5
    lr-d = 1e-4
    lambda-dispersion = 0.015
    data-seed = 42
    init-seed = 0
    resample = "both"
    ```
    

#### `[uncertainty.collect]`

| Key | Value |
| :--- | :--- |
| `n-bins` | integer ≥ 2 |

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.uncertainty.collect]
    n-bins = 20
    ```
    
=== "deconvolve.toml"

    ```toml
    [uncertainty.collect]
    n-bins = 20
    ```
    

#### `[leakage-check]`

| Key | Value |
| :--- | :--- |
| `poison` | boolean |
| `sentinel` | float |
| `seed` | integer |
| `init-seed` | integer |

Default:

=== "pyproject.toml"

    ```toml
    [tool.deconvolve.leakage-check]
    poison = False
    sentinel = -999.0
    seed = 42
    init-seed = 0
    ```

=== "deconvolve.toml"

    ```toml
    [leakage-check]
    poison = False
    sentinel = -999.0
    seed = 42
    init-seed = 0
    ```
!!! important
    `uncertainty run`, `evaluate`, and `config show` have no configurable keys.

## Environment variables

Every option that can be set with a key in a configuration file can also be set with an environment variable. The name is derived from the configuration key, all upper-case with `_` as the separator. For example:

| Key | Environment variable |
| :--- | :--- |
| `tool.deconvolve.log-level` | `DECONVOLVE_LOG_LEVEL` |
| `tool.deconvolve.train.n-epochs` | `DECONVOLVE_TRAIN_N_EPOCHS` |
| `tool.deconvolve.baseline.ibu.n-iterations` | `DECONVOLVE_BASELINE_IBU_N_ITERATIONS` |
| `tool.deconvolve.uncertainty.freeze.lr-g` | `DECONVOLVE_UNCERTAINTY_FREEZE_LR_G` |

Each command's `--help` shows the variable for every option that has one, as `[env var: ...]`.

## Precedence

From lowest to highest priority:

1. The option's built-in default.
2. The user-level `deconvolve.toml`.
3. The project-level `deconvolve.toml` or `[tool.deconvolve]`.
4. Environment variables.
5. Flags on the command line.

`deconvolve <command> --help` shows the value that will apply after the configuration files are taken into account.

## Options that cannot be configured

These must be given on the command line each time. Setting one in a configuration file is an error, and none of them has an environment variable:

- Positional arguments: `RUN_DIR`, `DESIGN_DIR` and `CELL`.
- `--force`, on every command that has it. A destructive rebuild cannot be inherited from a configuration file.
- `train --load-run`, which names one specific run.
- `config show --json`, an output format for one invocation.
- `uncertainty collect`'s `--n-datasets`, `--n-seeds`, `--data-seed` and `--init-seed`. These are read from the design's `design.json`; see below.

## Uncertainty designs

A variance design runs many `uncertainty run` cells, often as a SLURM array. If each cell were to read the configuration files, editing `deconvolve.toml` mid-array would silently compute different cells with different settings.

So `uncertainty run` reads neither configuration files nor environment variables. Instead, `uncertainty freeze` resolves the settings once, with the precedence above, and writes them to `DESIGN_DIR/design.json`, along with where each value came from. Every cell reads that file. An option passed on a cell's own command line still overrides the frozen value for that cell.

Hence a design must be configured under `[uncertainty.freeze]`, not `[uncertainty.run]`; the latter will raise an unknown-table error.

```shell
# once, before submitting. Writes runs/uncertainty_example/design.json
deconvolve uncertainty freeze runs/uncertainty_example -B8 -S8
# Cell 0 reads design.json
deconvolve uncertainty run 0 runs/uncertainty_example
deconvolve uncertainty collect runs/uncertainty_example
```

## Inspecting the configuration

`deconvolve config show` lists the configuration files found and every value set, with the file each value came from:

```text
$ deconvolve config show
Configuration files
 ~/.config/deconvolve/deconvolve.toml                  global
 ~/work/analysis/deconvolve.toml                       project

 setting                     value  source
 log-level                   debug  deconvolve.toml (project)
 baseline.omnifold.n-epochs  30     deconvolve.toml (project)
 train.batch-size            2048   deconvolve.toml (project)
 train.n-epochs              500    deconvolve.toml (project)
 train.n-layers              3      deconvolve.toml (global)

      Environment-only (not layered)
 DECONVOLVE_CACHE_DIR  .cache
 DECONVOLVE_TIMING     off (unset)
```

`deconvolve config show train` limits the listing to one command, and `--json` prints the same information as JSON. `config show` still runs when a configuration file is invalid, and reports the error.

`config show` only lists values set in files. Values that come from environment variables or flags aren't shown. For the complete record of a run, including those, see the `_origin` block in the run's `config.json`.

## Environment-only settings

The following settings can only be set through environment variables:

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `DECONVOLVE_CACHE_DIR` | `.cache` | Directory for downloaded datasets and the XLA compilation cache. |
| `DECONVOLVE_TIMING` | off | Set to `1` to record per-phase timings to `timing.json`. `0`, `false`, `no`, `off` and the empty string all mean off. |
