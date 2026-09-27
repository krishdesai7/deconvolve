<!-- markdownlint-disable no-inline-html -->
# Caching & Seeding

<span style="font-variant: small-caps;">Deconvolve</span> caches everything it can regenerate (datasets and compiled programs) under one directory, and controls randomness with two independent seeds.

---

## Cache directory

All regenerable data share one cache root, `.cache/` in the current directory by default:

```text
.cache/                        # or $DECONVOLVE_CACHE_DIR
├── gaussian_<hash>.npz        # one file per generated Gaussian dataset
├── mass.npz, mult.npz, …      # one file per jet observable, from Zenodo
└── jax/                       # persistent XLA compilation cache
```

- **Gaussian datasets** are cached under a hash of everything that determines the sample: the means and covariances, the number of events, `data_seed` and the version of the random-number generation. Changing any of them produces a new file rather than reusing a stale one.
- **Jet observables** are computed once from the Zenodo release on first use, and stored one file per observable, at both levels and for both generators. The raw downloads are deleted afterwards. The cached values are unstandardized; each run standardizes the observables it loads.
- **Compiled programs.** Compiling the training program is the largest single cost in a short run: a few seconds, against a few hundredths of a second per epoch on a GPU. XLA's persistent cache stores compiled programs on disk, keyed on the program and the JAX version, so later runs of the same architecture, in any process, reuse them. A cache entry that no longer matches is recompiled, never reused incorrectly. The `JAX_COMPILATION_CACHE_DIR` and `JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` environment variables take precedence if set.

Nothing in the cache needs to be preserved. Deleting it costs a download and some recompilation, not results.

### Relocating the cache

On clusters where `$HOME` has a small quota, point `DECONVOLVE_CACHE_DIR` at scratch storage:

```shell
export DECONVOLVE_CACHE_DIR=$SCRATCH/deconvolve-cache
```

This moves the whole tree. `~` is expanded, and an empty value is treated as unset, so a batch system that forwards an unset variable as an empty string still gets the default. The variable is read once, when <span style="font-variant: small-caps;">Deconvolve</span> is imported, so it must be set in the environment rather than in a configuration file (see [Configuration](../user-guide/configuration.md#environment-only-settings)).

<span style="font-variant: small-caps;">Deconvolve</span> does not derive the location from `XDG_CACHE_HOME`. That variable is set, or defaults to `~/.cache`, on most Linux systems, so following it would silently move the cache of every existing checkout and leave the downloaded jet data behind.

### Development tool caches

The development tools (pytest, ruff, complexipy and Python's bytecode cache) also write under `.cache/`, to keep the repository root clean. These are configured in `pyproject.toml` and `.env`, and do **not** follow `DECONVOLVE_CACHE_DIR`: they are small, and relocating them would make the variable mean two things.

---

## Seeding

<span style="font-variant: small-caps;">Deconvolve</span> uses two independent seeds:

| Seed | Default | Controls |
| :--- | :--- | :--- |
| `data_seed` | `42` | The Gaussian sample (for Gaussian data), the train/validation/test split, the order of training batches, and the subsamples used for [MMD model selection](../theory/mmd.md) |
| `seed` | random | The initialization of the network weights |

When `seed` is not given, one is drawn at random. Both seeds, including a randomly drawn one, are recorded in the run's `config.json`, so every run can be reproduced afterwards.

### Why two seeds

Separating the seeds separates two sources of run-to-run variation. With `data_seed` fixed and `seed` varied, every run sees the same events in the same order, and the spread of the results measures the sensitivity to initialization alone. With `seed` fixed and `data_seed` varied, the spread measures the sensitivity to the split and batch order.

Neither measures the statistical uncertainty due to the finite sample. For the jet dataset, varying `data_seed` changes how the same events are split and ordered, not which events are used. The [uncertainty design](uncertainty.md) therefore holds `data_seed` fixed and resamples the events instead (a bootstrap), crossed with a range of initialization seeds.

Comparisons between hyperparameter settings should share `data_seed`, so that every setting is trained and evaluated on identical events.

---

## Random-number generation in training

Batch order is generated on the device with JAX's stateless random-number functions. Training derives a key from `data_seed` and splits it once per epoch, and each epoch's permutation of the training events is a pure function of its key. As a result:

- Shuffling requires no transfer between host and device.
- Batch order depends only on `data_seed`. Nothing else consumes random numbers from the same sequence, so a second training run on the same splits sees exactly the same batches.

Each epoch is divided into groups of `n_disc_steps` batches, with one generator update per group. Events that do not fill a complete group are skipped for that epoch; because the permutation is redrawn every epoch, a different random subset is skipped each time.

The networks have no dropout or other stochastic layers, and the optimizer is deterministic, so the two seeds fully determine a training run, up to the non-deterministic accumulation order of some GPU reductions (see [Precision & Hardware](precision.md#reproducibility-on-gpus)).
