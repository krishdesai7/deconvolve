<!-- markdownlint-disable no-inline-html -->
# Caching & Seeding

Reproducibility and execution speed in <span style="font-variant: small-caps;">Deconvolve</span> rely on a unified cache directory hierarchy and deterministic PRNG key management.

---

## Unified Cache Hierarchy

All regenerable artifacts share a common cache root configured by `deconvolve.coretypes.constants`:

```shell
.cache/                    # Or $DECONVOLVE_CACHE_DIR
├── datasets/              # Generated Gaussian datasets (.npz)
├── jets/                  # Downloaded & standardized Zenodo jet data
├── xla/                   # Persistent XLA compilation cache
├── pytest/                # Pytest cache
├── ruff/                  # Ruff linter cache
└── complexipy/            # Complexity checker cache
```

### Relocating the Cache (`DECONVOLVE_CACHE_DIR`)

On high-performance computing (HPC) clusters where `$HOME` has strict quotas, set `DECONVOLVE_CACHE_DIR` to point to scratch storage. E.g.,

```shell
export DECONVOLVE_CACHE_DIR=$SCRATCH/deconvolve_cache
```

<span style="font-variant: small-caps;">Deconvolve</span> deliberately avoids `XDG_CACHE_HOME` because that variable often points to `~/.cache`, which would silently displace local checkouts.

---

## Two-Tier Seeding Architecture

<span style="font-variant: small-caps;">Deconvolve</span> separates dataset generation from model initialisation using two independent seeds:

| Seed Parameter | Default | Role           | What it Controls                                                   |
| :------------- | :------ | :------------- | :----------------------------------------------------------------- |
| `data_seed`    | `42`    | **Data Axis**  | Synthetic Gaussian draws, train/val/test splits, event subsampling |
| `seed`         | `42`    | **Model Axis** | Neural network weight initialisation, batch reshuffling            |

### Why Two Seeds?

In variance estimation, keeping `data_seed` fixed while varying `seed` allows measuring the initialisation variance of the algorithm on the exact same dataset.

Conversely, keeping `seed` fixed while varying `data_seed` or bootstrapping allows measuring sample variance (statistical uncertainty) independent of optimiser stochasticity.

---

## JAX PRNG Determinism

JAX uses an explicit, stateless PRNG model:

```python
key = jax.random.PRNGKey(seed)
key, subkey = jax.random.split(key)
```

Within the training loop, `deconvolve.data.device.train_indices` draws epoch batch orders directly on device using split PRNG keys, ensuring:

- Zero host-to-device roundtrips during epoch reshuffling.
- Completely deterministic batch ordering given the same seed.
- Multi-GPU launch parity without shared state races.
