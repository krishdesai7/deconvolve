# Seeding

Two independent randomness axes, deliberately kept separate:

| Seed        | Set by                            | Controls                                                         |
| ----------- | --------------------------------- | ------------------------------------------------------------------ |
| `data_seed` | `DeconvolveDataset` / `load_jet_dataset` | generation, shuffle, train/val/test split, per-epoch batch order |
| `seed`      | `engine.train`                    | weight initialization only                                       |

`engine.train(seed=None)` draws one from system entropy and **returns the value used**,
so a run stays reproducible after the fact without deciding up front that it is
worth reproducing. Both seeds are recorded in `config.json`; configs predating
this default to `data_seed=42`, which is what those runs actually used.

The HEP ensemble — rerun on the same inputs with fresh initializations and take
the variance as the model uncertainty — is a loop over `--seed` at fixed
`--data_seed`. Because the networks are Dense-only (no dropout or batch norm)
and Adam is deterministic, the two seeds together fully determine a run, up to
non-deterministic GPU reductions. Force those with
`XLA_FLAGS=--xla_gpu_deterministic_ops=true` if bitwise reproducibility is ever
needed; it costs throughput and is not needed for variance estimates.

Batch order comes from `jax.random`, inside the trace. `engine.train` seeds a
key from `data_seed` (carried on the splits and read by
`DeviceSplits.from_splits`) and splits it once per epoch; `train_indices(key,
...)` is a pure function of that key, so nothing can advance the sequence out
from under a caller. A second `engine.train` over the same `DatasetSplits`
therefore sees identical data with no rewind step needed.

`train_indices` also decides what an epoch skips. It permutes, then reshapes
into `(groups, n_disc_steps, batch_size)` — the generator updates once per
group, on the group's first batch. Whatever does not fill a whole group is
dropped, and since the permutation is redrawn each epoch it is a different
random tail every pass. A split too small for one group still trains: it
becomes a single group with every batch in it, and one generator update.

`deconvolve leakage-check` (in `src/deconvolve/workflows/leakage.py`) depends on this: both arms must
share `--seed` or initialization variance swamps the effect and the arms differ
even with no leakage. With it fixed, detector-level results are bit-identical
between the clean and poisoned arms.

`--poison` overwrites `z_true` with `--sentinel`, defaulting to
`POISON_SENTINEL` (-999). Any far-off-manifold value works, but it must not be
`TRUTH_SENTINEL`: a truth column set entirely to that is precisely what
`Populations.create` writes when there is no truth, so `has_truth` would call
the poisoned arm truthless and `require_truth()` would refuse the particle-level
comparison the check exists to make. `run_leakage_check` rejects that value up
front rather than after a full training run.
