# Data Representations

The same events carry three shapes. Two are defined in
`src/deconvolve/coretypes/events.py` and sit at opposite ends of the host pipeline; the
third, in `src/deconvolve/data/device.py`, is what training actually runs on.

`Populations` is the physics form and holds three things: `mc`, an `Events` pair
of generated particle level (`mc.z`) and simulated detector level (`mc.x`)
aligned per event; `data`, the measurement; and `truth`, the particle-level
answer key. `truth` is deliberately _not_ inside an `Events` alongside `data`,
so a function handed the simulation cannot reach `z_true` — the constraint
in the [Critical Constraint](../../CLAUDE.md#critical-constraint) section,
expressed in the type rather than by convention.

A real measurement has no `truth`, so `Populations.create` fills it with
`TRUTH_SENTINEL` (-2^15, exactly representable in every IEEE binary format,
which lets `has_truth` compare by equality at any precision — see
[precision.md](precision.md)); `has_truth` reports the difference and
`require_truth()` refuses, which is how the particle-level metrics read the
answer key. The stand-in has to be an ordinary number rather than NaN:
`interleave` puts `truth` in the nature rows of `z`, where `normalize_weights`
annihilates the generator's output by multiplying by `1 - y = 0`. That masks
a number but not a NaN, which would otherwise reach every weight in the batch
and every gradient.

`ZXY` is the transport form: an `Events` pair plus a label, `y = 1` for nature
and `y = 0` for MC. It shuffles and splits.

Sources build a `Populations` and call `interleave()`; analysis calls
`partition()` on the way back out. Only `partition(interleave(...))` is
lossless — the reverse discards the shuffled row order, and weight vectors are
indexed against a `Populations`, so nothing should round-trip.
`DatasetSplits.select(Split.TRAIN | Split.VAL)` draws one `ZXY` from any
combination of splits.

`TrainSplit`/`EvalSplit` are the training form, and unlike the other two they
are device-resident `jax.Array`s registered as JAX pytrees.
`DeviceSplits.from_splits(splits)` is the single host→device transfer of a run;
after it, no batch ever crosses the boundary again. The train split stays flat
and is gathered by index inside the scan, so XLA fuses the gather into the first
`Dense`; the eval splits are pre-batched to a uniform shape and padded, with a
`mask` field that is 1 for a real row and 0 for filler. The mask enters every
sum in `normalize_weights` and `bce_sums`, so a padded batch reports exactly the
number an unpadded one would.

Keeping the first two forms on host NumPy is deliberate: they feed SciPy,
Matplotlib, npz I/O and the IBU baseline, none of which want device arrays.
Only `src/deconvolve/data/device.py` is device-resident.

## Jet Column Order

For `--dataset jets`, the list of observables is an **ordering**, carried as a
`tuple[str, ...]` and never as a set. `load_jet_dataset` fills column `i` from
`variables[i]`; `_save_run` records that order in `config.json`; and
`deconvolve evaluate` and the baselines read the recorded list back **as a list**,
in order.

This was a `frozenset`, and it produced silently wrong physics. A frozenset's
iteration order depends on the per-process randomized hashes of the strings in
it, so `deconvolve train` built its columns in one order and recorded it, then each
later process rebuilt the same dataset in a _different_ order and labelled it
with the recorded one. Every jet metric came back under the wrong observable
name, and — because the generator was trained on one column order and evaluated
against another — the reload and `deconvolve evaluate` passes fed it permuted features
and reported large negative improvements. See `tests/test_jets.py`.

Two rules follow, both enforced rather than documented: `load_jet_dataset` raises
`TypeError` on a set (also on duplicates and unknown names), and
`cli._canonical_variables` sorts `--var` into `SUBSTRUCTURE_VARIABLES` order,
so `--var w --var m` and `--var m --var w` describe the same run — same
columns, same cache key, same `config.json`.
