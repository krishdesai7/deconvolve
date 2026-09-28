# Timing

`DECONVOLVE_TIMING=1` makes a run report where its wall clock went; unset, the layer
is a genuine no-op — `phase()` hands back one shared do-nothing context
manager, so a boundary costs no `perf_counter` call and no allocation. That
matters because the timers sit inside `workflows.train.run` and `engine.train`, which a
sweep crosses a few hundred times.

```bash
DECONVOLVE_TIMING=1 deconvolve train --dataset jets -e 100
```

Output is a Rich table on stderr plus `artifacts/timings.json` in the run
directory, written from a `finally` so a run that fell over still reports —
the phase that raised is recorded, marked `failed`, with the time it burned
before it did.

**Phases merge by name across passes, and each carries a `pass` field.**
`scripts/submit.zsh` invokes the package three times over one run directory,
and each write used to truncate the file: the final `deconvolve evaluate` pass left a
`timings.json` holding `evaluate` alone, with the training block — the only
part anyone wants — gone. A pass now replaces its own same-named phases and
leaves the rest, so the file accumulates `train`/`load`/`plots`/`evaluate`
together and `pass` says which invocation produced each row. `--run-dir` and `--load-run` name a directory up front so a crash there
still gets a file; a fresh run under the default timestamp has nowhere to write
until `_save_run` exists, and the table on stderr is then all there is.

The phases, nested ones indented under their parent:

| Phase          | Covers                                                                                                                                                                                                           |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`         | Building or loading the splits. The `Detail` column says which branch it took — `cache hit`, `generated`, `downloaded from Zenodo` — filled in by the loaders, which know, via `timing.note(..., to="data")` |
| `train`        | The whole of `engine.train()`                                                                                                                                                                                           |
| ` transfer`    | `DeviceSplits.from_splits`, the one host->device copy of a run                                                                                                                                                   |
| ` compile`     | XLA compiling the fused whole-run program                                                                                                                                                                        |
| ` epochs`      | Executing it                                                                                                                                                                                                     |
| ` select`      | `_select_by_mmd`, host-side, after the loop                                                                                                                                                                      |
| `particle_mmd` | The particle-level diagnostic curve in `_finish_run`                                                                                                                                                             |
| `save`         | `_save_run`: two `.keras` files plus `EpochParams`                                                                                                                                                               |
| `load`         | `_load_artifacts`, on the `--load-run` path instead of `train`/`save`                                                                                                                                            |
| `plots`        | `_draw_figures`; near-zero under `--no-plots`                                                                                                                                                                    |
| `evaluate`     | `evaluate_run`                                                                                                                                                                                                   |

`deconvolve baseline omnifold` writes its own `artifacts/timings_omnifold.json`
rather than merging into `timings.json`, and that is not tidiness. **`write`
merges by phase name alone, not by `(pass, name)`** — which is right for the
passes of one pipeline over one run, where `load` legitimately replaces
`train`'s `plots` row, and wrong for a different program over the same
directory. The baseline has phases called `data` and `evaluate` of its own, so
writing them into the shared file would silently destroy the training pass's.
Separate also keeps the baseline's cost separable from the method's, which is
the comparison the numbers exist for.

Its phases are `parse_config`, `data`, `omnifold` and `evaluate`, with the
worker's own breakdown nested under `omnifold`: `init`, `unfold`, a
`iter<n>_step<1|2>` row per MultiFold iteration, then `reweight`. Those come
back as numbers across the `.npz` rather than as blocks to wrap, so they enter
through **`timing.record`** — the one way into the tree for a phase this
process did not time itself. The per-iteration split is the useful part:
MultiFold's two steps are not symmetric (step 1 reweights at detector level,
step 2 at particle level), so a single `unfold` total cannot say which half a
long run spent its time in. OmniFold exposes no timing of its own, so the
worker wraps `RunStep1`/`RunStep2`; the wrapping is guarded, and a rename
inside OmniFold costs the breakdown rather than the baseline.

The iteration rows sit at the same depth as `unfold` rather than under it.
`_ordered` reconstructs a top-level phase's children by position and does not
recurse, so a genuine grandchild renders under whichever sibling precedes it
and its parent row prints after it. One level is what the format supports.

`timings.json` is flat, with a `depth` field rather than nested objects, so a
sweep can join it against `config.json` without walking a tree. `total_seconds`
sums the **top-level** phases only: a nested one is already inside its parent
and adding it double-counts.

Three things to know before acting on a number.

**`compile` is meaningless without knowing whether the cache was warm.** A warm
XLA persistent cache (see [caching.md](caching.md)) turns 4.6s into a fraction of a second, and
that is the common case — so a `compile` row read in isolation says compile is
free and points optimization at the wrong component. `timings.json` records
`compile_cache_warm`, sampled before the run could fill the cache.

**Timing changes how the fused path is compiled, not what it computes.**
Exposing the compile boundary needs `lower().compile()` and a call to the
compiled object, because an ordinary `jax.jit` call does both at once and shows
no seam between them. It is the same executable and the same persistent cache;
`tests/test_timing.py::TestTrainIntegration::test_timed_run_matches_an_untimed_one`
is what says so, asserting a timed run's history is bit-identical to an untimed
one's. The split is gated on `DECONVOLVE_TIMING`, so the default path stays the single
call `TestFusion` pins.

**JAX is async, so phase boundaries block.** A timer stopped before the arrays
are ready charges this phase's time to whichever phase runs next, so `transfer`
and `epochs` call `jax.block_until_ready` inside the clock. This shifts when the
wait happens, never what is computed — but a timed run and an untimed one are
not the same schedule.
