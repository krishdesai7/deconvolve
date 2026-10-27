<!-- markdownlint-disable no-inline-html -->
# Timing & Profiling

<span style="font-variant: small-caps;">Deconvolve</span> can report how a run's wall-clock time divides between its phases: loading data, compiling, training, evaluation and so on. Timing is off by default, and when off it costs nothing.

---

## Enabling timing

Timing is on by default. To turn it off, set `DECONVOLVE_TIMING` in the environment:

```shell
DECONVOLVE_TIMING=0 deconvolve train --dataset jets
```

`0`, `false`, `no`, `off` and the empty string disable it; any other value, or leaving it unset, enables it. The variable is read when <span style="font-variant: small-caps;">Deconvolve</span> is imported, so it cannot be set in a configuration file.

With timing enabled, a command prints a table of its phases, with each phase's share of the total, and writes the same data to `artifacts/timings.json` in the run directory. The file is written even if the run fails, and the phase that raised the error is marked as failed.

---

## Phases of a training run

Indented phases are nested within the phase above them.

| Phase | Covers |
| :--- | :--- |
| `data` | Building or loading the dataset. The detail column records whether it was a cache hit, newly generated, or downloaded. |
| `train` | The whole training run: |
| &emsp;`transfer` | copying the splits to the accelerator, |
| &emsp;`compile` | compiling the training program, |
| &emsp;`epochs` | executing it, |
| &emsp;`select` | [selecting the best epoch](../theory/mmd.md). |
| `particle_mmd` | The particle-level MMD diagnostic, when the dataset has a known truth. |
| `save` | Saving the networks. |
| `load` | Reloading a saved run (`--load-run`), in place of `train` and `save`. |
| `plots` | Drawing the figures. Recorded in `timings_shared.json`, not `timings.json` (see below). |
| `evaluate` | Computing the [distance metrics](../user-guide/evaluation.md). |

Several commands can run over the same directory, for example `train`, then `train --load-run` to redraw the figures. Each command replaces its own phases in `timings.json` and leaves the others, and every row records which command (`train` or `load`) produced it.

`deconvolve report` shows `timings.json` in the report's Timing section.

---

## Work shared by all methods

The figures overlay <span style="font-variant: small-caps;">Deconvolve</span>, IBU and <span style="font-variant: small-caps;">OmniFold</span> together, so the time spent drawing them is not charged to any one method. `train`'s `plots` phase is written to `artifacts/timings_shared.json`, and so is `deconvolve plot`, which redraws the figures once the baselines have run. Its phases are `parse_config`, `data`, `load` and `plots`. The report shows this file as its own block, after the methods' blocks.

---

## The IBU baseline

`deconvolve baseline ibu` writes to its own file, `artifacts/timings_ibu.json`, for the same reasons as OmniFold below. Its phases are `parse_config`, `data`, `ibu` (unfolding every observable) and `evaluate` (scoring the weights on the test split).

---

## The OmniFold baseline

Timing also covers `deconvolve baseline omnifold`, which writes to a separate file, `artifacts/timings_omnifold.json`. A separate file keeps the baseline's cost distinct from the method's, and prevents its `data` and `evaluate` phases from replacing those of the training run.

Its phases are `parse_config`, `data`, `omnifold` and `evaluate`. Within `omnifold`, the time spent in the separate OmniFold process is broken down into `init`, `unfold`, one row for each step of each MultiFold iteration (`iter<n>_step1`, detector level, and `iter<n>_step2`, particle level), and `reweight`.

The OmniFold process measures these times whether or not timing is enabled, so even without `DECONVOLVE_TIMING` the command logs a one-line summary of the `init`, `unfold` and `reweight` times.

---

## Reading the numbers

- **`compile` depends on the cache.** When XLA's [compilation cache](caching-seeding.md#cache-directory) already holds the training program, `compile` takes a fraction of a second rather than several seconds. `timings.json` records whether the cache was already populated when the run started (`compile_cache_warm`); read `compile` together with it.
- **Totals.** `total_seconds` sums the top-level phases only, since nested phases are already included in their parent.
- **Results are unaffected.** To measure compilation separately, a timed run compiles and executes the training program in two explicit steps rather than one, and waits for asynchronous device work to finish at each phase boundary. This changes when the work is scheduled but not what is computed: a timed run produces the same training history as an untimed one.

---

## Timing new code

Additional phases can be timed with the `phase` context manager:

```python
from deconvolve.instrumentation import phase

with phase("my_step", detail="optional note for the table"):
    ...
```

When timing is disabled, `phase` returns a shared context manager that does nothing, so it can be left in place at no cost.
