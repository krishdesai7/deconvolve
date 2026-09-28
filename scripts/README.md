# Slurm scripts

## submit.zsh

End-to-end unfolding run: train -> IBU baseline -> replot with the baseline overlaid -> metrics. Defaults to all six jet observables; every stage runs in one sequential job on one GPU.

### Examples

```shell
    sbatch scripts/submit.zsh                       # all 6 jet observables
    sbatch scripts/submit.zsh --seed 7              # extra flags reach `deconvolve train`
    sbatch scripts/submit.zsh --var m --var w       # a subset
    sbatch scripts/submit.zsh --dataset gaussian --config params/2d_correlated.yaml
```

### Queue and resource allocation

The default config uses one GPU, not four, so the queue is `shared` rather than `regular`: nothing in `deconvolve train` shards across devices, it is a single jitted program on device 0, and the other three GPUs on a whole node would sit idle while JAX preallocated ~75% of each. `shared` charges only for the quarter of the node actually used, and small jobs backfill into gaps a whole-node request cannot reach.

_`-c32` is mandatory:_ the gpu_shared queue requires exactly 32 logical cores per GPU (a quarter of the node's 128) and rejects anything else. There is deliberately no `--mem` line: the scheduler converts a memory request into an equivalent core count and enforces the larger of the two, so `--mem=<M>GB` can silently become a request for a different core count and fail the job. Omitting it lets memory come out proportional to the cores (~54GB), which does not need rederiving when the node spec changes and is far more than an individual run needs: 1M events x 6 observables is ~105MB on device (z, x and y across all three splits) and well under a gigabyte on host.

`-Cgpu` sets the device requested to an 40GB A100, of which Perlmutter has ~1200 nodes; `-Cgpu&hbm80g` may be chosen if more memory is needed. This requests the 80GB part but from a pool of ~200, so it will cost queue time.

`-t00:15:00` sets the runtime to 15 minutes. With the default parameters, a full run takes about ~5 minutes end to end, dominated by npz loading and matplotlib, not by the GPU. This is a ~3x margin. This does however assume the jet cache is already warm. A cold cache pulls 3.3GB from Zenodo (Pythia26 1.55GB + Herwig 1.75GB), and the job may overrun the 15 minute limit. Warm it on a login node first with `uv run python -c "from deconvolve.data import load_jet_dataset; load_jet_dataset(1000)"`.

### Run script

- `set -e` is important here because this script involves running four distinct stages, and a failed train must not go on to run IBU against a previous run directory.
- `DECONVOLVE_CACHE_DIR` is inherited from the submitting shell (SLURM exports the environment by default).
- `_save_run` names the run directory for the UTC timestamp and returns it only to its Python caller, so a shell has to find it. Hence this script anchors on a marker file rather than "newest directory": `runs/` already holds older runs, and a mistake here would silently attach IBU to one of them.
- IBU unfolds the same populations for comparison. The reload pass after it puts the baseline into the figures: `workflows.train.run` reads ibu_weights.npz only if it exists when the plots are drawn, and on the training pass it does not exist yet.
- The reload pass does not update metrics.json (it only forces on a fresh train), so it must be recomputed explicitly.
- The script logs evidence that the persistent compilation cache did its job. A populated directory makes the _next_ run skip ~4.6s of XLA compilation; an empty one means `DECONVOLVE_CACHE_DIR` points somewhere unwritable and JAX only warned about it.

### RAN Training Arguments

The default `TRAIN_ARGS` is _prepended_ to the command line arguments. Since click keeps the last occurrence of a scalar option, anything on the command line overrides the default. The only exception is `--var`, which is explicitly set to `multiple=True` and accumulates, which is why the six-variable default comes from `deconvolve train` itself (an empty `--var` means `SUBSTRUCTURE_VARIABLES`) rather than being part of `TRAIN_ARGS`. The default `TRAIN_ARGS` is as follows:

- `-n1000000`: The Zenodo release holds ~1.6M jets per generator, and `load_jet_dataset` refuses $n_{\text{samples}} > n_{\text{avail}}$. 1M leaves headroom for the two generators' counts not matching exactly.
- `-l3 -u128`: Width over depth. Andreassen et al. use 3x100 ReLU on exactly these six observables; 3x128 follows that style staying shallow, which matters in a min-max game where depth destabilises the balance between g and d faster than width does.
- `-e100` / `--n-epochs 100`: The training loop runs a fixed trip count; there is no early-stopping patience flag. Epoch selection happens afterward, on the host, from the recorded MMD history; every epoch's state is retained so any of them can be restored. 100 epochs at 1M is ~13.6k generator updates.
- `-b1024`: Batch size.
- `--seed`: Seed for the random number generator. `train` draws one from system entropy and records it in config.json, so the run is reproducible after the fact without pinning it in advance.

## submit_precision.zsh

- Runs 72 paired float64/float32 precision-benchmark seeds and appends the results to f64.log/f32.log at the repo root.
- Run it on a login node with `zsh scripts/submit_precision.zsh`. Do NOT sbatch this file itself; it computes the seed list and submits the job.
- One node = 4 A100x40G on Perlmutter. 72 seeds x 2 dtypes = 144 independent runs, so this script throttles them across the node's 4 GPUs with `srun --exact` steps in a background loop. 144 runs/4 GPUs should finish in under 30 min; `-t00:45:00` is 1.5x that; a comfortable margin while still landing in a favorable queue window.
- The heredoc is the batch script; it is quoted (`EOF`) so shell expansion happens at job runtime, not now. Everything the job needs reaches it via `--export`. `-o` lives on the command line because SLURM does not expand shell vars in `#SBATCH` lines.
- Each run is one `srun` step pinned to a single GPU. `--exact` lets multiple steps share the allocation simultaneously (without it the first step grabs the whole node and the rest block). `--gpus-per-task=1` sets `CUDA_VISIBLE_DEVICES` per step so JAX in each run sees exactly one GPU, which matters, because JAX preallocates most of the memory on every device it can see. ~16 cores/56GB per A100 (4 GPUs, 64 cores, 256GB on a Perlmutter GPU node).
- Each run's log is captured so a crash in one run is easy to find and does not affect the others or corrupt the aggregate file.
- The script throttles to `GPUS_PER_NODE` concurrent steps; starts the next as soon as one frees a GPU.
- `|| true` ensures a run that crashes does not abort the batch under `set -e`. It just leaves no `SUMMARY` line in its log, and the collection step tolerates the gap.
- Once the batch is complete, the script pulls the `SUMMARY` line out of each per-run log, in seed order, and appends to f64.log/f32.log. A run that crashed leaves no `SUMMARY` line and is silently skipped here; check its own log file under `RUN_DIR` to diagnose.

## submit_omnifold.zsh

The OmniFold baseline against an **existing** run directory: unfold -> redraw the
figures with the OmniFold curve on them -> re-score -> rebuild the report. The
RAN training in that directory is read, never repeated.

### Examples

```shell
    sbatch scripts/submit_omnifold.zsh runs/2026-09-06T203848Z
    sbatch scripts/submit_omnifold.zsh runs/2026-09-06T203848Z --niter 5
```

Extra flags after the run directory reach `deconvolve baseline omnifold`.

### Why it is a separate job

OmniFold measured ~41 minutes on the shipped configuration (1.6M samples, twelve
observables, 3 iterations, 50 epochs), against `submit.zsh`'s entire 10-minute
wall clock. Folding it in would have meant sizing that job for the baseline
rather than for the method. This one asks for 75 minutes, ~1.5x the whole
pipeline.

### Two things that fail quietly without help

**`module load cudatoolkit/12.9`**, which the script does and then unloads. The
default Perlmutter environment leads `LD_LIBRARY_PATH` with CUDA 13.2 trees and
the `tensorflow` wheel is a CUDA 12 build; one library goes unreachable
(`libcusolver.so.11`) and TensorFlow silently registers no GPU, running on the
CPU with correct results and no error. The package warns when the worker reports
a CPU device --- watch the log for it.

The unload is an `EXIT` trap rather than zsh's `{ } always { }`, because
`always` does not run under `set -e`.

`module` itself has to be initialised first --- it is a shell function only a
login shell gets, and a batch job otherwise dies with
`command not found: module`. `scripts/_lmod.zsh` does that and every script
calling `module` sources it before the first call; `tests/test_scripts.py`
checks that they do.

**The worker's uv environment must already exist.** It is a PEP 723 script, so it
is not in `uv.lock`, and first use pulls ~3.5GB of CUDA wheels --- which a compute
node generally cannot do. Warm it on a login node:

```shell
    uv run --no-project src/ran/baselines/_omnifold_worker.py
```

## submit_hparam.zsh

One hyperparameter arm sweep, packed into a single multi-node job: three levels of one knob x eight initialization seeds = 24 runs, one A100 each, a single wave on 6 nodes. Run it on a login node with `zsh scripts/submit_hparam.zsh`. Do NOT sbatch this file itself; it computes the arm directory and submits the job. The collect step runs inline at the end of the same job.

Run on the login node:

```shell
    zsh scripts/submit_hparam.zsh                                  # lr_g at 3e-5 / 1e-4 / 3e-4
    FLAG=--lr-d LEVELS="1e-4 3e-4 1e-3" zsh scripts/submit_hparam.zsh
    NODES=3 zsh scripts/submit_hparam.zsh                          # half the GPUs, two waves
```

### Why every arm runs the same seeds

Pairing is most of the available statistical power, and it is free. Across six default-configuration jet runs differing only in initialization, the per-run spread is SD 7.16 on particle jet mass and SD 1.91 on the mean over the six particle-level observables. Unpaired, resolving a 2-point difference needs ~201 runs per arm on mass and ~14 on the aggregate. The same seed in two arms starts from the same weights, and most of that variance cancels in the within-pair difference.

Every sweep in `benchmarks/README.md` under "What was ruled out" deconvolve **one** run per arm, each at a _different_ seed. `--data-seed 42` is pinned for the same reason: it is the second nuisance axis, and it has to be held rather than sampled.

### `--run-dir`

Without it every run lands on a second-resolution UTC timestamp under `runs/`, created with `exist_ok=True`. Twenty-four runs of identical shape launched together finish inside the same second, and the losers were overwritten with no error and no way to tell afterwards which arm had gone missing. `--run-dir` names each run so its arm is recoverable, and `workflows.train._new_run_dir` refuses a directory that already holds a `config.json` rather than landing on top of it. An _empty_ directory is accepted, because this script creates one per run to
redirect `train.log` into before training starts.

### Wall clock

`--no-plots`: the figures are a large share of a short run's wall clock and no part of scoring one. Metrics still run, and the collect step reads them. Training itself is ~15s per run at these parameters (`benchmarks/boundary.py`, A100); `-t01:00:00` is margin for the npz load, not for the GPU, since 24 processes read the 1M-event jet cache at once.

### Warm up

If the jet cache has not been populated, a cold cache pulls 3.3GB from Zenodo inside the job, 24 times over. Warm it on a login node first with `uv run python -c "from deconvolve.data import load_jet_dataset; load_jet_dataset(n_samples=1000)"`.

### Options

- `FLAG`, `LEVELS`, `SEEDS`: The arm axis and its levels, and the seeds swept at each. The collect step reads the knob off the saved configs, so changing `FLAG` to another flag needs no change downstream. Three levels around the default, so the read is a trend across levels, and at n=8 against SD 1.9 the argmax of two arms is mostly luck.
- `NODES`: n=8 x 3 levels = 24 runs. `NODES=6` -> 24 A100s -> a single wave, so the wall clock is one run's. Lower `NODES` to use fewer GPUs at the cost of extra waves.
- `TRAIN_ARGS`, `PROJECT_DIR`, `ARM_DIR`: passed straight through to the job; see the script for their defaults.
- `JOB`: The submitted job's ID, echoed for the logs listed above.

## submit_uncertainty.zsh

The bootstrap x seed variance design: `B` bootstrap datasets crossed with `S` initialization seeds, one `deconvolve uncertainty run` per cell, then one `deconvolve uncertainty collect` over the grid. The statistics — why a grid rather than two one-dimensional sweeps, why `data_seed` is held fixed, what the correction and the closure floor are for — are argued in `src/ran/uncertainty/README.md`; this section is only about the allocation.

### Wall clock

Written out rather than guessed at, since the last two launchers here were sized by feel and were wrong by a factor of six. A cell is one ordinary training run (~3 min at `-n1000000 -l3 -u128`) plus the jet cache read; call it 4 minutes. `B*S` cells over `NODES*4` GPUs is `ceil(B*S / (NODES*4))` waves. The default 8x8 on two nodes is 8 waves and about 35 minutes, inside the `-t01:00:00` requested. Raising `NODES` cuts waves and buys queue time; it does not make a cell faster, and a design that finishes in one wave on six nodes has spent an hour of queue to save twenty minutes of run.

### Warm up

Same as `submit_hparam.zsh`, and it matters more here: a cold cache would pull 3.3GB from Zenodo `B*S` times over. `uv run python -c "from deconvolve.data import load_jet_dataset; load_jet_dataset(n_samples=1000)"` on a login node first.

### Which grid to run

`B=8 S=8` is the decomposition: both axes need at least two levels for the mean squares to have degrees of freedom, and eight apiece puts the components at a useful precision. `B=100 S=2` trades the seed axis for the bootstrap one, which is the shape for the bin-to-bin covariance — a `K x K` matrix from `B` replicates wants `B` well above `K`, and at 20 bins `B=50` already gave the right aggregate structure (lag correlations, effective rank) but let individual off-diagonal entries move by up to 0.39 between reruns; the published numbers use `B=100`.

### Options

- `B`: Bootstrap datasets. Minimum 2.
- `S`: Initialization seeds. Minimum 2; `run` numbers cells seed-major, so a design cut short after `k*S` cells is still a complete grid over the datasets that finished.
- `N_EVAL`: Size of the common evaluation set held out before resampling. Every cell is read on exactly these events; the default 100k costs 400KB per cell on disk.
- `N_BINS`: Bins for the covariance, passed to `collect`. Equal-occupancy, so a discrete observable can come back with fewer.
- `RUN_ARGS`: Arguments passed to `deconvolve uncertainty freeze` on the login node, before any cell exists — not to the cells themselves, which read the frozen `design.json` instead. Defaults to the paper's configuration on purpose — a design run at cheaper settings is a variance budget for a model nobody is publishing.
- `NODES`, `PROJECT_DIR`, `DESIGN_DIR`, `JOB`: as in `submit_hparam.zsh`.

# Docs

## gen_api_docs.py

Writes the API reference under `docs/api/`: one page per public module under `src/deconvolve/`, an index per package, and the `SUMMARY.md` that literate-nav turns into that section's nav. Zensical has no hook for generated pages (it does not support `mkdocs-gen-files`), so `just doc-build` and `just doc-serve` run this first. The output is wiped and rewritten each time and is gitignored; never edit it by hand. Page content comes from docstrings, rendered by mkdocstrings at build time.
