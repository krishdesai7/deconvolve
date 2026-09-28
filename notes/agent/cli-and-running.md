# Running

The package installs a `ran` console script (`[project.scripts]` →
`deconvolve.cli:app`), which is the canonical entry point. In a checkout, prefix it
with `uv run` to use the project environment without activating it
(`uv run deconvolve train ...`); `python -m ran` still works via `__main__.py`. Shell
completion comes from `deconvolve --install-completion` and needs the script name, so
it does not work through `python -m`.

One Typer command tree. Flags are kebab-case; subcommands are
`train`, `evaluate`, `report`, `leakage-check`, `baseline {ibu,omnifold}`,
`uncertainty {freeze,run,collect}`, `config show`. `--log-level` is global and
goes before the subcommand.

Every layerable option above also resolves through the five-layer config
stack — code default, global `deconvolve.toml`, project `deconvolve.toml`/`[tool.deconvolve]`,
`DECONVOLVE_<COMMAND>_<OPTION>` environment, then the command line — documented in full in
[configuration.md](configuration.md). `deconvolve config show` prints the resolved
value of every setting next to the file, variable or default it came from;
`deconvolve config show train` scopes the listing to one command, and `--json` emits
the same content for scripting. `deconvolve uncertainty freeze DIR
[options]` resolves that stack once and writes `DIR/design.json`, which every
`deconvolve uncertainty run` cell in the array then reads instead of the ordinary
layers — see "The freeze path" in [configuration.md](configuration.md) for why
`uncertainty run` alone does not participate in the stack.

Every knob that changes a run is reachable from `deconvolve train` and recorded in
`config.json` — architecture (`-u`, `-l`), optimization (`--lr-g`, `--lr-d`,
`-k`/`--n-disc-steps`, `--lambda-dispersion`) and the loop (`-e`/`--n-epochs`)
and both seeds. `--lr-g` defaults to 3e-5, measured rather than chosen: see
"What tuning actually found" in `benchmarks/README.md`. `--lambda-dispersion`
penalizes the variance of `g`'s normalized MC weights and defaults to **0.015**,
also measured: it is the same axis `lr_g` acts on indirectly, pushed directly,
and "The dispersion penalty: the trade made explicit" in `benchmarks/README.md`
is the paired 12-observable sweep that picked the value. A run left at the
default has the penalty **on**, which is the setting any baseline should be
compared against.

Model selection is not a flag: it is fixed to the detector-level MMD argmin
(see [training-loop.md](training-loop.md)). `--no-plots` skips the figures,
which are a large share of a short run's wall clock and no part of scoring
one; metrics still run, and `--load-run` on the same directory draws them
afterwards.

`--run-dir` names where a run saves, and a sweep needs it. The default is a UTC
timestamp at second resolution, which several runs of identical shape launched
together will collide on; `_new_run_dir` refuses a directory already holding a
`config.json` and disambiguates the default rather than overwriting silently.
An empty directory is accepted, so a launcher can create one to redirect logs
into before training starts.

```bash
deconvolve train --config params/1d_default.yaml                     # 1D uncorrelated
deconvolve train --config params/1d_default.yaml --seed 7            # reproducible init (see seeding.md)
deconvolve train --config params/2d_correlated.yaml                  # 2D with covariance
deconvolve train --dataset jets                                      # train on all twelve jet variables
deconvolve train --dataset jets --var m --var w                      # a subset of jet variables
deconvolve train --dataset jets --lr-g 3e-4 -k 2 --no-plots          # tuning: see benchmarks/README.md
deconvolve train --dataset jets --seed 3 --run-dir runs/hp_x/lrg1e-4_seed03  # one arm of a sweep
deconvolve train --load-run runs/2026-03-14T061023Z                  # reload a saved run
deconvolve evaluate runs                                             # compute metrics for all runs
deconvolve evaluate runs/2026-...                                    # single run
deconvolve baseline ibu runs/2026-...                                # IBU comparison
deconvolve baseline omnifold runs/2026-...                           # OmniFold (see omnifold.md)
deconvolve report runs/2026-...                                      # PDF dossier (see reporting.md)
deconvolve leakage-check --clean                                     # z_true leakage sanity check
deconvolve config show                                                # resolved settings and their origins
deconvolve uncertainty freeze runs/unc_x -B 8 -S 8                    # freeze a design before submitting its array
deconvolve --log-level DEBUG train --config params/1d_default.yaml
sbatch scripts/submit.zsh                                      # end-to-end 12-var jet run
sbatch scripts/submit.zsh --dataset gaussian --config params/2d_correlated.yaml
bash scripts/submit_hparam.zsh                                 # hyperparameter arms, 3 levels x 8 seeds
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_...     # paired comparison of the arms
```

Development recipes go through `just` (`just` alone lists them):

```bash
just validate   # format, lint, typecheck, complexity, tests -- all read-only
just ci         # validate, then audit locked dependencies for known vulnerabilities
just lint-fix   # safe lint fixes, then format
just test -k train   # extra args forward to pytest
just test-fast  # the same suite minus `slow`, for a check mid-work
```

`just test-fast` deselects `@pytest.mark.slow`; `just test`, `just validate`
and CI run the whole suite. The split exists because the cost is wildly
uneven: 37 of the 601 cases are ~55s of a ~76s run, and the other ~500 are
~23s together, so a quick pass costs a third of the time and gives up a
fixed, known list rather than a random one.

The `slow` marker goes on a test for a reason, not a measured duration — a
stopwatch threshold rots as the hardware and the suite move. A test
is `slow` if it **runs a training program** (one `engine.train()` call is
~0.5s even with the XLA cache warm), **shells out to `pdflatex`**, or
**averages many
random draws to measure a statistical property** (`tests/test_mmd_floor.py`).
A test against the piece directly costs a few milliseconds and needs no
marker; a full run costs a hundred times that and does.

Nothing in the suite runs OmniFold: `tests/test_omnifold.py` substitutes a
stub worker over the same `.npz` contract and tests the seam instead — see
[omnifold.md](omnifold.md).

## `scripts/submit.zsh`

The full pipeline rather than a bare `deconvolve train`: it trains, runs the IBU
baseline on the same run directory, reloads once so the figures come back out
with the baseline overlaid (`workflows.train.run` picks up `ibu_weights.npz` only if
it exists when the plots are drawn), recomputes metrics, then
`module load texlive` and `deconvolve report` to leave a PDF at the top of the run
directory. It defaults to the full **twelve**-observable jet run at
`-n 1600000 -l 3 -u 128`, with `DECONVOLVE_TIMING=1` exported so the run reports
where its wall clock went. Extra flags reach `deconvolve train`; those defaults are
prepended, and click keeps the last occurrence of a scalar option, so
anything on the command line still wins.

The script does not name the twelve observables as `--var` flags, and instead
lets `load_jet_dataset`'s own default stand: `--var` is repeatable, so click
*appends* rather than replacing, and naming all twelve would turn
`sbatch scripts/submit.zsh --var m` into thirteen names with a duplicate, which
`load_jet_dataset` rejects. Left off, a subset stays selectable from the
command line.

`-n` is resolved against the cache rather than hardcoded: `load_jet_dataset`
raises rather than truncating when asked for more than is on disk, which at
1.6M — a number chosen to sit at the edge of the release — would burn the whole
allocation on an immediate `ValueError`. The script reads the real count off
the cache and clamps, reporting when it does; a cold cache falls through to
the requested number.

There is no `--patience` flag: `n_epochs` (100 by default) is a fixed `scan`
trip count, and the best state is restored **always**, selected on the host
after the run rather than during it (see [training-loop.md](training-loop.md)).
At the shipped 1.6M samples and 100 epochs that is ~21.8k generator and ~109k
discriminator updates (the train split is 70% of `n_samples`, batched at 1024
and grouped by `-k`).

The job asks for **`--qos=shared --time=00:15:00`** on a quarter node
(`--gpus=1 --cpus-per-task=32`, and deliberately no `--mem`): nothing shards
across devices, so three of a node's four A100s would sit idle, and `shared`
charges for the quarter actually used while backfilling into gaps a
whole-node request cannot reach. `-c 32` is mandatory: the `gpu_shared` queue
requires exactly 32 logical cores per GPU. Do not add `--mem` — the scheduler
converts a memory request into an equivalent core count and enforces the
larger of the two, so `--mem=64G` reads as a 38-core request and the queue
rejects it. Omitted, memory comes out proportional to the cores at ~54GB,
against a run that needs ~105MB on device and under a gigabyte on host.

`-C gpu` is the 40GB A100 (~1200 nodes). `-C gpu&hbm80g` gets the 80GB part
from a pool of ~200 — more queue time for headroom this job does not use.

The wall clock is sized from `benchmarks/boundary.py` on an A100 (4.6s
compile, 0.034s per epoch at the same 500k x 6D shape): training is ~15s at
the parameters above and the pipeline is minutes, dominated by npz loading
and matplotlib. **Warm the jet cache on a login node first** — a cold cache
pulls 3.3GB from Zenodo (Pythia26 1.55GB + Herwig 1.75GB) inside the job and
will exceed the `debug` ceiling:

```bash
uv run python -c "from deconvolve.data import load_jet_dataset; load_jet_dataset(n_samples=1000)"
```

The Zenodo release holds ~1.6M jets per generator, and `load_jet_dataset`
raises if `n_samples` exceeds what is on disk, which is why the script clamps
to what the cache actually holds instead of asserting a fixed number.
