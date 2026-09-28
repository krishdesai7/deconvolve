# OmniFold

`deconvolve baseline omnifold` is the second comparison baseline, and the only part of
this repository that does not run in this repository's environment. Three facts
make that necessary: OmniFold needs TensorFlow, the project environment must
never hold TensorFlow, and Keras binds its backend once per interpreter. All
three are **intra-interpreter** constraints, so all three dissolve at a process
boundary. (TensorFlow also publishes no wheels for 3.14, which is why the
worker pins `==3.13.*`; that is a fact about the worker's environment, not
about the project floor, which is `>=3.12`.)

`src/deconvolve/baselines/_omnifold_worker.py` carries a PEP 723 header pinning
`requires-python = "==3.13.*"` plus `omnifold` and `tensorflow`, and
`uv run --no-project` provisions exactly that, in an interpreter that cannot
import `ran`. The two halves exchange one `.npz` file. `--no-project` is
load-bearing: without it uv resolves the script against this project and runs
it in the project environment — the one environment that must never hold
TensorFlow, and whose interpreter is whatever the checkout is pinned to
(3.14 via `.python-version`), not the worker's `==3.13.*`. Lowering the floor
to `>=3.12` did not change this: a floor is not a pin, and it is the resolved
environment rather than the floor that the script would land in.

The worker is inside the package but is not part of it. Nothing imports it and
nothing may: its module-level `KERAS_BACKEND=tensorflow` would race the
package's `jax` pin. Three mechanisms keep it that way, all enforced rather than
documented — `pyproject.toml` pins `[tool.ruff.per-file-target-version]` for
`**/*_worker.py` to `py313`, pyrefly excludes the same glob, and
`tests/test_omnifold.py::TestQuarantine` asserts the module is absent from
`sys.modules` and that TensorFlow is not importable at all.

**The ruff pin is not hygiene, though it is currently not armed.** Ruff infers
its target from `requires-python`. At the old `>=3.14` floor it inferred
`py314` and its formatter rewrote `except (A, B):` into PEP 758's
unparenthesized form — a `SyntaxError` on 3.13, which killed the worker at
import the first time it was formatted. At today's `>=3.12` floor ruff infers
`py312`, so nothing in the repository is formatted into PEP 758 syntax and the
trap is disarmed at the source. The `py313` pin stays because it states the
worker's real target exactly, and is what keeps the trap disarmed for the
worker if the project floor ever rises again. A test compiles the worker to
catch a regression.

**`uv` must be on `PATH` at runtime**, since it is what provisions the worker.
Its absence is translated into a readable message rather than a
`FileNotFoundError` from inside `subprocess`, because the fix is an install.

**The worker runs under `PYTHONSAFEPATH=1`, and must.** A script's own directory
goes on `sys.path[0]`, and the worker's directory is `src/deconvolve/baselines/` —
which contains `omnifold.py`. So the worker's
`from omnifold import MLP, DataLoader, MultiFold` resolved to the _host half_
rather than to the installed package, and died on its `from .. import timing`
with "attempted relative import with no known parent package": an error naming
neither the collision nor the file that caused it. `PYTHONSAFEPATH` stops the
interpreter prepending that directory, which is exactly the shadowing and
nothing else — the worker imports nothing local, so it loses nothing. Renaming
this module would also have worked, at the cost of `deconvolve.baselines.omnifold` no
longer being named after the thing it runs.
`TestTheWorkerDoesNotImportThisPackage` reproduces the collision with a poisoned
sibling.

**The worker environment is not in `uv.lock`.** uv resolves the PEP 723 header
on first use, which needs outbound network, and compute nodes generally have
none. Warm it on a login node, the way the jet cache is warmed:

```bash
uv run --no-project src/deconvolve/baselines/_omnifold_worker.py
```

## It runs on the CPU, silently, without a CUDA 12 toolkit

**On Perlmutter `module load cudatoolkit/12.9` is mandatory.** The default
environment leads `LD_LIBRARY_PATH` with four CUDA **13.2** trees and the
`tensorflow` wheel is a CUDA **12** build; exactly one library goes unreachable,
`libcusolver.so.11`, and one is enough for TF to skip registering every GPU. It
then runs on the CPU and **raises nothing** — the weights come back correct,
tens of times slower, and the baseline looks like it worked.

So the worker reports the device it used and `_warn_if_on_cpu` warns when it was
not a GPU. That warning is the only signal this failure produces; do not silence
it. `benchmarks/gpu_coexistence.py` measures the whole thing and its README
section records the numbers.

What that benchmark also settled: a TensorFlow subprocess gets the GPU **even
with JAX's default 75% preallocation held by the parent**. The worker peaks at
1.07GB against the 9.4GB that survives, so no `XLA_PYTHON_CLIENT_*` tuning is
needed. That was the risk worth checking before any of this was written, and it
did not bind.

## Its own job, not a step in `submit.zsh`

`scripts/submit.zsh` does not run OmniFold. That job asks for
`--time=00:15:00`, and OmniFold alone measured **~41 minutes** on the shipped
configuration — 1.6M samples, twelve observables, `niter=3`, 50 epochs — so
it would not fit in what is left after RAN trains. It gets
`scripts/submit_omnifold.zsh` instead, which takes an existing run directory and
asks for 75 minutes:

```zsh
sbatch scripts/submit_omnifold.zsh runs/<timestamp>Z
```

That script loads `cudatoolkit/12.9`, runs the baseline, unloads it, redraws the
figures, re-scores and rebuilds the report. **The module unload is an EXIT trap,
not zsh's `{ } always { }`** — `always` does not run under `set -e`, which
ERR_EXIT leaves before reaching, so a failed unfolding would have left the CUDA
12 toolkit loaded over whatever deconvolve next in the allocation. Measured, not
assumed.

**`module` is not available in a batch script until it is initialised**, which
is what `scripts/_lmod.zsh` does and every script calling `module` sources
first. It is a shell function Lmod defines in a startup file that only an
interactive or login shell reads; a batch script is neither, so it reads
`/etc/zshenv` and `~/.zshenv` and nothing else, and the first `module load`
dies with `command not found: module` — under `set -e`, taking the job with
it. The asymmetry is confusing precisely because `whence module` at a login
prompt finds it; `zsh -c 'whence module'` is what the job actually sees.
`submit.zsh` had the same latent bug in its `module load texlive`, where it cost
only the PDF because it sits last. `tests/test_scripts.py` asserts the ordering,
and `zsh -n` parses every script — a shell script is otherwise covered by
nothing here.

## Getting OmniFold onto the figures

Presence is the mechanism, and it is the same one IBU has always used.
`_load_baseline_weights` returns one `BaselineOverlay` per `*_weights.npz` that
exists in `artifacts/` when the figures are drawn, so:

```zsh
deconvolve baseline omnifold runs/<timestamp>Z   # writes omnifold_weights.npz
deconvolve train --load-run runs/<timestamp>Z              # reloads, redraws with it
```

`--load-run` reloads the saved generator rather than training, so the redraw is
cheap and the run is untouched. There is no separate "add OmniFold to the plots"
command because there is nothing for it to do that `--load-run` does not.

`evaluation.plotting.BaselineOverlay` is what made a second baseline cheap. The overlay
used to be a bare `ibu_weights: list[EventArray] | None` threaded through six
functions; with two baselines that would have become two parameters in six
signatures. It carries one weight vector **per dimension**, because IBU unfolds
each observable separately and its weights genuinely differ between them;
`from_shared` repeats a single vector across the dimensions, which is what
OmniFold needs — it reweights events, not observables. OmniFold draws crimson
dash-dot with triangles against IBU's green dotted squares, distinguished by
linestyle as well as colour so the panels survive greyscale printing.

The report's tables carry the third arm too: `render` reads
`metrics_omnifold.json` when it exists, and fills the two OmniFold columns with
dashes when it does not, because the template fixes the column count. Columns
run Sim, IBU, OmniFold, RAN — the method under test last, where the eye lands,
behind what it is being compared against.

Eight columns do not fit at the default column padding. The six tables overran
the text block by ~24pt, which `pdflatex` reports as an overfull hbox and
_still compiles_ — so nothing failed and the numbers simply deconvolve off the page.
They now set `\tabcolsep` to 4pt and drop the method name from each improvement
heading (`impr. (\%)`, unambiguous because it sits beside its method's column).
`_TABLE_COLUMNS` is the one place the count lives, and a test asserts it against
the template's own `tabular` specification — the two have no other connection,
and had already drifted once.
