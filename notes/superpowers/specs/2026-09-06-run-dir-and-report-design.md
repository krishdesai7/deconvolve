# Run directory legibility and `ran report`

**Date:** 2026-09-06
**Status:** Approved, ready for implementation planning

## Problem

A finished run directory is the artifact a person reads to understand what
happened. Today it is not readable:

- Binaries (43 MB across five files) sit interleaved with the eight files a
  human actually opens, breaking the reading frame.
- The two metrics files hold ~110 numbers each at sixteen significant figures
  of JSON. Nobody reads them. They also disagree on key order --- `metrics.json`
  writes every detector entry then every particle entry, `metrics_ibu.json`
  interleaves the two per variable --- which is the shape of bug that surfaces
  the first time someone zips them positionally.
- `timings.json` is missing the entire training block on every pipeline run,
  because `scripts/submit.sh` makes three passes over one directory and the
  third overwrites the file written by the first.
- Three of the four figures have defects that range from cosmetic (clipped
  y-labels) to actively misleading (`losses.pdf` autoscaling a 0.4% band into
  an apparent divergence; `selection.pdf` overlaying three noisy series and
  covering a third of the axes with its legend).

The output of this work is a run directory whose top level is entirely
human-readable, plus a single `report.pdf` that explains a run without the
reader opening anything else.

## Non-goals

- **Replacing the JSON files.** `metrics.json`, `config.json` and `timings.json`
  are machine interfaces consumed by `benchmarks/hparam_collect.py`,
  `benchmarks/sliced.py` and the sweep collector. The report is a read-only
  consumer that sits alongside them.
- **Backward compatibility with existing run directories.** Explicitly waived
  by the project owner: the repo is unpublished, the ~40 existing run dirs are
  not worth preserving, and there is no migration command. Readers assume the
  new layout unconditionally.
- **Changing any physics.** No metric, loss, selection criterion or training
  behavior changes. The one substantive finding of the review --- the loss
  curves in `losses.pdf` --- was traced and found correct; see Appendix A.

## A. Run directory layout

### Structure

```text
runs/<timestamp>/
├── report.pdf            The dossier: what happened
├── config.json           What was run
└── artifacts/            Everything the report was built from
    ├── report.tex
    ├── metrics.json
    ├── metrics_ibu.json
    ├── timings.json
    ├── detector_level.pdf
    ├── particle_level.pdf
    ├── losses.pdf
    ├── selection.pdf
    ├── generator.keras
    ├── discriminator.keras
    ├── history.npz
    ├── params.npz
    └── ibu_weights.npz
```

Two files at the root, everything else one level down, flat.

**Why `config.json` stays at the root.** It is the one file where
human-readability and machine-readability coincide, and `_new_run_dir`
(`workflow.py:113`) claims a directory by its presence -- moving it would make
the claim check reach into a subdirectory that may not exist yet, and would
complicate the "an empty directory is accepted, so a launcher can create one to
redirect logs into" rule for no gain. It is also what a run that crashed before
producing a report still has to say for itself.

**Why `artifacts/` is flat.** Nesting `artifacts/figures/` and
`artifacts/binaries/` adds a second level of digging to protect a reading frame
nobody is using once they are inside the supporting material. Twelve files at
one level is fine.

**Why `artifacts` and not `binaries`.** The first name describes a role, the
second a file format. Once `report.pdf` exists, the figures and the metrics
JSONs are supporting material in exactly the sense the model checkpoints are.

### Implementation

Add to `src/ran/rantypes/constants.py`, beside `CACHE_DIR`:

```python
ARTIFACTS_DIR: Final[str] = "artifacts"

def artifacts_dir(run_dir: Path, /) -> Path:
    """The run's supporting-material subdirectory, created on demand."""
```

`artifacts_dir` creates the directory (`mkdir(parents=True, exist_ok=True)`) so
that writers need no separate setup step and readers get a path that exists.

### Call sites

Writers:

| File               | Line     | Artifact                                 |
| ------------------ | -------- | ---------------------------------------- |
| `workflow.py`      | 201, 202 | `generator.keras`, `discriminator.keras` |
| `workflow.py`      | 208      | `history.npz`                            |
| `workflow.py`      | 206      | `params.npz` (via `train.save_params`)   |
| `baselines/ibu.py` | 417      | `ibu_weights.npz`                        |

Readers:

| File                           | Line          | Artifact                                       |
| ------------------------------ | ------------- | ---------------------------------------------- |
| `workflow.py`                  | 238, 240      | `generator.keras`, `history.npz`               |
| `workflow.py`                  | 299           | `ibu_weights.npz`                              |
| `evaluate.py`                  | 498           | `generator.keras`                              |
| `benchmarks/ceiling.py`        | 280, 281, 415 | `generator.keras`, `params.npz`, `history.npz` |
| `benchmarks/averaging.py`      | 244           | `history.npz`                                  |
| `benchmarks/sliced.py`         | 317           | `generator.keras`                              |
| `benchmarks/hparam_collect.py` | 266           | `history.npz`                                  |

`train.save_params` and `train.load_params` both take a `run_dir` and join
`PARAMS_FILE` themselves (`train.py:150`, `train.py:166`), so routing
`params.npz` through `artifacts_dir` is a two-line change confined to those two
functions --- callers in `workflow.py`, `benchmarks/averaging.py` and
`benchmarks/ceiling.py` keep passing a run directory and need no edit. The one
exception is `benchmarks/ceiling.py:281`, which probes `(run_dir /
PARAMS_FILE).exists()` directly and must move to `artifacts_dir(run_dir) /
PARAMS_FILE`.

Beyond the binaries, these writers and readers also move into `artifacts/`:
`metrics.json` (`evaluate.py:486`), `metrics_ibu.json` (`ibu.py:396`),
`timings.json` (`timing.py:275`), and the four figure paths in `_draw_figures`
(`workflow.py:281-288`). `benchmarks/hparam_collect.py:209` and
`benchmarks/sliced.py:219` read `metrics.json` and follow. `config.json` does
not move.

Tests naming a moved artifact: `tests/test_completion_logging.py:69,172`,
`tests/test_workflow.py:256,387`, `tests/test_train.py:452,454`,
`tests/test_hparam_collect.py:320`.

Documentation: `README.md:322-329`, `overview.md:266-274`, and the Project
Structure section of `CLAUDE.md`.

## B. `config.json` formatting

`variables` renders on one line; every other key keeps `indent=2`.

`json.dump` cannot express this, so `_write_run_dir` (`workflow.py:231`) serializes
as today and collapses the one array with a targeted regex before writing:

```python
_VARIABLES_BLOCK = re.compile(r'"variables": \[\n(?:[^\]]*?)\n(\s*)\]')
```

replaced by a single-line rendering of the same list. A helper
`_compact_variables(text: str) -> str` holds this; nothing else in the file is
touched.

**Tests:** the rendered file contains `"variables": ["m", "M", ...]` on one
line; `json.loads` of the written file equals the dict that was dumped; a
config with no `variables` key (the Gaussian path) is returned unchanged.

## C. Figures

### C.0 Jet variable presentation order

Figures and the report's metrics tables both present the twelve jet observables
in a physics-motivated order rather than in column order. Add to
`rantypes/constants.py`:

```python
JET_DISPLAY_ORDER: Final[tuple[LiteralString, ...]] = (
    "m", "sdm",                       # mass and hard scale
    "lha", "w", "ang2",               # continuous angularities
    "zg", "tau21",                    # splitting and 2-prong substructure
    "M", "n_ch", "f_ch", "ptd", "q",  # hadronization and fragmentation
)

JET_VARIABLE_GROUPS: Final[tuple[tuple[str, tuple[LiteralString, ...]], ...]] = (
    ("Mass and hard scale (IRC-safe kinematics)", ("m", "sdm")),
    ("Continuous angularities (IRC-safe jet shapes)", ("lha", "w", "ang2")),
    ("Splitting and 2-prong substructure", ("zg", "tau21")),
    ("Hadronization, multiplicity and fragmentation (IRC-unsafe)",
     ("M", "n_ch", "f_ch", "ptd", "q")),
)
```

reading `m -> ln rho -> lambda^1_0.5 -> w -> lambda^1_2 -> z_g -> tau_21 -> M ->
n_ch -> f_ch -> p_T^D -> q`. The rationale, from the project owner: `M` adjacent
to `n_ch` exposes the baseline hadronization ratio (`n_ch / M ~ 2/3` from pion
isospin); `f_ch` bridges particle counting and track-based energy
reconstruction; `p_T^D` completes the classic quark/gluon discriminant system
with `M` and `n_ch` (gluons high-multiplicity and low-`p_T^D`, quarks the
reverse); `q` closes as the valence flavor indicator.

**This is a display permutation and nothing more.** `SUBSTRUCTURE_VARIABLES` is
unchanged: it is the cache key, the `config.json` record, and what
`cli._canonical_variables` sorts `--var` into, and the Jet Column Order section
of `CLAUDE.md` documents what happened the last time that ordering was allowed
to float. The column order carries no physics and must not acquire any; the
display order carries physics and must not touch a cache key.
`JET_DISPLAY_ORDER` is applied at render time only, by `plotting._plot_level`
and by `report.py`.

A helper does the work, and must handle subsets since `--var` selects any of
them:

```python
def display_order(variables: Sequence[str], /) -> tuple[int, ...]:
    """Indices into `variables`, reordered for presentation."""
```

It filters `JET_DISPLAY_ORDER` to the variables actually present, then maps back
to their column indices. For a non-jet dataset (`dim_0`, `dim_1`, ...) it
returns the identity, so the Gaussian configs are unaffected.

**Grouping renders in the tables, not the figures.** The group sizes are
2-3-2-5 and do not align with the rows of a 3-column grid. Colour-coding panel
titles would collide with the figure palette, where colour already means
"method". So the figures take the order alone --- groups stay contiguous in
reading order --- and `JET_VARIABLE_GROUPS` drives the grouped blocks in the
report's two metrics tables (see F), where LaTeX can show the structure
properly.

**Tests:** `display_order` on the full twelve returns the documented
permutation; on a two-variable subset returns those two in display order; on a
Gaussian run returns the identity. `SUBSTRUCTURE_VARIABLES` is asserted
unchanged, and `set(JET_DISPLAY_ORDER) == set(SUBSTRUCTURE_VARIABLES)` so the
two can never drift apart silently.

### C.1 Shared: label clipping

`_save_fig` (`plotting.py:212`) calls `figure.savefig(fname=save_path)` with no
`bbox_inches`. `plot_losses` (`plotting.py:435`) passes `bbox_inches="tight"`
and does not clip; the other three figures go through `_save_fig` and do. Add
`bbox_inches="tight"` to `_save_fig`, and route `plot_losses` and
`plot_selection` through it so there is one save path.

### C.2 Palette

Replace the color literals scattered through `_hist_ratio_panel`
(`plotting.py:112, 123, 134, 161, 169, 183, 200`) with named constants at module
scope:

```python
COLOR_NATURE: Final[str] = "C0"      # Data / Truth
COLOR_MC: Final[str] = "C1"          # Sim / Gen
COLOR_IBU: Final[str] = "green"
COLOR_RAN: Final[str] = "#6A3D9A"    # deep violet
```

`COLOR_RAN` replaces the `"black"` at lines 134 and 169. Violet is distinct
from the blue/orange/green already in the figure, reads as deliberate in print,
and greyscales to a dark mid-tone that separates from the `0.85` gridlines.

**Opacity is the other half of why RAN looks faded, and it is currently
backwards.** RAN's step histogram is `alpha=0.35` (`plotting.py:138`) and its
ratio line is `alpha=0.35` (`plotting.py:169`), while IBU's ratio line is
`alpha=0.75` (`plotting.py:205`). The baseline is drawn twice as opaque as the
method being showcased, on the same panel. Fix the hierarchy explicitly:

| Element | alpha |
| --- | --- |
| Filled `Data`/`Truth` histogram | 0.35 (background) |
| Filled `Sim`/`Gen` histogram | 0.35 (background) |
| RAN step histogram and ratio line | 0.90 |
| IBU step histogram and ratio line | 0.75 |

Named constants alongside the colors (`ALPHA_FILL`, `ALPHA_RAN`, `ALPHA_IBU`)
so the ordering is stated in one place rather than re-derived at seven call
sites.

### C.3 Panel grid

`_plot_level` (`plotting.py:280`) hardcodes `ncols=1`, producing a 1x12 column
for the shipped twelve-observable run. Replace with:

```python
ncols = min(3, dim)
nrows = math.ceil(dim / ncols)
```

Figure size becomes `(4.0 * ncols, style.height_per_dim * nrows)`. The outer
`GridSpec` becomes `(nrows, ncols)`, iterated over `display_order(...)` (C.0) so
that panel position `p` draws column `order[p]` at
`outer_grid[p // ncols, p % ncols]`; each cell keeps its existing 2-row
hist/ratio subgrid. The absolute `left`/`right`/`bottom`/`top` margins,
currently computed against `height`, are replaced by a single
`figure.tight_layout()` after the panels are drawn: fixed margins expressed in
inches do not survive a figure whose height now varies with `nrows`.

Consequences: 12 dims give 4x3, 6 give 2x3, 2 give 1x2, and the 1D Gaussian
config gives a single panel as today.

### C.4 `losses.pdf`

Four changes to `plot_losses` (`plotting.py:399`):

1. `ylabel="Weighted BCE"` (currently `"WeightedBCE"`, `plotting.py:430`).
2. The `log(2)` axhline (`plotting.py:423`) loses its `label=`, so it leaves the
   legend.
3. Fixed y-limits, never autoscaled: `ylim = (LN2 * (1 - 2**-4), LN2 * (1 + 2**-4))`
   = `(0.6498, 0.7365)`.
4. Explicit y-ticks at `LN2 * (1 + k * 2**-5)` for `k` in `-2..2`. The center
   tick is labeled `$\ln 2$`; the other four carry absolute values to four
   decimals. A twin right-hand axis shares the limits and labels the same five
   positions as percent deviation from `ln 2` (`-6.2%`, `-3.1%`, `0`, `+3.1%`,
   `+6.2%`).

The fixed scale is the substantive change. It makes the plot comparable across
runs, and the right-hand axis lets a reader see "everything is within 1% of
equilibrium" without arithmetic --- which is the true statement the current
autoscaled figure obscures. See Appendix A.

### C.5 `selection.pdf`

`plot_selection` (`plotting.py:439`) currently overlays two MMD curves and an
ESS twin axis on one set of axes, with `legend(loc="best")` landing over the
data. It is answering three questions at once. Restructure to two stacked panels
plus an inset:

**Top panel (MMD, ~70% of the height).**

- Both raw traces at `alpha=0.3`, `lw=1`.
- A 5-epoch centered rolling median of each, bold (`lw=2`), in the same colors.
- `SELECTION_MMD_LINTHRESH` (5e-4) drawn as a shaded `axhspan` from 0, labeled
  as the estimator's resolution floor.
- The selected epoch as a vertical dotted line.
- Legend placed outside the axes (`bbox_to_anchor`, right side), never over data.
- Keeps the existing `symlog` scale with `linthresh=SELECTION_MMD_LINTHRESH`.

**Inset (upper right of the top panel).** Scatter of detector MMD² against
particle MMD², one point per epoch, with the selected epoch marked. This is the
panel that answers whether the truth-free criterion tracks the particle-level
one --- a correlation that two overlaid noisy time series cannot show. Drawn
only when `val_mmd_particle` is present (a real measurement has no truth).

**Bottom panel (ESS, ~30%).** `val_ess` alone, plotted as percent of its
epoch-0 value, sharing the x-axis. The current twin axis spans 13.9k-14.9k over
the full figure height, making 7% drift look like a collapse; a percent axis
anchored to include 100% reports it as 7% drift.

The figure will still show a noisy criterion, because the criterion is noisy
(detector MMD² oscillates over a factor of ~4 between adjacent epochs, at 2-10x
the resolution floor). Stating that plainly is the goal; smoothing it away
would not be.

### C.6 Ratio-panel tick collision

The main panel's bottom y-tick (`0`) and the ratio panel's top y-tick (`1.5`)
land at the same height where the two subgrid axes meet, and overprint each
other. Prune the lower tick of the main axis:

```python
ax.yaxis.set_major_locator(MaxNLocator(prune="lower"))
```

`prune="lower"` drops the lowest tick label only when it sits at the axis edge,
so the `0` disappears and every tick above it stays. Preferred over hardcoding
`set_ylim` or hiding `ax.yaxis.get_major_ticks()[0]`, both of which misbehave
when the data range changes between panels.

**Test:** for a drawn panel, no main-axis tick label shares a y-position with a
ratio-axis tick label.

## D. Metric key ordering

`baselines/ibu.py:355-378` writes `detector_<var>` and `particle_<var>` into one
dict inside the per-variable loop, giving an interleaved order.
`evaluate.py:517-526` loops levels outermost, giving all detector entries then
all particle entries. Change IBU to match `evaluate`: accumulate `detector` and
`particle` dicts separately in the loop and merge detector-first when building
the `IBUResult`.

Nothing reads `metrics_ibu.json` positionally today. That is precisely why this
is cheap to fix now and expensive later.

**Test:** for a multi-variable run, `list(metrics_ibu.json)` equals
`list(metrics.json)`.

## E. Timings merge

### Behavior

`timing.write(run_dir)` currently overwrites (`timing.py:275`). It becomes a
merge:

1. Read `timings.json` if it exists; treat a malformed or unreadable file as
   absent rather than raising, since the timing layer must never take a run
   down.
2. Merge phase records by `name`: a phase produced by this pass replaces any
   existing record of the same name; a phase present only in the old file is
   retained.
3. Order: retained old phases first, in their original order, then new phases
   not already present.
4. Recompute `total_seconds` over the merged **top-level** (`depth == 0`)
   phases only, preserving the existing rule.
5. `compile_cache_warm` takes the new pass's value when the new pass has one,
   else retains the old.

### Pass attribution

Each phase record gains a `pass` field naming the invocation that produced it.
`write` takes a new keyword argument:

```python
def write(run_dir: Path, /, *, pass_name: str) -> None:
```

`workflow.run` (`workflow.py:419`) passes `"load" if load_run is not None else
"train"`. `baselines/ibu.py` is not timed today and stays untimed; the field
exists so that adding it later needs no format change.

This is what makes a merged file legible: a reader seeing an `epochs` row next
to a `plots` row can tell that the first came from the training pass and the
second from the reload.

### Why merging is safe

`_new_run_dir` (`workflow.py:113`) refuses a directory already holding a
`config.json`, so a fresh training run cannot merge its numbers into an
unrelated run's file. Only `--load-run` and the baseline reuse a directory, and
those are exactly the passes we want merged.

**Tests:** a train pass followed by a load pass into the same directory yields a
file containing both `epochs` (pass `train`) and `plots` (pass `load`); a
corrupt existing `timings.json` is overwritten rather than raising;
`total_seconds` after a merge equals the sum of merged top-level phases.

## F. `ran report`

### Entry point

`src/ran/report.py`, exposed as `ran report --run-dir <path>` in the existing
Typer tree. One tree rather than a second script: `ran --install-completion`
binds to the console-script name, so a second entry point means a second
completion install for no gain.

The run directory is a positional **argument**, not a required option:
`ran report runs/2026-09-06T203848Z` reads more naturally than the same thing
behind a flag, and a required option is a flag in name only. The rest are
options: `--force` (regenerate an existing `report.pdf`) and `--no-compile`
(emit `report.tex` and stop).

### Inputs and degradation

Reads `config.json`, `metrics.json`, `metrics_ibu.json`, `timings.json`, and the
four figure PDFs from the run directory. Every input except `config.json` is
optional:

- No `metrics_ibu.json`: the IBU cells carry `\multicolumn{1}{c}{---}` and a
  note under the table says the baseline was not run. This **reverses** an
  earlier decision to omit the column group: the template fixes the column
  specification at sixteen columns, so omitting a group would mean a second
  template, and one template with dashes is the cheaper honest answer.
- No `timings.json`: the timings table is omitted.
- A missing figure: handled by the template, not the generator. `\ReportGraphic`
  wraps `\IfFileExists` and substitutes a labelled placeholder box, so
  `report.py` always emits the path and never has to probe the filesystem.
- No `config.json`: hard error. There is no run without one.

### Templating

Placeholders are `<<NAME>>`, substituted by `str.replace`. The alternatives all
fail against LaTeX: `{name}` breaks `str.format` on LaTeX's braces, `$name`
collides with math mode under `string.Template`, and `%%NAME%%` silently
becomes a LaTeX comment when a substitution is missed. After substitution
`report.py` scans for any surviving `<<...>>` and raises naming the token, so a
template/code mismatch fails loudly instead of producing a subtly empty
document.

Template lives at `src/ran/templates/report.tex`, shipped as package data.
It is authored and owned by the project owner; the working copy delivered for
this design is `response.tex` at the repo root, and the implementation moves it
into place under its final name.

**The guard must match `<<[A-Z_]+>>`, not a bare `<<`.** The template's own
header comment documents the replacement contract and legitimately contains the
string `<<...>>`. A bare-`<<` check fails on a correctly substituted document.

**The template owns all rounding; `report.py` owns none of it.** Every numeric
cell goes into a siunitx `S` column that fixes `round-mode`, `round-precision`
and `table-format`, so the generator emits values at full precision and lets
LaTeX round and align them on the decimal marker. This is better than
formatting in Python --- a column formatted string-by-string cannot align.

Two consequences the generator must respect:

- **Plain decimal notation, never exponential.** An `S` column with a
  `table-format` cannot absorb an exponent. `f"{x:.10f}"` with trailing zeros
  stripped is the emission rule.
- **`table-format=2.8` on the distance columns is sized from real data.** The
  smallest metric in the reference run is `8.294e-5`, which at four significant
  figures is `0.00008294` --- eight decimals. The template was `2.7`, which
  silently misaligns rather than erroring; the implementation ships `2.8`, and
  `report.py` logs a warning if a value needs more, so a future overflow is
  visible instead of quietly ragged.

**Values are not escaped by the template; keys are.** `\ConfigKey` detokenizes
the key, but nothing protected the value, and `f_ch` inside a `\texttt{}` value
is a hard `Missing $ inserted` compile error --- found by compiling the template
against the reference run. The template gains a `\ConfigVal` wrapper for text
values, and `report.py` routes every text value through it.

Tokens:

| Token                | Expands to                                                            |
| -------------------- | --------------------------------------------------------------------- |
| `<<RUN_NAME>>`       | The run directory's basename                                          |
| `<<CONFIG_ROWS>>`    | `booktabs` row bodies: parameter & value                              |
| `<<TIMINGS_ROWS>>`   | Row bodies: phase (indented by `depth`), seconds, share, pass, detail |
| `<<DETECTOR_TABLE>>` | Row bodies for the detector-level metrics table                       |
| `<<PARTICLE_TABLE>>` | Row bodies for the particle-level metrics table                       |
| `<<FIGURE_DIR>>`     | Absolute path the `\includegraphics` calls resolve against            |

The template owns every `\begin{tabular}`, column specification, `\cmidrule` and
page break. `report.py` emits row bodies only. The template is authored by the
project owner and supplied separately; until it arrives, a minimal scaffold
template exercising all six tokens stands in.

### Emitting `<<CONFIG_ROWS>>` and `<<TIMINGS_ROWS>>`

The config table is four columns wide and holds two key/value pairs per row, so
`report.py` chunks the scalar entries pairwise into `\ConfigPair{k}{v}{k}{v}`,
padding a trailing odd entry with an empty pair. Long values --- `variables`,
and the two `mmd_sigmas_*` rows below --- use `\ConfigWide`, which spans the
remaining three columns. `compile_cache_warm`, which lives at the top level of
`timings.json` rather than in `config.json`, is emitted as a config row: it is
the fact that makes the `compile` timing interpretable at all.

The timing table is `Phase | Seconds | Share (%) | Detail`. A nested phase wraps
its name in `\TimingSubphase`; its Share cell is left **empty** rather than zero,
since a nested phase is already inside its parent's share. The body ends with a
`\midrule` and a bold total row carrying `total_seconds`.

The `pass` field added in E has no column here, and widening the table for it is
not worth it. It is appended to the Detail cell instead --- `cache hit - train`,
or just `train` where there is no detail --- which is enough to explain why an
`epochs` row and a `plots` row are in the same table.

### Rendering `mmd_sigmas_*` in the config table

`config.json` records `mmd_sigmas_detector` and `mmd_sigmas_particle` as five
bare floats each. Printed verbatim in the config table they are ten opaque
numbers that tell a reader nothing. They are not free parameters: they are the
median-heuristic bandwidth (`mmd.median_bandwidth`, computed from the data side
alone so every seed and every hyperparameter arm shares an identical kernel)
bracketed by `_SCALES = (1/2, 1/sqrt2, 1, sqrt2, 2)` (`mmd.py:47`), and the
kernel actually used is the sum of the five RBFs.

`<<CONFIG_ROWS>>` therefore collapses each list to one row naming the median and
the bracket --- for the reference run, `3.107 x (1/2 .. 2)` --- and the template
carries a one-line footnote saying what a bandwidth is for. `report.py` verifies
the recorded values are in fact `median * _SCALES` before collapsing, and falls
back to printing them in full if they are not, so a future change to the scale
set cannot make the report quietly lie.

### Metrics tables

Two tables of identical shape. Rows are the run's variables in **display
order** (C.0), split into the four `JET_VARIABLE_GROUPS` blocks, each introduced
by a rule and a spanning group-name row. A group with no variables in this run
--- possible under `--var` --- is omitted along with its header. A non-jet run
gets rows `dim_0 ... dim_n` and no grouping. Primary column groups are the three metrics; each carries the same
sub-columns.

Detector level, distance to **Data**:

| | Wasserstein | | | | | JS divergence | ... | Triangular | ... |
| Variable | Sim | IBU | IBU % | RAN | RAN % | Sim | ... | Sim | ... |

Particle level is identical with **Gen** in place of **Sim**, measuring distance
to Truth.

The third metric group is headed `VLC divergence [x10^3]`. The scale factor is
not cosmetic: `evaluate._triangular_from_histograms` (`evaluate.py:403`)
multiplies by `1e3` before writing, so the numbers in `metrics.json` are already
scaled and a header without the factor misstates them by three orders of
magnitude. `render_metrics` already labels its Rich column `Delta (x1e3)`.

Mapping onto the JSON: `Sim`/`Gen` is `<metric>_before` (identical in both
files, as both measure the same unreweighted baseline --- worth asserting);
`IBU` is `<metric>_after` from `metrics_ibu.json`; `RAN` is `<metric>_after`
from `metrics.json`; the `%` columns are the corresponding
`<metric>_improvement_pct`.

**Number formatting.** Distances at four significant figures (`%.4g` ---
`0.2089`, `1.480e-04`). Improvement percentages at one decimal place (`98.5`,
`-504.3`). Four significant figures on a percentage implies a precision that is
not there. A `_fmt_distance` and a `_fmt_pct` helper, both unit-tested including
the negative-improvement case that IBU's particle-level rows actually produce.

### Surfacing IBU's failures

Compiling the template against the reference run exposed a reporting hazard.
For `z_g`, IBU returns its input unchanged --- detector `Sim 0.06045 / IBU
0.06045 / 0.0%`, particle `Gen 0.1134 / IBU 0.1134 / 0.0%`. That is not IBU
performing badly; it is `unfold_variable`'s purity binning giving up, which
`IBUResult.outcomes` already records as a `VariableOutcome` and which
`ran sweep collect` deliberately reports rather than hides. A bare `0.0` in a
results table reads as a measurement.

`report.py` therefore reads `outcomes` alongside `metrics_ibu.json` and marks
every affected cell with a dagger, with one footnote under the table naming the
condition. Where the outcomes are unavailable --- an older `metrics_ibu.json`
written before this --- the fallback is to flag any variable whose IBU `after`
equals its `before` exactly, which is the observable signature of the same
event.

**Test:** a run whose IBU outcome reports a purity failure marks that row and
emits the footnote; a run with no failures emits neither.

### Document structure

| Page | Content                                         |
| ---- | ----------------------------------------------- |
| 1    | Run name, `<<CONFIG_ROWS>>`, `<<TIMINGS_ROWS>>` |
| 2    | Detector-level metrics table                    |
| 3    | Particle-level metrics table                    |
| 4    | `detector_level.pdf`                            |
| 5    | `particle_level.pdf`                            |
| 6    | `losses.pdf` and `selection.pdf`                |

Page assignment is the template's business; `report.py` guarantees only that
the tokens are available.

### Compilation

`report.py` writes `artifacts/report.tex` and compiles it to `report.pdf` at
the run root -- the only output of this command that is not supporting material.

`subprocess.run` of `pdflatex -interaction=nonstopmode -halt-on-error` twice
(the second pass settles `\includegraphics` box sizes and any references), run
in `artifacts/` with `-output-directory` pointed at the run root, with `.aux`/`.log`/`.out` removed on success and retained on
failure. A missing `pdflatex` raises a message naming the binary and pointing at
`--no-compile`, not a `FileNotFoundError` traceback. A non-zero exit surfaces the
tail of the LaTeX log.

## G. Testing

| Area | Test |
| --- | --- |
| Layout | A completed run leaves only `report.pdf` and `config.json` at the run root; everything else is under `artifacts/` |
| Layout | `--load-run` on a new-layout directory reloads successfully |
| Config | `variables` renders on one line; the file still round-trips through `json.loads` |
| Figures | `_save_fig` passes `bbox_inches="tight"` |
| Figures | 12 dimensions produce 12 hist axes across 4 rows and 3 columns |
| Figures | Panels are drawn in `JET_DISPLAY_ORDER`, and a `--var` subset filters it |
| Ordering | `display_order` returns the identity for a non-jet run |
| Ordering | `set(JET_DISPLAY_ORDER) == set(SUBSTRUCTURE_VARIABLES)`, and the latter is unchanged |
| Report | Table rows are grouped into the four `JET_VARIABLE_GROUPS` blocks; an empty group is omitted |
| Figures | RAN is drawn more opaque than IBU, and both more opaque than the filled histograms |
| Figures | The main axis prunes its lowest tick, so no label collides with the ratio panel |
| Figures | `plot_losses` sets `ylim` to `ln2 * (1 +/- 2**-4)` and `log(2)` is absent from the legend |
| Figures | `plot_selection` produces two panels; the inset appears only when `val_mmd_particle` is present |
| Ordering | `list(metrics_ibu.json)` equals `list(metrics.json)` for a multi-variable run |
| Timings | Train pass then load pass into one directory retains both `epochs` and `plots`, with correct `pass` fields |
| Timings | A corrupt existing `timings.json` is replaced, not raised on |
| Timings | `total_seconds` after a merge equals the sum of merged top-level phases |
| Report | `report.tex` contains the expected numbers and no surviving `<<[A-Z_]+>>` |
| Report | A template comment containing a literal `<<...>>` does not trip the guard |
| Report | Emitted numbers are plain decimals; no cell contains `e-` or `E+` |
| Report | A text value containing `_` compiles (regression: `f_ch` broke the first build) |
| Report | An IBU purity failure marks its cells and emits the footnote |
| Report | A run directory whose name contains `_` resolves its figure paths |
| Report | A run with no `metrics_ibu.json` omits the IBU columns and still renders |
| Report | Formatting helpers: 4 sig figs on distances, 1 dp on percentages, negatives handled |
| Report | `pdflatex` compile succeeds --- `skipif` the binary is absent, so CI without TeX still passes |

## Appendix A: the loss curves are correct

The falling train losses and rising validation loss in `losses.pdf` were
reviewed for an arithmetic bug. There is none. The chain checked:
`bce_sums` (`train.py:243`), `weighted_bce` (`train.py:256`), `_disc_loss`
(`train.py:299`), `_gen_loss` (`train.py:316`), the `-g_loss` un-negation in
`_group` (`train.py:462`), and the means in `one_pass` (`train.py:472`). The
recorded generator number is the **unpenalised** adversarial term
(`train.py:344`), so the dispersion penalty does not contaminate the scale.

For run `2026-09-06T203848Z`:

| Series    | Epoch 0 | Peak         | Epoch 99 |
| --------- | ------- | ------------ | -------- |
| `train_d` | 0.6891  | 0.6910 (~e6) | 0.6884   |
| `train_g` | 0.6888  | 0.6906       | 0.6870   |
| `val_d`   | 0.6901  | ---          | 0.6932   |

`ln 2 = 0.693147`. The entire plotted range is a ±0.4% band around it. Two
apparent anomalies, both expected:

- **`train_g` sits ~3e-4 below `train_d`.** Not a sign error. `train_d` averages
  over all `n_disc_steps` batches of a group; `train_g` is measured on
  `group_idx[0]` alone --- the batch `d` has just stepped on --- so `d` fits it
  marginally better and the BCE is marginally lower.
- **`val_d` overshoots `ln 2`** (0.6937 at epoch 92). A discriminator performing
  _worse than chance_ on held-out events is what overfitting the train split
  looks like, and the overshoot is ~1e-4.

The underlying behavior is the textbook one: `d` slowly memorizes the train
split while `val_d` climbs to chance, which is the adversarial equilibrium, and
the metrics agree (94% Wasserstein improvement on `m`). The fix is entirely
presentational and is section C.4.

## Appendix B: why the training block vanished from `timings.json`

`scripts/submit.sh` makes three passes over one run directory: `ran train`,
then `ran baseline ibu`, then `ran train --load-run` to redraw the figures with
the baseline overlaid. The third pass calls `timing.write` on the same
directory, and `write` overwrites. The reload path has no `train`, `transfer`,
`compile`, `epochs` or `select` phases --- it has `load` --- so the training
numbers are written by pass one and destroyed by pass three on every run of the
pipeline. Section E is the fix.
