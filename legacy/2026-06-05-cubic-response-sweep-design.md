# Cubic Response Sweep: RAN vs OmniFold — Design

**Date:** 2026-06-05
**Status:** Approved (pending spec review)

## Motivation

Demonstrate the central robustness claim of RAN versus OmniFold under a
controlled, deterministic non-linear detector response.

- RAN reweights **only** at particle (gen) level; its discriminator reads
  detector level but the generator only needs particle-level *support* overlap
  between `z_truth` and `z_gen`.
- OmniFold reweights at **both** detector and particle level (Step 1 matches MC
  reco to data reco; Step 2 pulls back to gen level), so it needs good overlap
  at **both** levels.

By sweeping a parameter `s` that controls a cubic detector distortion, we widen
the gap between the data and sim *detector-level* distributions while keeping the
*particle-level* distributions fixed (and overlapping in support). The hypothesis:
at small `s` both methods unfold equally well; as `s` grows, OmniFold degrades
because detector-level overlap collapses, while RAN stays accurate because
particle-level support is unchanged.

## Reference pseudocode (target behavior)

```python
z_truth = rng.normal(0, 1, N)
z_gen   = rng.normal(-1, 1, N)

r = lambda s, z: z + s * z**3
for s in np.linspace(0, 20, 25):
    x_data = r(s, z_truth)
    x_sim  = r(s, z_gen)
    z_unfolded_RAN      = RAN(z_gen, x_data, x_sim)
    z_unfolded_OmniFold = OmniFold(z_gen, x_data, x_sim)
    RAN_performance      += [wasserstein(z_truth, z_unfolded_RAN)]
    OmniFold_performance += [wasserstein(z_truth, z_unfolded_OmniFold)]
```

## Decisions

| Parameter | Value | Notes |
|-----------|-------|-------|
| Orchestration | SLURM array job | One array task per `s`; dependent collect job. |
| N (events/class) | 500,000 | Matches repo default `n_samples`. |
| `s` grid | `np.linspace(0, 20, 25)` | 25 array tasks. |
| Replicates | 1 (single seed) | No error bands; training noise shows as curve jitter. |
| Detector response | Deterministic `z + s·z³` | No added noise. Monotonic for `s ≥ 0`. |
| Particle samples | Drawn once, fixed seed | Same `z_truth`/`z_gen` reused for every `s`. |
| Dimensionality | 1D | `dim=1` throughout. |

## Architecture

### New module: `ran/experiments/cubic_sweep.py`

Run as `python -m ran.experiments.cubic_sweep <command>` via `fire`.

Core pieces:

- `response(s, z)` → `z + s * z**3`. Deterministic; shared by data and sim.
- `make_particles(n_samples, seed)` → `(z_truth, z_gen)` with
  `z_truth ~ N(0,1)`, `z_gen ~ N(-1,1)`. Drawn once per sweep with a fixed seed.
- `unfolded_wasserstein(z_truth, z_gen, weights)` → weighted Wasserstein:
  `scipy.stats.wasserstein_distance(z_truth, z_gen, v_weights=weights)`.
  `z_unfolded` is the `z_gen` empirical distribution reweighted by `weights`.
- `run_point(s_index, sweep_dir, n_samples=500_000, ...)`:
  1. `z_truth, z_gen = make_particles(...)`.
  2. `s = s_grid[s_index]`; `x_data = response(s, z_truth)`,
     `x_sim = response(s, z_gen)`.
  3. **RAN:** build `DatasetSplits` from `(z, x, y)` via the new
     `RAN_Dataset.splits_from_arrays`; `g, _, _ = train(splits, dim=1, ...)`;
     `w_ran = normalize(g(z_gen))`.
  4. **OmniFold:** `w_of = omnifold_unfold(x_data, x_sim, z_gen, ...)`.
  5. `ran_wd = unfolded_wasserstein(z_truth, z_gen, w_ran)`;
     `of_wd  = unfolded_wasserstein(z_truth, z_gen, w_of)`.
  6. Write `<sweep_dir>/s_{s_index:02d}.json` = `{s, ran_wd, omnifold_wd}`.
- `collect(sweep_dir)`:
  - Read all `s_*.json`, sort by `s`.
  - Save `<sweep_dir>/results.npz` (`s`, `ran`, `omnifold`).
  - Save `<sweep_dir>/wasserstein_vs_s.pdf`: both curves vs `s`, x-axis `s`,
    y-axis Wasserstein(`z_truth`, `z_unfolded`), legend RAN / OmniFold.

Weight normalization for RAN mirrors the existing convention in
`ran/train.py::_compute_weights`: `w *= len(w) / w.sum()` (mean → 1).

The sweep directory is created by the submit script (timestamped) and passed to
every task, so all array tasks and the collect job share one directory.

### Targeted edits to existing code

1. **`ran/data/datasets.py`** — add public method:
   ```python
   def splits_from_arrays(self, z, x, y) -> DatasetSplits:
       self.dataset = self._build_dataset(z, x, y)
       self.splits = self._split_dataset(self.dataset)
       return self.splits
   ```
   Wraps the existing private helpers that `generate_gaussian_dataset` already
   uses. No behavior change to the Gaussian path.

2. **`ran/baselines/omnifold.py`** — extract a reusable function:
   ```python
   def omnifold_unfold(x_data, x_sim, z_gen, niter=3, epochs=50) -> np.ndarray:
       """Train OmniFold on in-memory arrays; return mean-normalized gen weights."""
   ```
   This contains the `DataLoader` / `MultiFold` / `Unfold` / `reweight` logic
   currently inlined in `_run_and_evaluate`. The existing `_run_and_evaluate`
   is refactored to call it, preserving current baseline behavior.

### New script: `scripts/submit_sweep.sh`

SLURM array job following `scripts/submit.sh` conventions
(`--constraint=gpu`, account, `cd` to project, `uv run`).

- `#SBATCH --array=0-24` (25 tasks).
- The submit script computes one timestamped sweep dir
  (`runs/cubic_sweep_<timestamp>`) at submission time, `mkdir -p`s it, and passes
  it to every array task and the collect job via `--sweep_dir=$DIR`. All tasks
  therefore share a single directory; no inter-task coordination is needed.
- Each task: `uv run -m ran.experiments.cubic_sweep run_point
  --s_index=$SLURM_ARRAY_TASK_ID --sweep_dir=$DIR`.
- Dependent collect job (`--dependency=afterok:<arrayjobid>`):
  `uv run -m ran.experiments.cubic_sweep collect --sweep_dir=$DIR`.

## Data flow

```
make_particles (once, fixed seed)
        │
        ├─ z_truth ~ N(0,1)      z_gen ~ N(-1,1)
        │
   for each s_index (one SLURM array task):
        │
        ├─ x_data = response(s, z_truth)
        ├─ x_sim  = response(s, z_gen)
        │
        ├─ RAN:      splits_from_arrays → train → g → w_ran
        ├─ OmniFold: omnifold_unfold(x_data, x_sim, z_gen) → w_of
        │
        ├─ ran_wd = W(z_truth, z_gen; v_weights=w_ran)
        ├─ of_wd  = W(z_truth, z_gen; v_weights=w_of)
        │
        └─ write s_{index}.json
        ▼
   collect: gather s_*.json → results.npz + wasserstein_vs_s.pdf
```

## Error handling

- A failing array task writes no JSON; `collect` proceeds with whatever points
  exist and warns about missing indices (resilience is a key reason for the
  array-job choice).
- RAN training uses the existing early-stopping `train()`; no new stopping logic.
- OmniFold weights are mean-normalized; degenerate (all-equal or NaN) weights
  from a saturated Step-1 classifier are expected at large `s` and are part of
  the signal, not an error — but NaNs are guarded (replaced/flagged) so the
  Wasserstein call does not crash.

## Testing

- Unit: `response(0, z) == z`; `response(s, z)` monotonic in `z` for `s > 0`.
- Unit: `splits_from_arrays` returns three non-empty batched datasets with the
  expected `z`/`x` feature keys.
- Unit: `unfolded_wasserstein` with uniform weights equals the unweighted
  `wasserstein_distance(z_truth, z_gen)`.
- Smoke: `run_point` at small N (e.g. 2,000) and `s_index=0` produces a JSON with
  finite `ran_wd` and `omnifold_wd`; `collect` over a couple of stub JSONs writes
  `results.npz` and the PDF.

## Out of scope

- Multi-dimensional response, added detector noise, replicate seeds / error
  bands — deferred; can be layered on later without changing the module shape.
- Changing the RAN architecture or training hyperparameters.
