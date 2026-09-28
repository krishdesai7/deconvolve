# MMD model selection

Replace GAN-loss-based checkpoint selection and early stopping with a weighted
maximum mean discrepancy computed after training, over a retained generator
parameter history.

## Problem

RAN selects its restored checkpoint from the validation BCE. That is unsound,
and not because the direction was wrong.

A GAN's loss is not a proxy for a single scalar objective monotonically related
to model quality. At equilibrium it oscillates around its equilibrium value by
construction, and movement away from that value can reflect either network
improving. A flat loss can mean equilibrium, or it can mean both networks are
stuck and no longer supplying each other useful gradient. An oscillating loss
can mean healthy adversarial dynamics or divergence. The number is
underdetermined with respect to what we care about.

`log 2 - BCE` estimates the Jensen-Shannon divergence between the reweighted
distributions **only when `d` is optimal**, and nothing in the loop reports
whether it is. Both criteria built on it inherit the defect:

- `max_val` rewards `g` overshooting a stale `d`. A checkpoint above log 2 is
  one where `d` is worse than chance on held-out events.
- `dist_log2` is maximized by a *collapsed* `d` — a constant 0.5 scores
  perfectly with `g` doing nothing.

The patience-and-early-stopping model is an EM-shaped assumption applied to a
quantity that does not satisfy it. The fix is not a better reading of the loss;
it is to select on an actual divergence. MMD is zero if and only if the
distributions match (for a characteristic kernel), is monotone in mismatch, and
requires no adversary and no optimization — it is a closed-form functional of
the weights. Selecting on it restores exactly the property the original design
assumed it had.

Evidence from `runs/2026-08-24T020853Z` (six jet variables, 1M events):
`val_d` spans 0.691756 to 0.693594 across 100 epochs — a range of 1.8e-3 around
log 2 = 0.693147. `max_val` restores epoch 76; `dist_log2` restores epoch 23.
Same run, same curve, 53 epochs apart.

## Decisions taken

| Question | Decision |
| --- | --- |
| Which distributions does selection compare? | **Detector level**: `MMD(w·x_sim, x_data)`. Truth-free, so the method stays deployable. Particle-level MMD is computed and recorded every epoch as a diagnostic only. |
| What happens to the BCE criteria? | **Deleted.** `SelectionCriterion`, `--criterion`, and the recorded config key are removed. MMD is the only criterion. |
| Where does selection run? | **Post-hoc on the host**, over a retained parameter history. Not in the trace. |

## Architecture

`lax.while_loop` becomes `lax.scan` over a fixed `n_epochs`. The loop stops
carrying `best_state`, `best_score`, `best_epoch` and `wait`; instead each epoch
emits its trainable parameters, which `scan` stacks. Selection happens after the
loop, on the host.

This is **less machinery than exists today**, not more. Three consequences,
each of which independently justifies the choice:

1. **Truth never enters the traced program.** Computing particle-level MMD
   inside the loop would put `z_true`-derived Gram matrices into the trace,
   which is precisely what `Populations` keeping `truth` outside `Events` exists
   to prevent. Post-hoc, the constraint is untouched and `leakage.py`'s
   guarantee holds unchanged.
2. **Any future criterion is a re-read, not a rerun.** The sweep ahead will want
   to ask "what would criterion X have selected?" — with the history saved that
   is an array operation.
3. **The 1 GB Gram never coexists with the training program.** It is built once,
   after the loop.

The cost is that every run pays all `n_epochs`. At the measured 0.034 s/epoch on
an A100, 100 epochs is 3.4 s against a 4.6 s compile — early stopping was saving
less wall clock than compiling the loop that implemented it.

### Memory

`g` at `-l 3 -u 128 -D 6` is ~34k parameters. Retaining `g` and `d` for 100
epochs costs ~27 MB. `d` is retained rather than saved from the final epoch so
that `generator.keras` and `discriminator.keras` come from the *same* epoch;
nothing currently reads `discriminator.keras`, but an artifact directory whose
two models are from different epochs is a trap.

## The estimator

Weighted MMD², unbiased, with a multi-scale RBF kernel:

```
MMD² = term_xx − 2·(v_xy · w) + (wᵀK_yy w − Σᵢwᵢ²·diag_yy) / (1 − Σᵢwᵢ²)
ESS  = 1 / Σᵢwᵢ²                      (w normalized to sum to 1)
```

where `term_xx = (ΣK_xx − tr K_xx) / (n(n−1))` is the standard U-statistic on
the data side and `v_xy = mean_i k(x_i, y_·)`.

### What precomputes, and what does not

Only the MC-side weights change between epochs. Everything else collapses:

| Quantity | Shape | Note |
| --- | --- | --- |
| `term_xx` | scalar | data-side only; never recomputed |
| `v_xy` | `(m,)` | **`k_xy` collapses to a vector** because the data weights are uniform and fixed |
| `K_yy` | `(m, m)` | the only matrix that must be stored |
| `diag_yy` | `(m,)` | exactly `n_scales` for a sum of RBFs; stored anyway to keep the estimator kernel-agnostic |

Measured at m = 16384: storing `k_xx` and `k_xy` as well costs 3.22 GB against
1.07 GB. Computing squared distances as `‖a‖² + ‖b‖² − 2abᵀ` rather than
`sum((a[:,None,:] − b[None,:,:])**2, -1)` avoids a **6.4 GB** build-time
`(n, m, d)` intermediate. Evaluating `term_yy` as `w @ (K @ w)` rather than
materializing `w[:,None]*w[None,:]*K` avoids an m×m temporary per call.

Measured per-evaluation cost at m = 8192 on CPU: **2.79 ms** for the form above
against **30.08 ms** for the fully-materialized form. Both agree to 7e-7
relative, so this is purely structural.

### Precision

float32 and float64 agree **exactly** at m = 8192 on the multi-scale form.
MMD² is a small difference of O(1) terms, so cancellation was the concern; with
XLA's blocked reduction it does not materialize. A regression test pins this.

### Bandwidth

Median heuristic on the **data side only**, computed once:
`σ_med = sqrt(median(‖xᵢ − xⱼ‖²) / 2)`, then five bandwidths
`σ_med · {½, 1/√2, 1, √2, 2}` summed into one kernel. Data-only rather than
pooled because `x_data` is fixed by `data_seed`, which guarantees every
hyperparameter arm shares an identical kernel — a pooled heuristic would drift
with the MC side and make arms incomparable. The bandwidths are recorded in
`config.json`.

Detector and particle level get their own bandwidths, computed and recorded
separately.

### The subsample, and the resolution floor

A fixed subsample of m = min(16384, available) rows per side, drawn with a
dedicated PRNG stream from `data_seed`.

The estimator has a measured resolution floor. Interpolating weights from "none"
to "exact" against one fixed subsample at m = 8192:

```
t ∈ [0.00, 1.00]   spearman −1.000   20/20 monotone   range 7.4e-02
t ∈ [0.90, 1.00]   spearman −0.909   16/20 monotone   range 4.8e-04
t ∈ [0.98, 1.00]   spearman +1.000    0/20 monotone   range 3.5e-05
```

Below roughly 5e-4 in MMD² the ranking **inverts**: the empirical MMD is
minimized by weights matching the sample rather than the distribution. This is
val-subsample overfitting, it scales as ~1/m, and it must be designed around
rather than ignored.

Two consequences:

- **The reported MMD for the selected checkpoint is recomputed on a fixed
  *test* subsample.** Selecting 100 times against one val sample is exactly the
  regime above; the number quoted must not be the one selection optimized.
- **The floor is measurable and gets measured.** Evaluating one weight vector
  against several independent val subsamples gives the floor directly. This is
  the first validation step below.

### Degenerate weights

As ESS → 1, `1 − Σwᵢ²` → 0 and the unbiased estimator is undefined — one
effective sample cannot estimate `E[k(y,y')]`. Guard: return `+inf` when
`1 − Σwᵢ² < _MIN_DENOM`, so a collapsed reweighting is never selected. ESS is
recorded per epoch regardless, which is the weight-degeneracy diagnostic the
adversarial objective's linear-in-`w` structure makes worth watching.

## Modules

### `src/ran/mmd.py` (new)

Pure, host-side, no knowledge of training.

```python
class MMDCache(NamedTuple):
    k_yy: Float[Array, "m m"]
    v_xy: Float[Array, " m"]
    diag_yy: Float[Array, " m"]
    term_xx: Float[Array, ""]
    sigmas: tuple[float, ...]

def median_bandwidth(x, /) -> float
def build_cache(x_data, y_mc, /, *, sigmas) -> MMDCache
def weighted_mmd(cache, raw_w, /) -> tuple[Float[Array, ""], Float[Array, ""]]
def mmd_curve(cache, raw_w_per_epoch, /) -> tuple[NDArray, NDArray]
```

`weighted_mmd` is jitted and takes **raw** generator output, normalizing
internally — the caller never has to remember which normalization applies.
`mmd_curve` takes an `(epochs, m)` array of raw weights and returns the MMD and
ESS curves.

Producing that array is the caller's job and is one `vmap` of `g`'s stateless
forward pass over the stacked parameter history against the fixed subsample's
`z` rows: 100 passes over 16384 rows through a 34k-parameter MLP, which is
negligible next to building the Gram.

### `src/ran/train.py`

- `lax.scan` over epochs; `still_running` and `_make_epoch`'s patience arguments
  removed.
- `RunCarry` shrinks to `state` and `key`. Everything else it carried is now a
  scan output: the per-epoch parameters, and the three BCE columns. The
  `(n_epochs, 3)` history buffer written a row at a time with `.at[].set()`
  disappears with it — `scan` stacks those outputs directly.
- After the loop: subsample val, build the detector cache, evaluate the curve,
  `argmin`, restore that epoch's `g` and `d`.
- `train()` loses `patience`, `min_delta`, `criterion`; keeps `n_epochs`.
- `TrainResult` gains `g_history` so the caller can compute further curves
  without retraining. `history` gains `val_mmd` and `val_ess`.

`train.py` never touches truth.

### `src/ran/workflow.py`

Computes the **particle-level** diagnostic curve from `TrainResult.g_history`
when `splits` carries truth, merges `val_mmd_particle` into the history, and
records bandwidths and the selected epoch. Drops `--patience`/`--min-delta`
plumbing.

### `src/ran/plotting.py`

New `selection.pdf`: detector and particle MMD against epoch, selected epoch
marked, ESS on a twin axis. Divergence between the two curves is the
ill-posedness made visible — the plot that answers whether truth-free selection
costs anything.

`losses.pdf` keeps the BCE curves. They remain informative as *dynamics*
diagnostics; what they are not is a selection signal.

## Artifacts

`config.json` gains `mmd_subsample`, `mmd_sigmas_detector`,
`mmd_sigmas_particle`, `best_epoch`, `mmd_test` (the honest final number), and
loses `criterion`, `patience`, `min_delta`.

`history.npz` gains `val_mmd`, `val_ess`, and `val_mmd_particle`.

## Testing

1. **Known answers.** MMD² ≈ 0 for identical samples; large for a shifted
   sample; returns near the floor when exact importance weights are applied.
2. **Equivalence.** The vectorized form matches a naive fully-materialized
   reference to 1e-6 relative.
3. **Precision.** float32 matches a float64 reference (pins the finding above).
4. **Weight invariance.** `weighted_mmd` is invariant to rescaling `raw_w`,
   since only the normalized weights enter.
5. **Degeneracy guard.** Weights concentrated on one event return `+inf`, not
   a division blow-up.
6. **Monotonicity.** MMD decreases monotonically along the interpolation from
   unweighted to exact weights, outside the measured floor.
7. **Selection.** Given a synthetic parameter history, the restored epoch is
   the curve's argmin.
8. **Fused/eager agreement**, updated for `scan`.
9. **Leakage.** `train.py` imports nothing that reaches truth; the existing
   `leakage-check` arms stay bit-identical at detector level.

## Removed

`SelectionCriterion`, `--criterion`, `--patience`, `--min-delta`,
`RunCarry.best_score`/`best_epoch`/`wait`, `still_running`. The patience default
of 5 — which stopped the six-variable jet run at epoch 7 and restored epoch 1 —
ceases to exist rather than being documented.

## Validation before trusting it

In order, on the flagship configuration:

1. **Measure the floor.** One weight vector against ~10 independent val
   subsamples at m ∈ {4096, 8192, 16384}. Confirms the ~1/m scaling and fixes
   the resolution.
2. **Check the dynamic range.** Detector MMD across epochs against that floor.
   *This is the risk the detector-level decision was taken with open eyes:* the
   run already reaches 89–98% detector improvement, so the curve may be flat
   relative to the floor, in which case selection is choosing on noise. If so,
   raise m, or move to tiled-exact over the full 50k val rows, or reconsider the
   level.
3. **Compare against the BCE criteria** one last time, before deleting them, by
   evaluating all three against the same retained history. Records what the
   change bought.

## Rejected alternatives

- **In-loop MMD selection** (the original proposal). Requires either truth in
  the trace or the parameter history anyway; with the history, post-hoc is
  strictly simpler.
- **Hybrid** — in-loop MMD for early stopping, post-hoc for selection. Two
  mechanisms to save three seconds.
- **Tiled-exact MMD over the full val split.** No subsample floor, ~10 ms/eval,
  but more code. Held in reserve for validation step 2.
- **Random Fourier features.** Linear in n, the only option that survives
  `n_samples` growing 10×. Held in reserve; `weighted_mmd`'s interface is
  "weights → scalar", so it can replace the internals without touching callers.
