# Precision

**RAN is float32 end to end, and the pin lives in one place:**
`EVENT_DTYPE` in `src/deconvolve/coretypes/constants.py`, with its annotation-space
twin `EventArray` in `coretypes/types.py`. `JAX_ENABLE_X64=0` and the `dtype=`
arguments in `src/deconvolve/training/models.py` follow from it. There is no dtype parameter
anywhere in the pipeline and no `astype` on the containers; there used to be
generics (`Events[T]`, `Populations[T]`, …) whose only purpose was letting IBU
carry float32 through a float64 pipeline, and with one dtype they were
ceremony.

The evidence, because this is the kind of decision that gets re-litigated:

- Every jet observable is float32-clean. `mass` and `mult` are bit-exact
  through a float32 round trip; `w`, `tau21`, `zg` and `sdm` lose exactly half
  a ULP, the minimum a cast can cost. There is no structure below float32.
- Weight normalization is stable in float32 out to a 10²⁴ weight dynamic range
  (error ~1e-8), and the batched scan reduction lands within 5e-4 of
  `min_delta`, because summing 8192-element batches then 61 partials is
  effectively pairwise summation.
- 320 paired seeds put float32 and float64 within 3.5 sigma of each other on
  unfolding improvement — indistinguishable, and the seed-to-seed spread
  within either precision is larger than the gap between them. See
  `benchmarks/precision.py` and `benchmarks/compare_precision.py`.

The pin is enforced twice. The type checkers enforce `EventArray` at author
time, and `Populations.__post_init__` checks at run time that all four arrays
are `EVENT_DTYPE` and 2-D, with `truth` matching `mc.z` and `data` matching
`mc.x` in feature count. That check is the one runtime boundary: every source
builds a `Populations`, and each of the three (`_draw_gaussian`,
`load_jet_dataset`, and the sample construction in `workflows/leakage.py`)
narrows explicitly before doing so. It validates rather than coerces, so a
source that forgot to narrow fails where it built its events instead of deep
inside a jitted function. Test doubles are held to the same contract: a
float64 fixture fails there too. `EventArray` stays `NDArray[np.single]`
rather than a jaxtyping `Float[Array, ...]`: it annotates host NumPy arrays
of either rank (event matrices, single columns, weight vectors), and jaxtyping's
`Float` admits any float dtype and is opaque to the static checkers.

One thing the pin does **not** cover:

- **`deconvolve.data.download` stays float64 on purpose.** `_get_var` upcasts before
  computing observables, because the ε it uses to protect degenerate jets is
  below the smallest float32 denormal — narrowing there would hand back `NaN`
  for exactly the jets the ε exists to protect. The narrowing happens after, in
  `load_jet_dataset`.

Five gotchas worth knowing:

- **Scores are not pinned.** Wasserstein, JS and the triangular discriminator
  are float64 and stay there. What is pinned is the data, not the measurement
  taken of it. The reductions over the full sample in `deconvolve.evaluation.evaluate` — the
  sort-and-scan behind Wasserstein, the histogram scatter behind the other
  two — are necessarily float32, so each is arranged so its error is relative
  to the answer rather than to the largest intermediate: the Wasserstein scan
  accumulates the _signed_ weights, whose running total is the CDF gap being
  measured, instead of two CDFs that both climb to 1 and then cancel; the
  histogram scatters _centered_ weights and adds the mean back through an
  exact count. Everything downstream — the divergences themselves, reductions
  over `dim x n_bins` values — is float64 on the host. Against a float64
  reference this lands JS within 9e-9 on a CPU and ~1.3e-8 on an A100.
  **The gap is platform-dependent, so do not pin a measured constant as a
  tolerance.** The scatter accumulates in float32 and the order is the
  hardware's choice; the same assertion that holds at 1.26e-8 locally returned
  1.288e-8 on the cluster. Bound these against what `metrics.json` prints —
  a tenth of the last printed digit — not against the last measurement, which
  is what `TestFloat32Histograms` does. The number is also a statement about
  _bias_, and on a GPU it is smaller than the run-to-run noise — see the next
  bullet.
- **`metrics.json` is reproducible to ~4e-8 on a GPU, not to the last digit.**
  `_counts` bins with `empty.at[index].add(...)`, which lowers to a scatter-add;
  many events land in one bin, so on a GPU that is an _atomic_ accumulation and
  the summation order is whatever the hardware chose that pass. Two
  `deconvolve evaluate` runs over the same run directory therefore return histogram
  counts differing in the last float32 ulp, and JS values differing by ~4e-8
  relative — measured, non-systematic: it moves up on some dimensions and down
  on others. On a CPU the scatter is sequential and the numbers repeat
  exactly, which is why this only ever appears on the cluster. It is a
  deliberate trade against a sorted segment-sum or a one-hot matmul over the
  full sample for a reduction that is otherwise free. Two consequences.
  `metrics.json` prints six decimals and the sixth is not stable on a GPU, so
  a diff of two evaluations of the same run is expected to be non-empty;
  compare with a tolerance rather than by equality. And a test must never
  build the same histogram twice and compare the halves at a tight
  tolerance — that is a determinism assertion wearing a divergence's clothes, and
  it is what
  `tests/test_evaluate_metrics.py::TestDivergencesPerDim::test_js_matches_scipy_on_a_continuous_sample`
  did until it started failing on the A100 and passing locally.
  Build the histograms once, hand the same pair to both sides. Where that is
  impossible because the double binning _is_ the claim — `TestFusedMetrics`
  asks whether the fused and unfused paths agree — widen the tolerance
  instead and say why: those compare at `rtol=1e-6`, since the noise has been
  measured at 1.05e-7 relative and `assert_allclose`'s default `rtol` is
  1e-7, which put them right on the line (a coin flip on the cluster, a
  certainty on a CPU). If bitwise reproducibility is ever actually needed,
  `XLA_FLAGS=--xla_gpu_deterministic_ops=true` buys it at a throughput cost
  (the same flag [seeding.md](seeding.md) mentions).
- **`np.float32` is not JSON-serializable.** `np.float64` subclasses Python
  `float`, so `json` accepts it silently; `np.float32` raises. Anything
  writing numbers to JSON has to coerce first — see
  `deconvolve.evaluation.evaluate._metric_entry`, which puts every value through
  `float()` on the way into `metrics.json` for exactly this reason.
- **`keras.ops.mean` is not float64-safe.** For float64 input it selects a
  float32 compute dtype internally and returns a float64 result carrying ~1e-8
  relative error. `src/deconvolve/training/engine.py` uses plain `jnp` and never touches it,
  reducing with `jnp.sum(...) / n` instead, which `tests/test_train.py`
  guards. Anything that reaches for `keras.ops` again needs to know. `ops.sum`
  is unaffected.
- **JAX preallocates ~75% of GPU memory on its first device allocation.**
  Nothing in the package pins itself to CPU to avoid this. On a shared node,
  `scripts/submit_uncertainty.zsh` and `submit_hparam.zsh` give each cell
  exactly one visible GPU via `srun --gpus-per-task=1`, so the first step to
  start cannot swallow the whole card.
