<!-- markdownlint-disable no-inline-html -->
# Precision & Hardware

<span style="font-variant: small-caps;">Deconvolve</span> stores events and trains its networks in single precision (float32). Double precision is used only in a few places before and after training, listed at the end of this page. This page also describes how the numerical backend is configured, and what to expect when the same run is repeated on different hardware.

---

## Single precision

The event dtype is defined once, as `EVENT_DTYPE = np.single` in `deconvolve.coretypes.constants`, with the corresponding array type `EventArray` in `deconvolve.coretypes.types`. The network layers are declared float32, and 64-bit JAX arrays are disabled (`JAX_ENABLE_X64=0`), so values are not silently promoted to float64 during training. There is no option to change the dtype.

Single precision is sufficient for this problem. The jet observables lose at most half a unit in the last place when stored as float32. In a comparison over 320 paired seeds, float32 and float64 training gave indistinguishable unfolding performance: the difference between them was smaller than the variation between seeds within either precision (`benchmarks/precision.py`, `benchmarks/compare_precision.py`).

Each of the three data sources (the Gaussian generator, the jet loader and `leakage-check`) converts its output to float32 explicitly. The dtype is checked twice: by the type checkers when code is written, and at run time when a source builds its events into a `Populations`, which rejects any array that is not float32 and two-dimensional (see [Data Model](data-model.md)).

---

## Backend configuration

Keras and JAX read their configuration from environment variables once, when they are first imported. Importing `deconvolve` therefore sets two defaults before loading any of its submodules:

```python
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "0")
```

These are defaults: a value already present in the environment takes precedence. If Keras has already been imported with a different backend, importing `deconvolve.training.engine` raises a `RuntimeError` that explains the problem, rather than failing later inside a compiled function.

---

## Matrix multiplication precision

On recent NVIDIA GPUs (Ampere and later), XLA performs float32 matrix multiplications in TF32 by default, which rounds the inputs to a 10-bit mantissa. Where this would visibly change a result, <span style="font-variant: small-caps;">Deconvolve</span> requests full float32 precision (`"highest"`):

- **Gaussian datasets.** Drawing correlated Gaussian samples involves matrix products. At TF32 these would differ from a CPU draw by about \(5 \times 10^{-4}\) relative, so a dataset cached on one machine would not match the same configuration drawn on another.
- **MMD model selection.** The [MMD](../theory/mmd.md) is assembled from terms much larger than the result, so TF32 rounding of the inputs would dominate it.

Full precision removes the dominant hardware dependence. It does not make results bitwise identical across CPU and GPU, whose floating-point libraries may still round differently in the last place.

---

## Reproducibility on GPUs

On a CPU, repeating a computation gives identical results. On a GPU, a few reductions accumulate in an order that varies between executions. The most visible case is the histogram filling used for the [Jensen–Shannon and Vincze–Le Cam divergences](../user-guide/evaluation.md): evaluating the same run twice on a GPU can change these metrics by about \(4 \times 10^{-8}\) relative, which is visible in the last digit that `metrics.json` records. Compare results from different evaluations with a tolerance rather than for exact equality.

Where bitwise reproducibility is required, setting `XLA_FLAGS=--xla_gpu_deterministic_ops=true` enforces a deterministic order, at some cost in throughput.

---

## Double precision

Double precision is used in the following places, none of which is part of training:

- **Computing the jet observables.** When the jet dataset is first downloaded, the observables are computed from the Zenodo release in float64 and cached as float64. Some of these computations guard degenerate jets with a small constant that lies below the float32 range. The data are converted to float32 once, when they are loaded for a run.
- **Dataset configuration.** The means and covariances of a Gaussian configuration are parsed, and validated by Cholesky decomposition, in float64.
- **Evaluation metrics.** The distance metrics are reported in float64. The expensive reductions over the full sample run in float32 on the device, and are arranged so that their rounding error is relative to the result rather than to the largest intermediate value. The Wasserstein distance accumulates the signed difference of the two samples' weights rather than two separate CDFs. The histograms accumulate each weight's deviation from the mean, with the mean added back through the integer bin count, which float32 represents exactly up to \(2^{24} \approx 1.7 \times 10^7\) events per bin. The divergences are then computed from the histograms on the host in float64.
- **Uncertainty decomposition.** The variance components of an [uncertainty design](uncertainty.md) are computed in float64 on the host.

The histogram bin edges are the one deliberate exception in the other direction. They are computed on the host but kept in float32, because the device converts float64 inputs to float32. Edges computed in float64 would be rounded differently from the values being binned, and an event lying on a bin boundary could be assigned to the neighbouring bin.
