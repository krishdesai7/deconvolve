<!-- markdownlint-disable no-inline-html -->
# Precision & Hardware

<span style="font-variant: small-caps;">Deconvolve</span> is strictly single-precision end-to-end. This section explains how precision is enforced, how backend initialisation is handled, and how hardware divergence is prevented.

---

## The Float32 Pin

High Energy Physics calculations often mix double and single-precision arithmetic unpredictably. <span style="font-variant: small-caps;">Deconvolve</span> eliminates silent casts and memory bloat through a strict single-precision design:

1. **`EVENT_DTYPE`**: Pinned in `deconvolve.coretypes.constants` as `np.single`.
2. **`EventArray`**: Pinned in `deconvolve.coretypes.types` as `Float[Array, "N D"]` with `beartype` validation.
3. **`JAX_ENABLE_X64=0`**: Set before JAX is imported, preventing silent upcasting to float64.

---

## Backend Bootstrapping (`deconvolve/__init__.py`)

<span style="font-variant: small-caps;">Keras 3</span> and JAX inspect environment variables once, at the exact moment they are imported. To ensure consistent behaviour, `deconvolve/__init__.py` sets the following defaults before any other submodule is loaded:

```python
os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("JAX_ENABLE_X64", "0")
```

If a user script imports `keras` or `jax` before importing `deconvolve`, `deconvolve.training.engine` detects this and raises an informative `RuntimeError` rather than failing cryptically inside an XLA trace.

---

## Deterministic Matrix Multiplication (`HIGHEST`)

Modern accelerators (NVIDIA Ampere/Hopper/Blackwell) run matrix multiplications at reduced precision (TF32) by default, trading numerical accuracy for throughput.

For synthetic Gaussian sampling and covariance decomposition, this hardware-dependent precision causes the same seed to produce different random numbers on different machines (e.g. login node CPU vs compute node A100).

To make datasets hardware-invariant:

```python
jax.lax.Precision.HIGHEST
```

is explicitly set during sample generation and Cholesky smearing. A dataset drawn with a given seed produces the exact same bitwise numbers on an Apple CPU as on an NVIDIA GPU.

---

## Where Double Precision Still Lives

Double precision is preserved in exactly two non-training locations:

1. **Host-Side Bin Edges**: `np.linspace` builds histogram bin edges on the host in single precision so device tracing never re-rounds them.
2. **Integrity Assertions**: Verifying that event counts match exact integers and that weights normalise to mean 1 accumulates in double precision, because single precision ceases to represent consecutive integers beyond \(2^{24} \approx 1.67 \times 10^7\).
