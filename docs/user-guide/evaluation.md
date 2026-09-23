<!-- markdownlint-disable no-inline-html -->
# Evaluation & Metrics

<span style="font-variant: small-caps;">Deconvolve</span> evaluates the quality of reweighted distributions using three complementary distance metrics, comparing MC against nature distributions before and after reweighting.

---

## The Three Distance Metrics

All metrics are evaluated across every feature dimension on the held-out test split:

### 1. Wasserstein–1 Distance

The earth mover's distance:

\[\mathcal{W}_1(p, q) = \int \left\vert\int_{-\infty}^t \lbrack p(x) - q(x) \rbrack \, \d x\right\vert \, \d t\]

In <span style="font-variant: small-caps;">Deconvolve</span>'s implementation, the two CDFs are intentionally not accumulated separately. Accumulating them separately causes catastrophic cancellation when subtracting two numbers near 1. Instead, signed weights are accumulated in a single scan, keeping the running value at the size of the answer while maintaining single-precision accuracy.

### 2. Jensen–Shannon Divergence

The JS Divergence is a symmetrized, bounded version analog to the Kullback-Leibler divergence:

\[\text{JSD}(p \parallel q) = \frac{1}{2} D_{\text{KL}}(p \parallel m) + \frac{1}{2} D_{\text{KL}}(q \parallel m)\]

where \(m = \frac{1}{2}(p + q)\). The JS Divergence is computed across uniform bins over the combined feature range.

### 3. Vinze–LeCam Divergence (Triangular Discriminator)

A symmetric \(f-\)divergence with desirable numerical properties near zero. The Vinze--LeCam Divergence is defined as:

\[\Delta(p, q) = \int \frac{(p(x) - q(x))^2}{p(x) + q(x)} \, \d x\]

---

## On-Device Vectorized Evaluation

Every metric evaluation in `deconvolve.evaluation.evaluate` runs directly on the JAX accelerator device:

- **Vectorized over dimensions**: A single JAX dispatch computes the metrics across all dimensions simultaneously.
- **Minimal host transfer**: Only the final scalar distance values cross back from the device to the host.
- **Performance**: Evaluates 100k-vs-100k samples in 6 dimensions in ~0.28 seconds.

---

## Output: `metrics.json`

Running `deconvolve evaluate` writes a structured `metrics.json` file inside the run directory:

```json
{
  "unweighted": {
    "wasserstein_1d": [0.354, 0.289],
    "js_divergence": [0.048, 0.039],
    "triangular": [0.092, 0.075]
  },
  "reweighted": {
    "wasserstein_1d": [0.012, 0.009],
    "js_divergence": [0.0008, 0.0006],
    "triangular": [0.0015, 0.0011]
  }
}
```
