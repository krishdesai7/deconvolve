<!-- markdownlint-disable no-inline-html -->
# Evaluation & Metrics

`deconvolve evaluate` measures how well a trained generator's weights close the gap between simulation and data. It compares the two samples before and after reweighting, per observable, using three distance metrics, and at both detector and particle level.

```shell
deconvolve evaluate runs/2026-09-10T132409Z        # one run
deconvolve evaluate runs                           # every run under runs/
deconvolve evaluate runs/2026-09-10T132409Z --force
```

A directory is treated as a run if it contains a `config.json`. When `RUN_DIR` isn't a run itself, every run directly inside it is evaluated in turn, and a run that fails is logged and skipped rather than stopping the rest. A run that already has `artifacts/metrics.json` is skipped unless you pass `--force`.

`deconvolve train` evaluates the run that was just trained, so `evaluate` is only needed to re-score a run, or to score one whose metrics were deleted.

---

## What is compared

Evaluation reloads the generator from `artifacts/generator.keras` and rebuilds the dataset from `config.json`. It uses the recorded `data_seed`, so the **test split** is the same one the run held out during training. Every metric is computed on that test split only.

The generator's weights \(w = g(z_\text{gen})\) are computed on the test split's Generation events and normalized to mean 1. Then, at each level:

| Level | Reference (Nature) | Compared sample (MC) |
| :--- | :--- | :--- |
| `detector` | \(x_\text{data}\) | \(x_\text{sim}\) |
| `particle` | \(z_\text{true}\) | \(z_\text{gen}\) |

- **Before** compares the reference against the simulation unweighted.
- **After** compares it against the simulation reweighted by \(w\). The reference is never reweighted.

"Detector level" measures the performance of the algorithm in ensuring that reweighted Simulation indistinguishable from Data. "Particle level" evaluates the unfolded Generation against Truth. It is the only place \(z_\text{true}\) is ever read. That's possible for the datasets described in [Datasets](datasets.md) because the benchmark datasets have a known truth, and it is an after-the-fact score. No network ever sees \(z_\text{true}\), which can be validated using `deconvolve leakage-check --poison`.

Each observable (each column) is scored separately. For jets, the columns are the observables. For Gaussian data, they are `dim_0`, `dim_1`, and so on.

---

## Distance metrics

For all metrics, lower is better, and 0 means the two distributions agree identically.

### 1. Wasserstein–1 distance

The earth mover's distance, computed exactly from the two samples, with no binning:

\[\mathcal{W}_1(p, q) = \int_{-\infty}^{\infty} \left\vert \int_{-\infty}^x \lbrack p(t) - q(t) \rbrack \, \d t\right\vert \, \d x\]

This is the same estimator as `scipy.stats.wasserstein_distance`, implemented as a fast, fused, vectorized JAX operation. It is in the observable's own units, and therefore cannot be compared across observables.

The two CDFs are not accumulated separately. Each climbs to 1 while their difference stays small, so subtracting them afterwards would cancel away most of a float32 mantissa. Instead, the signed weights of both samples are accumulated in a single scan, which keeps the running value at the size of the answer.

### 2. Jensen–Shannon divergence

A symmetrized, bounded counterpart of the Kullback-Leibler divergence:

\[\text{JSD}(p \parallel q) = \frac{1}{2} D_{\text{KL}}(p \parallel m) + \frac{1}{2} D_{\text{KL}}(q \parallel m), \qquad m = \frac{1}{2}(p + q)\]

It is computed in \(\text{nats}\) from histograms, and is bounded above by \(\ln 2\). This is the square of the Jensen-Shannon _distance_ computed by `scipy.spatial.distance.jensenshannon(p, q)`.

### 3. Vincze–Le Cam divergence (triangular discriminator)

A symmetric \(f-\)divergence that behaves well when the two distributions are close:

\[\Delta(p, q) = \sum_i \frac{(p_i - q_i)^2}{p_i + q_i}\]

It is computed from the same histograms as the JS divergence, with bins where both are empty skipped.

### Display scale

`metrics.json` stores all three metrics exactly as defined above. Their raw values are small. Therefore, user facing values are multiplied by \(10^3\) for readability. Improvement percentages are ratios, so the scale doesn't affect them.

### Binning

Both divergences use **100 uniform bins** per observable. The bin range covers the combined range of the two samples being compared, so both histograms share identical edges. Each histogram is normalized to unit mass before the divergence is computed.

---

## On-device evaluation

The expensive part of every metric runs on the JAX device (the accelerator, if there is one):

- **Vectorized over observables:** one dispatch computes every column at once.
- **Little host transfer:** the Wasserstein distances come back as one number per observable, and the histograms as `observables × 100` bin counts. The two divergences are then computed from those histograms on the host, in double precision.
- **Performance:** measured at 100k nature and 100k MC events in 6 dimensions, about 0.28 s of compute, plus a one-time XLA compilation.

---

## Output

### `artifacts/metrics.json`

Metrics are written to `RUN_DIR/artifacts/metrics.json`. There is one entry per level and observable, keyed `<level>_<observable>`. Detector-level entries come first, then particle-level ones, each in the column order of `config.json`'s `variables`. For example, for a jet run:

```json
{
  "detector_m": {
    "wasserstein_before": 0.20894503593444824,
    "wasserstein_after": 0.011569535359740257,
    "wasserstein_improvement_pct": 94.46288096388663,
    "jensenshannon_before": 0.006761803014036163,
    "jensenshannon_after": 9.511884587269733e-05,
    "jensenshannon_improvement_pct": 98.59329167567807,
    "triangular_before": 0.02693145105189114,
    "triangular_after": 0.0003598454484998638,
    "triangular_improvement_pct": 98.66384678713925
  },
  "detector_M": { "...": "..." },
  "particle_m": { "...": "..." },
  "particle_M": { "...": "..." }
}
```

Every entry has the same nine fields:

| Field | Meaning |
| :--- | :--- |
| `<metric>_before` | Distance between the reference and the unweighted simulation. |
| `<metric>_after` | Distance between the reference and the reweighted simulation. |
| `<metric>_improvement_pct` | \(\left(1 - \frac{\text{after}}{\text{before}}\right) \times 100\%\). Positive values mean that reweighting helped. `0` if `before` is `0`. |

where `<metric>` is `wasserstein`, `jensenshannon` or `triangular`. Values are unscaled; see [Display scale](#display-scale).

The baselines write their scores in exactly the same format, so the three are directly comparable: `deconvolve baseline ibu` writes `artifacts/metrics_ibu.json`, and `deconvolve baseline omnifold` writes `artifacts/metrics_omnifold.json` (see [Baselines](baselines.md)). `deconvolve report` reads all three into the run's PDF (see [Reporting & Artifacts](reporting.md)).

### Terminal summary

`evaluate` also prints one table per level, with a Wasserstein, JS and \(\Delta\) row for each observable, all multiplied by \(10^3\) for readability:

```text
             2026-09-10T132409Z — Detector level
┏━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Variable ┃ Metric      ┃   Before ┃   After ┃ Improvement ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━━━━┩
│ m        │ Wasserstein │ 208.9450 │ 11.5695 │      +94.5% │
│          │ JS div      │   6.7618 │  0.0951 │      +98.6% │
│          │ Delta       │  26.9315 │  0.3598 │      +98.7% │
└──────────┴─────────────┴──────────┴─────────┴─────────────┘
                     All distances x1000
```
