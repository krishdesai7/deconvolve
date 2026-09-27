<!-- markdownlint-disable no-inline-html -->
# Saturation & Diagnostics

A central question in adversarial unfolding is: which hyperparameters matter?

Extensive diagnostic experiments in Deconvolve (`benchmarks/ceiling.py`, `benchmarks/tilt.py`, `benchmarks/response.py`) established that the detector–level objective is fundamentally saturated, and that detector–level metrics cannot uniquely identify particle–level truth.

---

## 1. The Detector-Level Objective is Saturated

Before reweighting, a converged classifier easily separates simulated events \(x_{\text{sim}}\) from observed data \(x_{\text{data}}\) by 14.786 mnat.

After reweighting, a fresh, independently trained classifier finds only 0.087 mnat of remaining discrepancy. <span style="font-variant: small-caps;">Deconvolve</span> removes 99.4% of the available mismatch.

!!! note
    There is virtually no signal left at detector level for a larger discriminator, a deeper generator, or an alternative optimizer to find. The objective is saturated.

---

## 2. Detector-Level Objectives Do Not Identify Truth

Consider scoring the oracle weight function \(w^*(z)\), i.e. the true particle–level likelihood ratio fitted directly on unobserved truth, against <span style="font-variant: small-caps;">Deconvolve</span>'s weights using held-out detector–level and particle–level MMD:

| Weights             |       Detector MMD²          |     Particle MMD²        | Effective Sample Size (ESS) |
| :------------------ | :------------------------:   | :--------------------:   | :-------------------------: |
| **Unweighted**      |   \(3.96 \times 10^{-2}\)    | \(5.90 \times 10^{-2}\)  |            100%             |
| **Oracle $w^*(z)$** |   \(+8.02 \times 10^{-4}\)   | \(-1.90 \times 10^{-4}\) |            80.1%            |
| **RAN**             | **\(-2.32 \times 10^{-4}\)** | \(+4.58 \times 10^{-3}\) |            73.3%            |

Notice that <span style="font-variant: small-caps;">RAN</span> scores better than the truth on the detector–level criterion, despite scoring worse at particle–level.

This is not overfitting or noise: the detector response \(p(x \mid z)\) is many-to-one. The particle–level likelihood ratio pushed through detector resolution is not identical to the detector–level likelihood ratio.

!!! warning "Consequence"
    No truth–free criterion constructed purely on detector–level agreement can rank the true particle–level distribution first. Sharpening the detector–level estimator makes the preference more confident, not more correct.

---

## 3. Capacity vs. Objective

To test whether network capacity limits performance, `benchmarks/tilt.py` replaced the neural generator with an exponential tilt family \(w(z; b) = \exp(-b \cdot T(z))\) fitted by convex moment-matching without an adversary:

| Method                       | Parameters | Particle Agreement | Detector Agreement |
| :--------------------------- | :--------: | :----------------: | :----------------: |
| **Tilt (Degree 1)**          |   **6**    |       +78.5%       |       +92.5%       |
| **Tilt (Degree 2)**          |     27     |       +77.0%       |       +94.8%       |
| **RAN (Neural Network)**     |  ~34,000   |       +78.9%       |       +92.1%       |
| **Oracle (Fitted on Truth)** |     —      |     **+93.2%**     |       +82.8%       |

Six parameters achieve the same particle–level performance as 34,000 neural network parameters.

Furthermore, Degree 2 improves detector–level agreement (92.5% \(\to\) 94.8%) while degrading particle–level accuracy (78.5% \(\to\) 77.0%) in a deterministic, convex solve. This demonstrates that performance limits stem from the physics of detector folding, not optimization failure.
