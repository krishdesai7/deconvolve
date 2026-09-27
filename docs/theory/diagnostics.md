<!-- markdownlint-disable no-inline-html -->
# Saturation & Diagnostics

This page summarizes a set of diagnostic studies to determine the theoretical limits of <span style="font-variant: small-caps;">Deconvolve</span>'s particle–level accuracy on the jet benchmark. The question they address is practical: before tuning hyperparameters, how much improvement is available at all, and would a better score on the training objective translate into a better unfolded result?

The findings are that the detector–level objective is essentially saturated, and that it does not identify the particle–level truth. The remaining particle–level error is therefore a property of the problem, not of the optimizer or the network capacity. The jet mass is further limited by a detector response that differs between the two generators.

Unless stated otherwise, results use the six observables of the <span style="font-variant: small-caps;">OmniFold</span> [study](https://doi.org/10.1103/PhysRevLett.124.182001) \((m, M, w, \tau_{21}, z_g, \ln\rho)\), from the jet dataset (see [Datasets](../user-guide/datasets.md)), with \(10^6\) events. Section 3 also considers all twelve observables. Every number is reproducible with the scripts in `benchmarks/`, which are documented in `benchmarks/README.md`. Several of these diagnostics fit networks directly on \(z_\text{true}\). That is legitimate only because their purpose is to measure the method against the truth, and it is why they live outside the `deconvolve` package.

---

## 1. The detector–level objective is saturated

The discrepancy between two samples can be measured by training a classifier to separate them. For balanced classes, \(\ln 2 - \mathcal{L}_\text{BCE}\), evaluated on held-out events with a converged classifier, is a lower bound on the Jensen–Shannon divergence between the two distributions.

| Comparison | \(\ln 2 - \mathcal{L}_\text{BCE}\) \[nat\] |
| :--- | ---: |
| \(x_\text{data}\) vs. unweighted \(x_\text{sim}\) | \(1.48 \times 10^{-2}\) |
| \(x_\text{data}\) vs. \(x_\text{sim}\) reweighted by <span style="font-variant: small-caps;">Deconvolve</span> | \(8.7 \times 10^{-5}\) |

The second classifier is trained from scratch against the frozen generator's weights (`ceiling.py`, diagnostics A and C). The reweighting removes 99.4% of the measurable detector–level discrepancy. Repeating with other initialisation seeds gives residuals of \(1.7 \times 10^{-4}\) and \(1.8 \times 10^{-4}\) nat, so at least 98.8% in every case.

Little discrepancy therefore remains at detector–level for a larger discriminator, a different generator architecture or a different optimiser to remove. Further gains on the training objective are bounded by this residual.

---

## 2. The detector–level criterion does not rank the truth first

A second question is whether, among weight functions that all match the data at detector level, the selection criterion prefers the one closest to the truth. To test this, the oracle weight function \(w^*(z)\), the particle–level likelihood ratio between \(z_\text{true}\) and \(z_\text{gen}\), is fit directly on truth. It is then scored alongside <span style="font-variant: small-caps;">Deconvolve</span>'s weights on the held-out test split, with the same weighted MMD used for [model selection](mmd.md) (`ceiling.py`, diagnostic D):

| Weights | Detector–level \(\widehat{\text{MMD}^2}\) \[nat\] | Particle–level \(\widehat{\text{MMD}^2}\) \[nat\] | ESS |
| :--- | ---: | ---: | ---: |
| Unweighted | \(3.96 \times 10^{-2}\) | \(5.90 \times 10^{-2}\) | 100% |
| Oracle \(w^*(z)\) | \(+8.02 \times 10^{-4}\) | \(-1.90 \times 10^{-4}\) | 80.1% |
| <span style="font-variant: small-caps;">Deconvolve</span> | \(-2.32 \times 10^{-4}\) | \(+4.58 \times 10^{-3}\) | 73.3% |

(The unbiased estimator scatters about zero when the distributions agree, so small negative values indicate agreement within noise.)

The oracle matches the truth at particle–level, as it must, but leaves a clear residual at detector–level. <span style="font-variant: small-caps;">Deconvolve</span> matches the data at detector–level, on events it never saw, but leaves a clear residual at particle–level, where the oracle’s is consistent with zero. Each weight function achieves what it was fit to achieve, and no single weight function achieves both.

This is expected if the detector response differs between the two generators. Reweighting Simulation assumes that the response \(r(x \mid z)\) of the MC also holds for the Nature. If it does not, then the true particle–level distribution, folded through the MC response, does not reproduce the observed detector–level Data. Any method that achieves detector–level closure must then settle on a particle–level distribution other than the truth. Section 3 shows that the response does differ, and by how much.

!!! warning "Consequence"
    No selection criterion based on detector–level agreement alone can rank the true particle–level distribution first, since the truth scores worse on it. A more precise estimate of detector–level agreement makes the criterion more confident in its ranking, not more correct.

---

## 3. The jet mass response is not universal

The response is *universal* if there is a shared response kernel \(r(x \mid z)\) for both generators. Labelling each event with its generator \(S \in \{P, H\}\), this is equivalent to the conditional independence criterion, \(X \perp S \mid Z\). Its violation is measured by the conditional mutual information \(I(S; X \mid Z)\), which equals the difference in binary cross-entropy between Bayes-optimal classifiers of \(S\) trained on \(z\) alone and on \((z, x)\) (`response.py`). Because learned classifiers fall short of Bayes-optimal, the estimate is a lower bound.

For the detector--level jet mass \(X_m\), conditioning on progressively more of the particle--level event:

| Conditioning variables \(Z\) | \(I(S; X_m \mid Z)\) \[mnat\] | Relative to \(Z = m\) |
| :--- | ---: | ---: |
| \(m\) | \(9.01 \pm 0.46\) | — |
| \(m\), \(\lambda^1_2\) | \(4.49 \pm 0.33\) | \(-50\%\) |
| six <span style="font-variant: small-caps;">OmniFold</span> observables | \(2.37 \pm 0.32\) | \(-74\%\) |
| all twelve observables | \(1.62 \pm 0.38\) | \(-82\%\) |

As a control, splitting a single generator into two pseudo-samples with different \(p(z)\) but identical \(p(x \mid z)\) gives values consistent with zero: \((-3.3 \pm 7.8) \times 10^{-1}\) mnat for <span style="font-variant: small-caps;">Herwig</span> and \((-4.6 \pm 3.3) \times 10^{-1}\) mnat for <span style="font-variant: small-caps;">Pythia</span>.

The detector–level mass therefore carries information about which generator produced the event beyond what the particle–level mass does. Conditioning on more of the particle–level event reduces this dependence substantially but does not remove it. The response to the jet mass alone is not universal. Part of the reason is that it averages over particle–level information, such as the radiation pattern constrained by \(\lambda^1_2 \approx \frac{m^2}{(p_T R)^2}\), whose distribution differs between the generators.

The same conclusion follows from applying the oracle to each observable in isolation. The fraction of each observable's detector–level \(\widehat{\text{MMD}^2}\) that the oracle leaves unremoved is:

| Observable | Residual after oracle reweighting |
| :--- | ---: |
| \(m\) | 45.3% |
| \(f_{ch}\) | 12.0% |
| \(M\) | 5.6% |
| \(p_T^D\), \(n_{ch}\), \(\tau_{21}\), \(\ln\rho\) | \(\le 3.5\%\) |
| \(w\), \(z_g\), \(\lambda^1_2\), \(\lambda^1_{0.5}\), \(q\) | 0.0% |

For five observables, reweighting at particle–level explains the detector–level discrepancy completely, as a universal response requires. The jet mass is the clear exception. It is also the only observable whose detector–level distribution carries substantially more information about the generator than its particle–level distribution does (a ratio of 3.7, against at most 1.2 for every other observable), and reweighting \(z\) can remove only the particle–level part.

---

## 4. Network capacity does not limit the result

If the particle–level error were due to insufficient capacity or a poor optimum, a more flexible or better-optimized generator should reduce it. `tilt.py` tests the opposite extreme. It replaces the neural generator with an exponential family,

\[w(z; b) = \exp\left(-b \cdot T(z)\right),\]

whose parameters \(b\) are found by matching detector–level moments. With \(T(z)\) the observables themselves (degree 1), this matches first moments. With all pairwise products added (degree 2), it also matches second moments. The fit is a convex root-finding problem with a unique solution, with no adversary, no stochastic training and no epoch selection.

Agreement is the mean, over the six observables, of the improvement in Wasserstein distance relative to unweighted simulation (see [Evaluation & Metrics](../user-guide/evaluation.md)):

| Method | Parameters | Particle–level agreement | Detector–level agreement |
| :--- | ---: | ---: | ---: |
| Tilt, degree 1 | 6 | +78.5% | +92.5% |
| Tilt, degree 2 | 27 | +77.0% | +94.8% |
| <span style="font-variant: small-caps;">Deconvolve</span> | ~34,000 | +78.9% | +92.1% |
| Oracle \(w^*(z)\) | — | +93.2% | +82.8% |

Across more than three orders of magnitude in the number of parameters, every method fitted at detector–level reaches 77–79% at particle–level. The oracle, fitted on truth, reaches 93.2%, at the cost of detector–level agreement as described in Section 2. The gap of roughly 15 percentage points is the price of fitting without truth, and it is common to all of these methods.

The degree-2 tilt shows this most directly. Its 21 additional parameters improve the fitted, detector–level agreement from 92.5% to 94.8%, while particle–level agreement falls from 78.5% to 77.0%. The solve is convex and deterministic, so this cannot be attributed to the optimizer, to training noise or to epoch selection. Many weight functions match the data at detector–level, and they differ at particle–level in directions that detector–level data do not constrain. Which of them a method returns is determined by its inductive bias, not by the data.

---

## Implications

- **Tuning.** Because the detector–level objective is saturated and does not identify the truth, hyperparameters mainly change *which* detector–equivalent solution is found, not how good it is. In practice, a change can improve some observables at the expense of others at nearly constant aggregate performance. `benchmarks/README.md` documents such a trade-off for the generator learning rate.
- **Automatic model selection.** An automated search that maximizes a detector–level criterion will follow these trade-offs without regard to which observables matter physically.
- **Jet mass.** The jet mass is limited separately by the non-universal response of Section 3, which no reweighting of particle–level simulation can correct.

## References

A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman and J. Thaler, "OmniFold: A Method to Simultaneously Unfold All Observables", *Phys. Rev. Lett.* **124** (2020) 182001.
