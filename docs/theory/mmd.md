<!-- markdownlint-disable no-inline-html -->
# MMD Model Selection

Training runs for a fixed number of epochs, and the parameters of every epoch are retained. After training, a single epoch is selected and its generator is used as the result. This page describes the criterion used for that selection.

The adversarial loss is not a suitable criterion. By construction it oscillates about its equilibrium value, so a flat loss curve cannot be distinguished from a stalled one. The quantity \(\ln 2 - \mathcal{L}\) estimates a divergence between the two samples only when the discriminator is optimal, and nothing in training certifies that it is. <span style="font-variant: small-caps;">Deconvolve</span> instead selects the epoch that minimizes the maximum mean discrepancy (MMD)[^gretton2012] between data and reweighted simulation. The MMD is a closed-form function of the weights: it involves no adversary and no optimization, and with a suitable kernel it vanishes if and only if the two distributions are equal.

Selection is performed at detector level, comparing \(x_\text{data}\) with \(x_\text{sim}\) reweighted by \(w(z_\text{gen})\). It therefore requires no particle-level truth, and the procedure applies unchanged to real data.

---

## Maximum mean discrepancy

For distributions \(P\) and \(Q\) and a positive-definite kernel \(k\), the squared MMD is

\[\text{MMD}^2(P, Q) = \mathbb{E}_{x, x' \sim P}\lbrack k(x, x')\rbrack - 2\, \mathbb{E}_{x \sim P,\, y \sim Q}\lbrack k(x, y)\rbrack + \mathbb{E}_{y, y' \sim Q}\lbrack k(y, y')\rbrack,\]

where \(x, x'\) and \(y, y'\) denote independent draws. It is the squared distance between the mean embeddings of \(P\) and \(Q\) in the reproducing kernel Hilbert space of \(k\). When \(k\) is characteristic, as a Gaussian kernel is, \(\text{MMD}(P, Q) = 0\) if and only if \(P = Q\).

---

## Weighted estimator

Let \(\{x_i\}_{i=1}^{n}\) be Data events, each with unit weight, and \(\{y_j\}_{j=1}^{m}\) Simulation events with generator weights normalized to unit sum,

\[\bar{w}_j = \frac{g(z_j; \beta)}{\sum_{l=1}^{m} g(z_l; \beta)}.\]

Each expectation is replaced by an average over distinct pairs:

\[
\begin{aligned}
\widehat{\text{MMD}}^2
&= \frac{1}{n(n-1)} \sum_{i \neq i'} k(x_i, x_{i'}) \\
&\quad - \frac{2}{n} \sum_{i=1}^{n} \sum_{j=1}^{m} \bar{w}_j\, k(x_i, y_j) \\
&\quad + \frac{1}{1 - \sum_j \bar{w}_j^2} \sum_{j \neq j'} \bar{w}_j \bar{w}_{j'}\, k(y_j, y_{j'}).
\end{aligned}
\]

The first term is the standard unbiased U-statistic. The last is the weighted mean of the kernel over distinct simulated pairs: since \(\sum_j \bar{w}_j = 1\), the pair weights \(\bar{w}_j \bar{w}_{j'}\) with \(j \neq j'\) sum to \(1 - \sum_j \bar{w}_j^2\). For uniform weights \(\bar{w}_j = 1/m\), it reduces to the usual U-statistic.

Two properties of this estimator matter in practice.

- **It can be negative.** Excluding the diagonal terms removes the estimator's bias but not its variance, so when the two distributions agree the estimate scatters about zero.
- **Weight concentration is reported, but not penalized.** The estimator is returned together with the effective sample size \(\text{ESS} = 1 / \sum_j \bar{w}_j^2\). A biased estimator that retains the diagonal terms would add a contribution proportional to \(\sum_j \bar{w}_j^2\), implicitly penalizing concentrated weights. Since the adversarial objective can favour exactly such concentration, it is measured separately so that it can be inspected rather than being mixed silently into the criterion. If \(1 - \sum_j \bar{w}_j^2\) falls below \(10^{-6}\) (effectively all weight on one event), the estimate is set to \(+\infty\) and that epoch cannot be selected.

---

## Kernel and bandwidths

The kernel is a sum of five Gaussian kernels:

\[k(x, x') = \sum_{s=1}^{5} \exp\left(-\frac{\lVert x - x' \rVert^2}{2\sigma_s^2}\right), \qquad \sigma_s = c_s\, \sigma_0, \quad c_s \in \left\lbrace \tfrac{1}{2},\ \tfrac{1}{\sqrt{2}},\ 1,\ \sqrt{2},\ 2 \right\rbrace.\]

A single bandwidth is sensitive to discrepancies at roughly one length scale. A sum of Gaussian kernels is still characteristic, so bracketing the central bandwidth widens the range of scales at no cost to the MMD's properties.

The central bandwidth \(\sigma_0\) follows the median heuristic, chosen so that the kernel equals \(e^{-1}\) at the median squared distance between data events:

\[\sigma_0 = \sqrt{\tfrac{1}{2}\, \operatorname{median}_{i, i'} \lVert x_i - x_{i'} \rVert^2}.\]

The median is taken over the data sample only. The data subsample is fixed by `data_seed`, so every run with the same `data_seed` (for example, every arm of a hyperparameter scan) is scored with an identical kernel. A heuristic computed on the pooled sample would shift with the simulated side and make runs incomparable.

---

## Selection procedure

1. **Subsamples.** From the validation split, draw \(n = 16384\) data events and \(m = 16384\) simulated events (fewer if the split is smaller), without replacement. Both draws are seeded from `data_seed`.
2. **Kernel.** Compute the bandwidths from the data subsample.
3. **Score.** Evaluate the generator from every retained epoch on the simulated subsample's \(z_\text{gen}\), and compute \(\widehat{\text{MMD}}^2\) and the ESS for each epoch.
4. **Select.** Choose the epoch with the smallest \(\widehat{\text{MMD}}^2\), and restore both networks to that epoch.
5. **Report.** Recompute \(\widehat{\text{MMD}}^2\) for the selected epoch on an independent subsample of the test split, using the same bandwidths.

The test-split value is the one reported. Selecting the minimum over all epochs against one validation subsample favours epochs whose weights happen to fit that particular subsample, so the minimized validation value is biased low.

The per-epoch validation MMD and ESS are stored in the run's `history.npz` and plotted in `selection.pdf` (see [Reporting & Artifacts](../user-guide/reporting.md)). When the dataset has a known truth, the same curve is also computed at particle level, comparing \(z_\text{true}\) with reweighted \(z_\text{gen}\). This curve is a diagnostic for assessing whether the detector-level criterion tracks the quantity of interest. It plays no part in selection, and it is computed outside the training program, so \(z_\text{true}\) never enters training.

---

## Computational structure

Only the simulated weights change from one epoch to the next, so every weight-independent quantity is computed once:

| Quantity | Size | Computed |
| :--- | :--- | :--- |
| \(\frac{1}{n(n-1)} \sum_{i \neq i'} k(x_i, x_{i'})\) | scalar | once |
| \(v_j = \frac{1}{n} \sum_i k(x_i, y_j)\) | vector of length \(m\) | once |
| \(k(y_j, y_{j'})\) | \(m \times m\) matrix | once |
| \(\widehat{\text{MMD}}^2\) for one epoch | scalar | per epoch, from the above |

The cross term is then \(-2 \sum_j \bar{w}_j v_j\), and the simulated term is a quadratic form in \(\bar{w}\). Scoring one epoch therefore costs one \(m \times m\) matrix-vector product, and all epochs are scored together in a single vectorized evaluation. At \(m = 16384\) the retained matrix occupies about 1 GB in single precision. The cross-kernel matrix is never stored, since only its column means are needed.

---

## Resolution

The estimator has a finite resolution. Its standard deviation under the null hypothesis (two samples from the same distribution) sets the smallest difference in \(\widehat{\text{MMD}}^2\) that can be attributed to a real difference between epochs or runs, and it scales approximately as \(1/m\). For the twelve-observable jet configuration at \(m = 16384\), it was measured as \(1.16 \times 10^{-4}\) (see `benchmarks/mmd_floor.py`). Epochs whose validation scores differ by less than this are statistically indistinguishable, and the choice among them is effectively arbitrary.

[^gretton2012]: A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf and A. Smola, "A Kernel Two-Sample Test", *Journal of Machine Learning Research* **13** (2012) 723.
