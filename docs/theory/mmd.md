<!-- markdownlint-disable no-inline-html -->
# MMD Model Selection

In adversarial games, generator and discriminator losses oscillate around equilibrium and cannot be used for early stopping or checkpoint selection. <span style="font-variant: small-caps;">Deconvolve</span> employs **Maximum Mean Discrepancy (MMD)** evaluated on a validation split as a principled, non-parametric model selection criterion.

---

## Maximum Mean Discrepancy Formulation

The Maximum Mean Discrepancy between two distributions \(P\) and \(Q\) mapped into a Reproducing Kernel Hilbert Space (RKHS) \(\mathcal{H}\) with kernel \(k(\cdot, \cdot)\) is:

\[\text{MMD}^2(P, Q) = \mathbb{E}_{x, x' \sim P}\lbrack k(x, x')\rbrack - 2 \mathbb{E}_{x \sim P, y \sim Q}\lbrack k(x, y)\rbrack + \mathbb{E}_{y, y' \sim Q}\lbrack k(y, y')\rbrack\]

For discrete samples \(\{x_i\}_{i=1}^n\) (Data) and \(\{y_j\}_{j=1}^m\) with weights \(w_j\) (Simulation):

\[\widehat{\text{MMD}}^2 = \frac{1}{n(n-1)} \sum_{i \neq i'} k(x_i, x_{i'}) - \frac{2}{n} \sum_{i, j} w_j k(x_i, y_j) + \sum_{j \neq j'} w_j w_{j'} k(y_j, y_{j'})\]

---

## Multi-Scale Gaussian Mixture Kernel

To capture discrepancy across multiple spatial resolutions, <span style="font-variant: small-caps;">Deconvolve</span> uses a mixture of \(K\) Radial Basis Function (RBF) kernels:

\[k(x, x') = \frac{1}{K} \sum_{k=1}^K \exp\left( -\frac{\|x - x'\|^2}{2 \sigma_k^2} \right)\]

### Bandwidth Heuristic

Bandwidths are determined using the median heuristic over pairwise Euclidean distances on the validation set:

1. Compute the median pairwise distance \(\sigma_0\).
2. Scale geometrically across \(K = 5\) bandwidths:

\[\sigma_k = \sigma_0 \times 2^{k - \lfloor K/2 \rfloor}, \quad k \in \{0, 1, 2, 3, 4\}\]

---

## The MMDCache & Subsampling

Evaluating MMD naively has \(\mathcal{O}(N^2)\) complexity. <span style="font-variant: small-caps;">Deconvolve</span> optimizes this with an exact caching strategy:

1. **Subsampling**: A fixed subsample (e.g. \(N    = 4096\)) of validation events is drawn before training starts.
2. **Fixed Kernel Cache**: The data-data kernel matrix \(K_{\text{data}, \text{data}}\) is computed **once** and cached on device.
3. **Epoch Evaluation**: At the end of each epoch, only the cross-terms \(K_{xy}\) and simulation-simulation terms \(K_{yy}\) are evaluated against the current generator weights \(w_j = g(z_j; \beta)\).

The epoch that achieves the minimum validation \(\text{MMD}^2\) is automatically selected and saved as the final model checkpoint.
