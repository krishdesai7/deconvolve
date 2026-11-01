<!-- markdownlint-disable no-inline-html link -->
# Adversarial Reweighting

Measurements in particle physics are recorded at detector level, where finite resolution, limited acceptance and reconstruction inefficiencies distort the underlying particle-level distributions. Unfolding is the inverse problem of inferring the particle-level distribution from the observed detector-level data. Since the detector response is known only through simulation, the problem is ill-posed in general.

A common strategy is to reweight simulation rather than to invert the response directly. Each simulated event carries a particle-level configuration \(z\) and its detector-level counterpart \(x\). If weights assigned as a function of \(z\) alone make the simulated detector-level distribution agree with data, then the same weights, applied at particle level, provide an estimate of the unfolded distribution. Iterative Bayesian Unfolding[^dagostini1995] and <span style="font-variant: small-caps;">OmniFold</span>[^andreassen2020] are both instances of this strategy.

<span style="font-variant: small-caps;">Deconvolve</span> obtains the weights through an adversarial game, analogous to a generative adversarial network[^goodfellow2014]. A generator \(g\) assigns a weight to each simulated event based on its particle-level features alone. A discriminator \(d\) attempts to distinguish the reweighted simulation from data using detector-level features alone. The weights are therefore constrained at detector level, where data exist, but defined at particle level, where the unfolded result is required. At no stage does either network have access to the particle-level truth of the data.

---

## The Adversarial Formulation

Let:

- \(z \in \mathcal{Z}\) denote particle-level (nominal) kinematics.
- \(x \in \mathcal{X}\) denote detector-level (reconstructed) kinematics.
- \(p(x)\) denote the distribution of observed nature events at detector level, Data.
- \(q(z, x)\) denotes the joint distribution of MC events, with marginal Simulation distribution \(q(x) = \int q(x \mid z)\, q(z) \, \d z\).

The goal is to find a per-event weight function \(w(z)\) defined on particle-level features such that the reweighted Simulation is statistically indistinguishable from Data at detector level:

\[\widetilde{q}(x) = \int q(x \mid z) \, w(z) \, q(z) \, \d z = p(x)\]

---

## Neural Networks

### 1. Generator \(g(z; \beta)\)

The generator maps particle-level features \(z\) to positive weights:

\[g(z; \beta): \mathbb{R}^{\dim(z)} \to \mathbb{R}^+\]

Parameterized as a multi-layer perceptron with a `softplus` (\(\log(1 + e^u)\)) activation on the output layer.

To preserve the overall MC event yield, weights are normalized per batch:

\[w_i = \frac{g(z_i; \beta)}{\frac{1}{N_{\text{mc}}} \sum_{j=1}^{N_{\text{mc}}} g(z_j; \beta)}\]

### 2. Discriminator \(d(x; \phi)\)

The discriminator operates strictly at detector level:

\[d(x; \phi): \mathbb{R}^{\dim(x)} \to (0, 1)\]

Parameterized as an MLP with `sigmoid` activation, estimating the posterior probability that an event originated from Data rather than Simulation:

\[d(x; \phi) \approx \mathbb{P}(y = 1 \mid x)\]

---

## Objective & Loss Functions

The training objective is the weighted binary cross-entropy:

\[\mathcal{L}(d, g) = -\mathbb{E}_{x \sim p} \lbrack \ln d(x; \phi) \rbrack - \mathbb{E}_{(x, z) \sim q} \lbrack w(z) \ln(1 - d(x; \phi)) \rbrack\]

### The Alternating Optimization

1. **Discriminator Step**: With \(g\) fixed, minimize \(\mathcal{L}\) with respect to \(\phi\).

2. **Generator Step**: With \(d\) fixed, update \(\beta\) to maximize the discriminator's loss.

\[\min_\phi \left(- \sum_{x_i \in \text{Data}} \ln d(x_i; \phi) - \sum_{(z_j, x_j) \in \text{Sim.}} w(z_j; \beta) \ln(1 - d(x_j; \phi)) \right)\]

\[\min_\beta \sum_{(z_j, x_j) \in \text{Sim.}} w(z_j; \beta) \ln(1 - d(x_j; \phi))\]

### The Dispersion Penalty

The generator step also pays for spread in its weights. The weights are normalized to mean one over Simulation, and \(\lambda\) is set by `--lambda-dispersion`:

\[\min_\beta \sum_{(z_j, x_j) \in \text{Sim.}} w(z_j; \beta) \ln(1 - d(x_j; \phi)) \;+\; \lambda \operatorname{Var}_{\text{Sim.}}\big[w(z; \beta)\big]\]

The discriminator can only check the weights through their average at fixed detector-level features, \(\bar w(x) = \mathbb{E}[w \mid x]\), because the reweighted Simulation is \(\widetilde{q}(x) = \bar w(x)\, q(x)\). That splits the variance into the part the data constrain and the part they cannot see:

\[\operatorname{Var}(w) = \underbrace{\operatorname{Var}\big(\bar w(x)\big)}_{\text{visible}} + \underbrace{\mathbb{E}\big[\operatorname{Var}(w \mid x)\big]}_{\text{residual}}\]

The penalty is a regularizer, not part of the equilibrium. At \(\lambda > 0\) the generator trades a little detector-level agreement for less spread.

### Choosing the Penalty Strength

The unfolded weights carry over to Data only if the detector response \(r(x \mid z)\) is the same in nature and in Simulation. The value of \(\lambda\) states how far you trust that assumption.

- **The response is universal (\(\lambda = 0\)).** For example, a closure test, or a toy whose Data and Simulation are smeared by the same response by construction. Here any spread the data demand is real, and penalizing it biases the result. On the 1D Gaussian benchmark, \(\lambda = 0\) unfolds to a particle-level Wasserstein distance of 0.016. That beats <span style="font-variant: small-caps;">OmniFold</span> (0.023) and IBU (0.042). The default \(\lambda = 0.015\) gives 0.098.
- **The response may not be universal (\(\lambda \approx 0.015\), the default).** For example, real data unfolded with a simulated detector, or a Pythia/Delphes vs Herwig/Delphes study. Here matching detector level exactly can push response differences, and the generator's own estimation noise, into the weights. Shrinking them protects the particle-level result. On the jets benchmark, \(\lambda = 0.015\) improves the joint particle-level sliced-Wasserstein distance by 87.3 ± 1.0% (three seeds), against 77.3% at \(\lambda = 0\).

Larger values encode stronger doubt. Nothing at detector level can pick \(\lambda\) for you. On jets, \(\lambda = 0.015\) matches detector level slightly *worse* than \(\lambda = 0\) while unfolding better, so a detector-level criterion would choose the wrong one. The value is a modelling assumption, and like the choice of Simulation it belongs in the systematic uncertainty.

An alternative that penalized only the residual term was tested and dropped. It preserves more of the visible spread, but on jets it lost to the total-variance penalty by 3.5 points, beyond seed noise.

---

## Theoretical Equilibrium

At global equilibrium:

- The reweighted simulation perfectly reproduces observed data: \(\widetilde{q}(x) = p(x)\).
- Even a well trained discriminator is completely uninformative:
  \(d(x; \phi^*) = \frac{p(x)}{p(x) + \widetilde{q}(x)} = \frac{1}{2}\).
- The losses for both networks converge to:
  \(\mathcal{L}^* = -2\ln\left(\frac{1}{2}\right) = 2\ln(2) \approx 1.386\).

[^dagostini1995]: G. D'Agostini, "A multidimensional unfolding method based on Bayes' theorem", _Nucl. Instrum. Meth. A_ **362** (1995) 487.
[^andreassen2020]: A. Andreassen, P. T. Komiske, E. M. Metodiev, B. Nachman and J. Thaler, "OmniFold: A Method to Simultaneously Unfold All Observables", _Phys. Rev. Lett._ **124** (2020) 182001.
[^goodfellow2014]: I. Goodfellow _et al._, "Generative Adversarial Nets", _Advances in Neural Information Processing Systems_ **27** (2014).
