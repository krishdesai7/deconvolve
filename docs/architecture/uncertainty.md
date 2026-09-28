<!-- markdownlint-disable no-inline-html -->
# Uncertainty Quantification

The `deconvolve uncertainty` commands estimate how much an unfolded result varies under repetition, and attribute that variation to its sources. They produce two outputs:

- a **variance decomposition** of the unfolded mean of each observable, and
- the **bin-to-bin covariance** of each unfolded spectrum.

Both describe statistical variation of the unfolded *shape*. Systematic uncertainties, such as dependence on the generator used for the simulation (see [Saturation & Diagnostics](../theory/diagnostics.md)), and the overall normalization are outside their scope.

---

## Sources of variation

A single <span style="font-variant: small-caps;">Deconvolve</span> result depends on three random choices:

| Source | Controlled by | Status |
| :--- | :--- | :--- |
| The finite data and simulation samples | the events collected | Statistical uncertainty of the measurement |
| Network initialization | `seed` | Variance of the method |
| Train/validation/test split and batch order | `data_seed` | Variance of the method |

The first is a property of the data: it is how much the result would change if the experiment, and the simulation, were repeated with samples of the same size. The other two are properties of the algorithm, and averaging over them (ensembling) reduces them. The finite size of the evaluation sample, described below, contributes a further statistical uncertainty.

Varying `data_seed` does not estimate the statistical uncertainty. For the jet dataset it changes how a fixed set of events is split and ordered, not which events are used. The statistical uncertainty is estimated instead by the bootstrap: resampling the events with replacement.

---

## The design

The design is a grid of \(B \times S\) training runs, or *cells*: \(B\) bootstrap replicates of the dataset, each trained with the same \(S\) initialization seeds.

**Bootstrap replicates.** Each replicate draws \(n\) events with replacement from each sample of size \(n\), so every replicate has the size of the original. The simulated and data samples are resampled independently, since they are independent samples. The pairing within each is preserved: a simulated event keeps its particle- and detector-level values together. A replicate is determined by `data_seed` and its index alone, so every cell in a row of the grid trains on the same replicate.

**Splitting before resampling.** In an ordinary run, the training, validation and test sets share no events. To preserve this in a replicate, each cell assigns the *original* events to the three sets and then includes each event in its set as many times as the bootstrap drew it. All copies of an event therefore fall in the same set. (Splitting an already-resampled sample would instead put about half of each validation set into the training set as well, and the [epoch selection](../theory/mmd.md) that reads the validation set would no longer be the procedure being measured.)

**Split and order per cell.** Each cell draws its own split, batch order and model-selection subsamples, from a seed derived from `data_seed` and the cell's position in the grid. The split therefore varies independently of both the replicate and the initialization seed, and its variance is attributed to the residual term of the decomposition, together with the other variance of the method. If the split were instead fixed by the replicate, its variance would be counted in the data component, which is the one reported as the statistical uncertainty.

**Crossed seeds.** Seed \(s\) produces identical initial weights on every replicate, so a seed means the same thing in every row of the grid. This makes seed a *crossed* factor, and a seed effect common to all replicates is identifiable.

**Common evaluation set.** Before any resampling, a fixed set of \(n_\text{eval}\) simulated particle-level events (100 000 by default) is removed from the simulation and held out. Every cell's generator is evaluated on exactly these events, so every cell produces a weight vector over the same events, and the outputs of different cells can be compared event by event. The held-out events are never used in training. The unfolded result of a cell is the evaluation set reweighted by that cell's generator.

---

## Variance decomposition

Write the result of the cell with replicate \(D\) and seed \(S\) as

\[T(D, S) = \mu + a(D) + b(S) + \varepsilon(D, S),\]

with independent random effects of variances \(\sigma_a^2\) (data), \(\sigma_b^2\) (initialization) and \(\sigma_\varepsilon^2\) (residual). \(T\) is the unfolded mean of an observable, or the content of one bin of its spectrum. The residual contains the interaction between replicate and seed, the per-cell split and batch order, and any run-to-run non-determinism of the hardware. With one run per cell the design cannot separate these, and all three are variance of the method rather than of the data.

The balanced two-way crossed random-effects analysis of variance gives

\[\mathbb{E}[\text{MS}_\text{data}] = \sigma_\varepsilon^2 + S\sigma_a^2, \qquad \mathbb{E}[\text{MS}_\text{init}] = \sigma_\varepsilon^2 + B\sigma_b^2, \qquad \mathbb{E}[\text{MS}_\text{interaction}] = \sigma_\varepsilon^2,\]

from which the three components are estimated by the method of moments:

\[\hat\sigma_a^2 = \frac{\text{MS}_\text{data} - \text{MS}_\text{interaction}}{S}, \qquad \hat\sigma_b^2 = \frac{\text{MS}_\text{init} - \text{MS}_\text{interaction}}{B}, \qquad \hat\sigma_\varepsilon^2 = \text{MS}_\text{interaction}.\]

These estimators are unbiased but can be negative when a component is small compared with its sampling error. They are reported as computed: a negative value means the component is not resolved by the grid, not that it is zero. The estimates are themselves uncertain, with \(B - 1\), \(S - 1\) and \((B - 1)(S - 1)\) degrees of freedom respectively, so the grid shape determines how precisely each is known.

### Why a grid rather than two sweeps

Two one-dimensional sweeps (varying the seed at one fixed dataset, and the dataset at one fixed seed) estimate \(\sigma_b^2 + \sigma_\varepsilon^2\) and \(\sigma_a^2 + \sigma_\varepsilon^2\). Adding them gives \(\sigma_a^2 + \sigma_b^2 + 2\sigma_\varepsilon^2\), which exceeds the total variance by \(\sigma_\varepsilon^2\). In the adversarial training used here the residual is not small, so this overestimate is significant. The output reports the sum the two sweeps would have given alongside the correct total.

### Uncertainty of the evaluation sample

The unfolded result is the evaluation set reweighted by the generator, and a different draw of those events would change it even with the generator held fixed. Because every cell uses the same evaluation events, this fluctuation is common to all cells and cancels from every component above. It is computed separately, from each cell's weights, by the delta method. For the unfolded mean \(m = \sum_i w_i z_i / \sum_i w_i\),

\[\sigma_\text{eval}^2 = \frac{\sum_i w_i^2 (z_i - m)^2}{\left(\sum_i w_i\right)^2},\]

and analogously for the bin fractions of the spectrum. The result is averaged over cells. It grows as the weights become more unequal, that is, as the effective sample size of the reweighted evaluation set falls.

### What to quote

The variance appropriate to a result depends on how that result is produced:

- **A single run:** \(\sigma_a^2 + \sigma_b^2 + \sigma_\varepsilon^2 + \sigma_\text{eval}^2\).
- **The average of an ensemble of \(S'\) runs on the same data, each with its own seed and split:** \(\sigma_a^2 + (\sigma_b^2 + \sigma_\varepsilon^2)/S' + \sigma_\text{eval}^2\). Ensembling reduces the method terms in proportion to the ensemble size; it does not remove them.

The statistical uncertainty of the measurement is \(\sigma_a^2 + \sigma_\text{eval}^2\).

The decomposition is computed for each observable's unfolded **mean**, calculated exactly from the weighted events. The mean involves no binning choice. Estimating it from bin centres instead would add a spurious contribution from the bin widths, which is large for observables with wide outer bins.

---

## Bin-to-bin covariance

Each cell's unfolded spectrum is histogrammed in \(K\) bins (20 by default) and normalized to unit sum. The bin edges are quantiles of the evaluation set, so that each bin holds a similar number of events. Duplicate edges, which occur for discrete observables, are merged, so an observable may have fewer bins.

The three components are then estimated as \(K \times K\) covariance matrices, generalizing the decomposition above, together with the covariance of the evaluation sample. The covariance of the dataset-averaged spectra estimates \(\Sigma_a + \Sigma_\varepsilon / S\), not \(\Sigma_a\): at finite \(S\), averaging over seeds leaves part of the interaction in each replicate's mean. The reported data covariance is corrected for this, \(\hat\Sigma_a = \widehat{\operatorname{Cov}}(\text{dataset means}) - \hat\Sigma_\varepsilon / S\), and the initialization covariance analogously.

Two properties of these matrices must be taken into account when reading them.

- **Normalization.** Every spectrum sums to one, so each row of each covariance matrix sums to zero, and the matrices have rank at most \(K - 1\). For bins of equal variance, this constraint alone makes the *average* correlation of a bin with the others equal to \(-1/(K - 1)\), the value for a multinomial distribution with equal bin probabilities. The constraint fixes only this average; it does not fix individual entries. Structure in the correlation matrix, such as strong positive correlation between neighbouring bins, is information that the normalization does not imply. The output records \(-1/(K - 1)\) for each observable as a reference.
- **Rank.** A covariance estimated from \(B\) replicates has rank at most \(B - 1\). For \(B \le K\) it is singular and the correlations are constrained to \(\pm 1\) regardless of the physics. `collect` warns in this case. Reliable off-diagonal elements require \(B\) well above \(K\), for example \(B \approx 100\) for \(K = 20\).

---

## Running a design

```shell
deconvolve uncertainty freeze runs/unc_x -B 100 -S 2   # fix the settings, once
deconvolve uncertainty run 0 runs/unc_x                # one cell; repeat for 0 … B·S−1
deconvolve uncertainty collect runs/unc_x              # decompose and write the outputs
```

`freeze` records the settings of the design in `design.json`, and every cell reads them from there, so all cells are trained identically. `--resample` chooses which samples the bootstrap resamples: `both` (the default) for the combined statistical uncertainty, or `data` or `mc` alone. Physics analyses usually quote the statistical uncertainties of the data and of the simulation separately; they are obtained from two designs, one resampling each. Cells are independent and can run in parallel; `scripts/submit_uncertainty.zsh` runs a design as a SLURM job. Cells are numbered so that the first \(k \times S\) complete \(k\) replicates, and `collect` requires the full grid.

A square grid such as \(8 \times 8\) determines the variance components well. The covariance needs many replicates rather than many seeds, so a design such as \(100 \times 2\) suits it better. With \(S = 2\), the correction to the data covariance removes half the interaction covariance, so it depends on the interaction being well estimated.

`collect` writes three files to the design directory:

| File | Contents |
| :--- | :--- |
| `variance.json` | Per observable: the variance components of the unfolded mean and the evaluation-sample variance, as variances and standard deviations; the run-to-run total (\(\sigma_a^2 + \sigma_b^2 + \sigma_\varepsilon^2\), excluding the evaluation sample); the naive sum of two sweeps; and the \(-1/(K - 1)\) reference. |
| `variance.npz` | Per observable: bin edges, mean spectrum, the three component covariance matrices, the evaluation-sample covariance, and the correlation matrix of the data component. |
| `correlation.pdf` | The data-component correlation matrix of each observable. |

---

## Limitations

- **Fewer simulated events.** The replicates train on slightly less simulation than a full run, since the evaluation set is withheld from the simulated sample before resampling. The estimated variance is therefore that of a sample smaller by \(n_\text{eval}\) simulated events.
- **Separating the residual.** The interaction, the split and the hardware non-determinism are reported as one residual. All are variance of the method, so this does not affect the statistical uncertainty; separating them would require additional runs per cell.
- **Scope.** The outputs describe statistical variation of the unfolded shape. Systematic uncertainties and the overall normalization require separate treatment.
