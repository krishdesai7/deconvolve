# Uncertainty

The variance budget for a RAN measurement, and the bin-to-bin covariance the
field has been assuming away.

## Three sources, not two

A published number needs to say how much it would move if the experiment were
repeated. For an unfolding trained by an adversarial game there are three
things that could be repeated differently, and they do not have the same
status:

| Source                | Varied by   | What it is                                             | What to do with it                               |
| --------------------- | ----------- | ------------------------------------------------------ | ------------------------------------------------ |
| Finite sample         | bootstrap   | the 1M events are one draw from the population         | **report it**: it is the statistical uncertainty |
| Split and batch order | `data_seed` | which events land in train/val/test, and in what order | remove it by ensembling                          |
| Initialization        | `seed`      | where the two networks started                         | remove it by ensembling                          |

Only the first is an uncertainty on the measurement. The other two are
_method_ variance --- artifacts of the algorithm being order- and
init-dependent, which a competitor could eliminate by averaging --- and
folding them into a quoted band inflates the error bar with something that is
not a property of the data.

Varying `data_seed` does **not** estimate the first: every run still sees the
same events, reshuffled. The nonparametric bootstrap does, so the dataset axis
is a bootstrap replicate.

Two details keep that axis purely statistical:

- **The split varies per cell, independently of both axes** (`split_seed`,
  which also sets the batch order and the MMD selection subsamples). A split
  fixed by the replicate would have its variance charged to the dataset
  component, the one reported as the statistical uncertainty. Drawn per cell,
  it lands in the residual with the rest of the method variance.
- **Events are split before they are resampled.** A replicate is a
  multiplicity per original event (`bootstrap_multiplicities`), drawn once per
  dataset index so every cell in a row trains on the same replicate; each
  cell splits the *original* events and repeats each within its split
  (`replicate_splits`). Splitting an already-resampled sample puts copies of
  one event into different splits --- about half of every validation set
  would also be training data --- and the epoch selection that reads the
  validation set would no longer be the procedure being measured.

`--resample data` or `--resample mc` resamples one side only, so two designs
give the data and simulation statistics separately, as analyses usually quote
them.

## Why a grid and not two sweeps

Write a run as `T(D, S)`, and decompose it as `mu + a(D) + b(S) + eps(D, S)`.
The obvious pair of measurements --- vary the seed at one fixed dataset, vary
the dataset at one fixed seed --- give `sigma_b^2 + sigma_eps^2` and
`sigma_a^2 + sigma_eps^2`. Summing them in quadrature counts the interaction
twice:

\[\text{naive} = \sigma_a^2 + \sigma_b^2 + 2 \sigma_eps^2\]
\[\text{truth} = \sigma_a^2 + \sigma_b^2 + \sigma_eps^2\]

`sigma_eps^2` is the part of a run that depends on the _combination_ and is
attributable to neither axis --- here the interaction together with the
per-cell split and order and any hardware non-determinism, which one run per
cell cannot separate. All of it is method variance. In a min-max game it is not small: the effect of
an init seed already fails to transfer across `lr_g` arms (measured
`r = +0.04`), which is the same phenomenon in a different coordinate. So the
naive sum is not a safe over-estimate to quote --- it is a wrong number in a
known direction, and the only way to know by how much is to run the grid.

`decompose` reads a `B x S` grid and returns all three components from the
balanced two-way crossed random-effects ANOVA. The seed is a _crossed_ factor
rather than a nested one because a seed means the same thing in every cell:
`keras.utils.set_random_seed(s)` puts identical initial weights on the network
whichever dataset it is about to see, so a seed main effect is a real thing
the design can see.

Components are moment estimators, so an unlucky grid can return a small
negative value where the truth is near zero. They are reported raw. A
`max(0, .)` would turn "we cannot resolve this" into "this is exactly zero",
and a variance budget that never admits the first is not one to trust.

## The common evaluation set

Bootstrap replicates hold different --- and duplicated --- events, so their
per-event weight vectors are not commensurable and cannot be stacked into the
matrix a covariance is computed from. `reserve_evaluation_set` therefore holds
out a fixed block of gen-level MC events _before_ any resampling, and every
cell is read on exactly those. Two consequences:

- No replicate can have trained on an evaluation event.
- The finite size of the evaluation set shifts every cell together and cancels
  out of the across-cell contrast entirely, so it contributes nothing to any
  component. That makes it a missing term, not a negligible one: the unfolded
  result *is* those weighted events. `evaluation_variance` and
  `evaluation_covariance` supply it by the delta method,
  `sum(w^2 (z - m)^2) / sum(w)^2` for the mean, averaged over cells, and
  `collect` reports it as its own line to be added to the total.

Only the MC side is reserved: `g` is evaluated on `z_gen` and never on a
nature event, so there is nothing to hold out on that side.

The set is regenerated by `collect` from two recorded seeds rather than stored
in each cell --- one copy per cell would be hundreds of megabytes of the same
array, and having exactly one construction makes it impossible for the cells
to disagree about it.

## The covariance, and the argument it is for

Unbinned unfolding weights get propagated as though their bin-to-bin
correlations were zero. They are not. The reason the assumption survives is
plausibly not that anyone believes it: measuring the covariance takes ~100
retrainings, which at OmniFold's cost is not an analysis anyone runs, and at
RAN's is a node-hour. That reframes the speed result --- "1000x faster at
comparable accuracy" invites _so what, we already have the answer_; "fast
enough to bootstrap the full unfolding a hundred times, which is how you find
out the covariance you assumed diagonal is not" is a capability claim.

Two things keep the measurement honest:

**The correction.** The dataset-averaged spectrum still carries `mean_S eps`,
so the raw between-dataset covariance estimates `Cov_a + Cov_eps / S` and has
to be corrected before it means what its name says. Skipping the step inflates
the off-diagonals in the flattering direction.

**The closure reference.** RAN's weights preserve the total count, so a
spectrum's bins sum to a fixed number, its covariance is singular with rank
`K - 1`, and every row sums to zero. For bins of equal variance that fixes the
_average_ correlation of a bin with the others at `-1 / (K - 1)`, the
multinomial value for equal-occupancy bins, which `multinomial_off_diagonal`
writes into the output next to the measurement. The constraint fixes nothing
else: individual entries are free and can lie on either side of it, so it is
a reference, not a floor. Structure around it --- neighbouring bins
correlating more than distant ones --- is the part normalization cannot
explain, and the part the argument rests on.

Bins are equal-occupancy (`quantile_edges`) because a `K x K` covariance from
`B` replicates needs every bin to carry enough events to be a measurement
rather than a coin flip, and linear edges over a jet observable put most of
them in a tail.

**`B` has to exceed `K`.** A covariance estimated from `B` replicates has rank
at most `B - 1`, so at `B <= K` it is singular and every correlation saturates
at `+-1` --- a heatmap of solid red and blue that reads as a very strong
result and is entirely an artifact of the sample size. `collect` warns when
the grid cannot support the binning it was asked for; 20 bins wants `B` around 100.

## Running it

One cell per invocation, so a cluster puts every cell on its own GPU and the
whole design costs one training run of wall clock:

```bash
deconvolve uncertainty freeze runs/unc_x -B 8 -S 8
deconvolve uncertainty run 0 runs/unc_x
deconvolve uncertainty collect runs/unc_x
bash scripts/submit_uncertainty.zsh          # the packed 8x8 on SLURM
B=50 S=2 bash scripts/submit_uncertainty.zsh # replicates on the bootstrap axis
```

Cells are numbered seed-major, so a design cut short after `k * S` cells is a
complete grid over the datasets that finished rather than a ragged one. A
partial grid is refused by name: the mean squares assume one run per cell, and
decomposing whatever landed would charge the imbalance to the dataset axis.

`collect` writes three things into the design directory:

- `variance.json` --- the per-observable decomposition of the **unfolded
  mean**, which is the summary because it depends on no binning choice, plus
  what the naive quadrature sum would have claimed, and the evaluation-sample
  variance (`var_evaluation`, outside `var_total`).
- `variance.npz` --- per observable, the bin edges, the mean spectrum, all
  three component covariances, the evaluation-sample covariance and the
  bootstrap correlation matrix.
- `correlation.pdf` --- one heatmap per observable, each captioned with its
  closure reference.

## What the design measured

> **Measured under the previous design.** Everything below predates three
> corrections made on 2026-09-27: replicates were split *after* resampling
> (about half of each validation set was also training data), the split was
> fixed by the replicate (so its variance was counted in the data
> component), and the evaluation sample's own variance was not reported. The
> data component and every figure derived from it --- in particular the
> 0.63-0.80x ratio below --- are therefore not the statistical uncertainty
> they are presented as, and must be re-measured. The structural findings
> (no seed main effect; a large residual; strongly correlated neighbouring
> bins) are the most likely to survive, but that too is to be confirmed.

> **Superseded configuration.** Everything below was measured at
> `-n1000000`. The shipped run is now `-n1600000` (`scripts/submit.sh`), and a
> variance budget at one sample size does not describe a measurement at
> another: the finite-sample component is the one being reported, and it is the
> one that moves with N. Both grids are being rerun at 1.6M — the decomposition
> _and_ the covariance, since mixing sizes across the two would describe no
> single model. Until those land, the numbers here are the best available and
> are quantitatively wrong for the current run. The structural findings
> (initialization has no main effect; the interaction dominates) are what is
> expected to survive, because neither is a statement about sample size.

Three grids on the twelve-observable jet run at the then-shipped configuration
(`-n1000000 -l3 -u128`, `lr_g` 3e-5, `lambda_dispersion` 0.015), 2026-09-02 to
2026-09-03: an 8x8 for the decomposition, and 50x2 then 100x2 for the
covariance, the second superseding the first once it confirmed nothing moved.
All 364 cells across the three trained, no non-finite scores, ESS/m 0.851-0.852
throughout. The numbers below are final: 8x8 for the decomposition, 100x2 for
the covariance.

**Initialization has no main effect.** Across all twelve observables the seed
component of the unfolded mean runs from -3.7% to +11.2% of the total
variance, most of it negative --- that is, unresolved at this grid size and
consistent with zero. The same is true of the selection criterion itself:
decomposing `mmd_test` gives `data 89.5% / init -1.1% / interaction 11.6%` on
the 8x8, `97.9% / -0.0% / 2.1%` on the 50x2, and `96.1% / -0.0% / 3.9%` on the
100x2 --- stable across three independent grids of increasing size.

This is not the same as saying initialization does not matter. It matters
almost entirely _in combination with the dataset_: the interaction carries
38-59% of the variance of the unfolded mean. A seed has no dataset-independent
effect, which is why the effect of a seed failed to transfer across `lr_g`
arms (`r = +0.04`) --- there was never a transferable thing to find. What the
literature calls "ensemble spread" is, here, an interaction term.

**Consequences, in the order they matter:**

- Quadrature-summing two one-dimensional sweeps overstates the total SD by
  **18-26%** (mean 20.8%), which is the interaction being counted twice.
- Ensembling over seeds at fixed data reduces the initialization and
  residual terms in proportion to the ensemble size, converging on the
  bootstrap component. Its SD is **0.63-0.80x** a single run's spread (mean
  0.73x); that factor is the most ensembling can buy.
- Which bootstrap replicate was drawn determines fit quality more than
  which seed did: at the 100x2 grid, 10 of the 19 datasets with at least one
  cell reading `mmd_test > 5e-4` (~4 floors) have it in _both_ seeds, against
  2.1 expected if the two seeds failed independently. A bad fit is a property
  of the dataset, not of the initialization drawn for it.

**The off-diagonals are large, and coherent.** Against a closure reference of
`-1/19 = -0.053`, the measured adjacent-bin correlation of the unfolded
spectrum is **0.90-0.94**, falling to 0.42-0.68 at lag 3 and to
-0.09-0.27 at lag 5, then turning negative across the spectrum. The
covariance is dominated by two or three modes: effective rank 2.3-3.6 out of
20 bins (16 for `n_ch`), with the top two eigenvalues carrying 73-98% of the
trace. Propagating a diagonal covariance understates the error on a linear
functional of the spectrum by a median factor of **1.60** (range 1.22-2.62 for
the mean; 1.49-1.97 for a top-quartile fraction).

So the assumption is not merely wrong in magnitude, it is wrong in shape: the
true covariance is nearly rank-2, and no rescaling of per-bin error bars can
represent it. This held to within 0.01 on every lag correlation and 0.2 on
every effective rank between the 50x2 and 100x2 grids; only individual
off-diagonal entries moved (by up to 0.39 at B=50). This motivated the larger
grid; entry-by-entry stability was not checked beyond B=100.

**Two caveats, because they bound what the numbers support.**

- `S = 2` makes the `Cov_eps / S` correction subtract half the interaction ---
  the largest correction any design shape asks for. The 8x8 measures the
  interaction to 49 degrees of freedom, so the correction is well determined;
  it is still the term to check first if a number looks wrong.
- **Quote the exact weighted mean, not the binned proxy.** Scoring the mean
  off bin centres inflates its SD by a median 24%, and by 2.5x on jet mass,
  whose outer quantile bin is wide enough to give a fluctuation there a large
  lever arm. `weighted_means` is the exact quantity, and the summary table
  reports it; `variance.npz`'s covariances are for binned functionals.
