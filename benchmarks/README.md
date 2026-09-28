# Benchmarks

Four of these — `ceiling.py`, `tilt.py`, `response.py`, `averaging.py` — were
written to answer one question: **which hyperparameters matter?** The answer is
that none of them can improve the objective, which is saturated, and that this
says nothing about which of the objective's many equivalent solutions the
optimiser lands on. That distinction is the whole finding and it is worth
writing down. The investigation is summarised first; each tool is documented
below it. `hparam_collect.py` is the fifth and measures the second half;
`mmd_floor.py` is the sixth and calibrates the line between the two, since
"which solutions does the objective accept as equivalent" turned out to be the
question everything else reduces to.

---

## What the diagnostics found

_Six-variable jet configuration, 1M events, August 2026. Every number below is
reproducible from a command in this directory._

### 1. The detector-level objective is saturated

A converged, unweighted classifier separates `x_sim` from `x_data` by
**0.014786 nats** (`ceiling.py` A). After Deconvolve reweights, a _fresh_ converged
classifier at a learning rate that demonstrably works finds only **0.000087
nats** (`ceiling.py` C) — `g` removes **99.4%** of the available mismatch.
Stable across seeds (0.000087 / 0.000173 / 0.000176).

There is essentially nothing left at detector level for a better `d`, a better
`g`, or a better criterion to find.

### 2. The detector-level objective does not identify the truth

Scoring the **oracle** weight function `w*(z)` — the particle-level likelihood
ratio, fitted on truth — against Deconvolve's own weights on the criterion Deconvolve selects
with (`ceiling.py` D, held-out test split):

| weights     | detector MMD² | particle MMD² |   ESS |
| ----------- | ------------: | ------------: | ----: |
| unweighted  |      3.960e-2 |      5.899e-2 |  100% |
| oracle `w*` |     +8.020e-4 |     −1.898e-4 | 80.1% |
| Deconvolve         | **−2.322e-4** |      4.579e-3 | 73.3% |

**Deconvolve scores better than the truth on the detector-level criterion**, on events
it never saw, while scoring far worse at particle level. This is not noise and
not overfitting: `p(x | z)` is many-to-one, so the particle-level likelihood
ratio pushed through the detector response is not the detector-level likelihood
ratio. Both weight functions do exactly what they optimise.

The consequence is that **no truth-free criterion built on detector-level
agreement can rank the right answer first**, because the right answer scores
badly on it. Sharpening the estimator makes the preference more confident, not
more correct.

### 3. Capacity is irrelevant; the objective sets the performance

`tilt.py` replaces the generator with an exponential family`
`w(z; b) = exp(-b·T(z))`, fitted by moment-matching at detector level — a convex
root-find with no adversary and no training:

| method                   | parameters |   particle | detector |
| ------------------------ | ---------: | ---------: | -------: |
| tilt, degree 1           |      **6** |     +78.5% |   +92.5% |
| tilt, degree 2           |         27 |     +77.0% |   +94.8% |
| Deconvolve                      |    ~34,000 |     +78.9% |   +92.1% |
| oracle (fitted on truth) |          — | **+93.2%** |   +82.8% |

**Six parameters match thirty-four thousand.** Across three orders of magnitude
of capacity, every method that fits detector level lands at 77–79% at particle
level; the method that fits _truth_ reaches 93.2%. That ~15-point gap is the
price of not having truth, and it is not an optimisation failure.

Degree 2 is the sharpest demonstration: 21 extra parameters **improve the
fitted objective (92.5 → 94.8) and degrade the target (78.5 → 77.0)** in a
deterministic convex solve. Deconvolve's pathology reproduces with no adversary, no
stochasticity and no epoch selection, which rules out every explanation
involving the optimiser.

### 4. Jet mass is separately limited by a non-universal response

Measured two independent ways.

**Conditional mutual information** (`response.py`), with a within-generator null
closure consistent with zero (Herwig −0.000329 ± 0.000777, Pythia −0.000458 ±
0.000328):

| conditioning `Z` |    `I(S; X_m \| Z)` |  vs `Z = m` |
| ---------------- | ------------------: | ----------: |
| `m`              | 0.009006 ± 0.000461 |           — |
| `m, ang2`        | 0.004488 ± 0.000329 |  −50%, 8.0σ |
| all six          | 0.002365 ± 0.000322 | −74%, 11.8σ |
| all twelve       | 0.001623 ± 0.000380 |        −82% |

**Oracle residual** (`ceiling.py` D, per observable in isolation) — the share of
each observable's detector-level MMD² that the _true_ particle-level likelihood
ratio cannot remove:

| var                   |  residual | detector/particle info |
| --------------------- | --------: | ---------------------: |
| **m**                 | **45.3%** |               **3.73** |
| f_ch                  |     12.0% |                   0.95 |
| M                     |      5.6% |                   1.20 |
| ptd, n_ch, tau21, sdm |     ≤3.5% |                  ≤1.14 |
| w, zg, ang2, lha, q   |  **0.0%** |                  ≤0.85 |

For five observables the oracle removes the detector discrepancy _completely_ —
reweighting `z` fully explains `x`, as universal-response theory requires. Mass
leaves 45%, six times the next worst in absolute terms. The last column is the
mechanism: mass is the only observable carrying _more_ generator-discriminating
information at detector level than at particle level, and reweighting `z` can
only remove the particle-level part.

`ang2` (the angularity λ¹₂ ≈ `m²/(p_T R)²`) is the single most informative
conditioning variable, delivering 68% of what all five other OmniFold
observables achieve. This is a statement about physics — λ₂ constrains the
radiation pattern that determines how the detector smears the mass — and not
about dimensionality.

### What tuning actually found: `lr_g`

The first knob measured under the protocol above (three levels, eight
replicates, paired). **`lr_g = 3e-5` beats the old default of 1e-4 by
+1.22 ± 0.53 points on the aggregate (p = 0.022)** — combining two independent
sweeps, one replicating over `seed` (+0.45 ± 0.79) and one over `data_seed`
(+1.86 ± 0.72), which agree to 1.3 sigma. It is now the default.

Small, but real, and it is the first hyperparameter effect this project has
measured rather than asserted. Going lower does nothing: 1e-5 against 3e-5 is
+0.32 ± 0.46 (p = 0.51), so the knob is bracketed on both sides.

**The plateau below 3e-5 is cancellation, not saturation**, and that is the
interesting part. Per observable, 3e-5 → 1e-5:

| observable     |               Δ |    p |
| -------------- | --------------: | ---: |
| f_ch           | **+12.0 ± 4.9** | 0.04 |
| m              |  **−6.3 ± 1.8** | 0.01 |
| tau21          |      −1.6 ± 0.4 | 0.01 |
| the other nine |            flat |    — |

Neither p clears Bonferroni at 24 tests, so read the sizes loosely — but the
signs are the point. Lowering `lr_g` does not improve the solution uniformly;
it **relocates which observables it matches**, at essentially constant
aggregate. That is what §2 predicts a non-identifying objective must do, seen
for the first time in the tuning data rather than in the oracle diagnostics.

It is also why 1e-5 is rejected: it pays 6.3 points of jet mass — the
observable of physics interest, and already the weakest in the set — to buy
`f_ch`, which nobody is unfolding for. An automatic search maximising the
aggregate would have taken that trade, which is the argument against handing
this objective to one.

`f_ch` is the one observable where the `lr_g` effect is unambiguous:
+13.2 ± 8.4 and +15.6 ± 4.8 in the two independent sweeps, combining to
**+15.0 ± 4.2 (p = 0.0003)**, which does clear Bonferroni.

Two cautions for whoever runs the next knob:

- **Replicate over `data_seed`, not `seed`.** Data order is the larger nuisance
  (baseline SD 3.16 against 2.13) and it is the axis on which pairing works:
  it cut the standard error by 43%, against 1–12% for `seed`.
- **Check whether arms are re-runs.** The 1e-5 sweep re-ran its 3e-5 and 1e-4
  arms at configurations identical to the previous sweep's; 12 of 16 came back
  bit-identical and 4 diverged through GPU non-determinism. Those two sweeps
  are one experiment, not two, and must not be pooled.

### The dispersion penalty: the trade made explicit

`--lambda-dispersion` penalises the variance of `g`'s normalised MC weights,
which is the same axis `lr_g` acts on indirectly — ESS runs 71.1% → 74.3% →
76.7% as `lr_g` falls, so `lr_g`'s effect _was_ a dispersion effect. The penalty
pushes it directly, and turns the §2 pathology into a one-parameter family you
can walk.

The full response, 12 jet observables, n=8 replicated over `data_seed`, paired.
Admissibility is `averaging.py`'s tied-set question asked across arms: how many
resolution floors separate an arm's `mmd_test` from the best arm's. An arm the
criterion can separate is one a truth-free pipeline would refuse whatever it
scores against truth. The floor is **measured** by `mmd_floor.py`, at
1.159e-4 ± 1.0e-5, and the threshold is two floors because the floor is one
sigma of a zero-mean estimator:

|        λ |    11 obs |  jet mass | detector |   ESS | Δ MMD (floors) | in the tied set |
| -------: | --------: | --------: | -------: | ----: | -------------: | :-------------- |
|        0 |     85.65 |     32.57 |    93.13 | 74.3% |           +0.2 | yes             |
|    0.003 |     87.93 |     23.03 |    93.78 | 78.8% |           +0.0 | yes             |
| **0.01** | **91.02** | **18.54** |    92.42 | 82.8% |           +1.1 | **yes**         |
|     0.03 |     85.45 |     41.15 |    80.96 | 88.9% |           +9.2 | no              |
|      0.1 |     55.42 | **78.87** |    51.95 | 95.7% |          +67.6 | no              |
|      1.0 |     10.54 |     19.60 |     9.91 | 99.9% |         +255.6 | no              |

**The peak of the aggregate is the trough of jet mass.** λ=0.01 maximises the
eleven non-mass observables and minimises the twelfth, in the same run. Paired
against the shipped λ=0: **+5.37 ± 0.65 on the eleven (p < 0.001)** and
**−14.03 ± 3.62 on mass (p = 0.006)**. The aggregate gain is four times the
`lr_g` effect and the largest hyperparameter effect measured here; it is not
carried by one observable (tau21 +11.9, zg +8.5, sdm +3.8, M +2.4).

**Mass only improves where the criterion rejects.** Inside the tied set it
degrades monotonically (−9.55 at 0.003, −14.03 at 0.01). Outside it, mass
climbs to 41.2 at λ=0.03 and **78.9 at λ=0.1** — the top of the headroom range
below, reached by a command-line flag. λ=0.1 sits 68 floors from the best arm
and the criterion separates it decisively.

Two corrections this sweep forced, both recorded because they nearly went the
other way. The floor in use was an extrapolation and is **2.2× too loose**
(1.159e-4 measured against 2.5e-4 assumed), so the criterion is sharper than
assumed. And the first admissibility rule here compared `|mmd_test|` against
the floor as though it were a threshold; at the measured floor that rule calls
even the shipped λ=0 arm inadmissible, which is how the error surfaced. The
floor is a noise scale, the question is distance from the best arm, and the
verdicts above survive both fixes unchanged.

This is §2 in its sharpest form. Previously it was an observation about which
_epoch_ happened to be truth-best; here it is a dial, and turning it toward
truth is exactly turning it away from the objective. At λ=0.1 the generator is
barely reweighting (ESS 95.7%) and mass still reaches 78.9%, so most of the
particle-level mass discrepancy is removable by a _gentle_ reweighting and the
aggressive reweighting the detector objective demands is what destroys it —
which is what §4's non-universal mass response predicts.

**ESS is a mechanism variable, not a target.** The oracle's 80.1% falls between
λ=0.003 (78.8%) and λ=0.01 (82.8%) — i.e. at the aggregate's peak and mass's
trough. Calibrating a coefficient to the oracle's ESS puts you at the worst
setting for jet mass, because ESS is a scalar summary of a weight _function_
and reaching the oracle's spread through a detector-shaped penalty gives the
oracle's dispersion with a different function.

**No admissible λ recovers jet mass.** A sweep at 0.015 / 0.02 / 0.025 resolved
the two boundaries, and they fall in the order that closes the question: the
criterion separates an arm from the best at **λ ≈ 0.0162** (two floors), while
mass climbs back through the λ=0 baseline of 32.57 only at **λ ≈ 0.0240**. The
criterion stops accepting the arm **1.49× in λ before** mass recovers. The
headroom is separated from the admissible set by the criterion itself.

|         λ |    11 obs | jet mass |   ESS | Δ MMD (floors) | in the tied set |
| --------: | --------: | -------: | ----: | -------------: | :-------------- |
|         0 |     85.65 |    32.57 | 74.3% |           +0.2 | yes             |
|      0.01 |     91.02 |    18.54 | 82.8% |           +1.1 | yes             |
| **0.015** | **91.51** |    19.63 | 84.7% |           +1.6 | **yes**         |
|      0.02 |     90.37 |    27.00 | 86.4% |           +3.3 | no              |
|     0.025 |     88.27 |    33.89 | 87.7% |           +5.6 | no              |
|      0.03 |     85.45 |    41.15 | 88.9% |           +8.8 | no              |
|       0.1 |     55.42 |    78.87 | 95.7% |          +64.2 | no              |

Judge admissibility against the best arm **ever measured**, not the best in one
job: this sweep held no λ=0 arm, so λ=0.02 reads as tied against its own job and
separated against the global reference. `--reference-mmd` exists for that.

**The default is λ=0.015.** On the twelve-observable aggregate the paper
reports, paired against λ=0, it is **+4.30 ± 0.88 points (p = 0.0017)** — the
+5.86 across eleven observables outweighs the −12.94 on jet mass once averaged.
λ=0.01 is close behind at +3.76 ± 0.59 (p = 0.0004) and sits slightly further
inside the tied set.

This is a presentation trade, not a free win: one of the twelve panels gets
visibly worse. It is taken because jet mass is not privileged here — every
unfolding method struggles with it, for the reasons §4 measures — and eleven
observables improving by ~6 points is the better honest number to report.

### What was ruled out

Two of these are measurements and two are not, and the difference is the
per-run spread. Across six default-configuration jet runs differing only in
initialization, particle jet mass has **SD 7.16**; unpaired, resolving a
2-point difference on it needs ~201 runs per arm. Every arm below was **n=1**,
each at a _different_ seed, so the bullets marked underpowered record that
nothing was found, not that nothing is there. `hparam_collect.py` and
`scripts/submit_hparam.sh` exist to redo them with pairing and replicates.

- **`lr_d`** — _underpowered, not null._ One run per level across a 30× range,
  each at its own seed. What is solid is the reason to expect a null:
  `log2 − val_d` stays at 0.0011–0.0014 and never approaches the 0.0148 floor,
  because after reweighting that signal is no longer there to find. That is an
  argument, and the four runs do not test it.
- **Generator capacity** — solid where it rests on `tilt.py` (§3), which is a
  deterministic convex solve with no seed: 6 parameters reach +78.5% and 27
  reach +77.0%. The _Deconvolve_ number in that table (+78.9%) is a single run and
  carries a ±3 error bar, so read the tilt ladder, not the comparison to it.
- **More observables** — 6 → 7 (`+ang2`) → 12 gave particle mass 22.5% → 31.1%
  → 31.4%, which looks like the predicted pattern but is **not evidence**: nine
  six-variable runs differing only in seed span **10.0%–34.1%** (SD 9.1). Both
  new arms sit inside that range. Resolving a 7-point effect needs ~25 runs per
  arm.
- **Averaging the tied epochs** (`averaging.py`) — the detector criterion
  cannot separate ~55 of 100 epochs, so averaging their weights is the
  principled alternative to picking one. Across four runs it moved the particle
  mean by −1.7, +0.7, +0.4, +4.0 and made **jet mass worse in all four**
  (−3.4 to −6.3). Every average stayed inside the tied set on the criterion, so
  these are legitimate detector-level solutions — just not better ones.

### The headroom, and why it is unreachable

`averaging.py` reports the truth-optimal epoch alongside the shipped one. It is
**+6.8 to +10.9** points better at particle level, with jet mass at 62–80%
instead of 22–31%. The headroom is large.

Its detector MMD² is **+1.09e-3, +4.4e-4, +1.6e-4, +2.3e-4** — all positive,
while every shipped epoch and every average is negative. **The truth-optimal
epoch scores _worse_ on the criterion than the epochs around it**, reproducing
§2 at the epoch level within single runs. That is why no truth-free selection
rule recovers the headroom, and why better resolution would not help.

### Reading

Deconvolve performs at the ceiling its **objective** permits. The remaining gap is a
property of the problem — the detector-level objective is unidentifying, and
for jet mass the response is measurably non-universal between generators. The
oracle comparison quantifies the ceiling; the tilt shows it is reached by six
parameters; the response measurement gives the mass caveat a number and an
error bar.

**That is not the same as "no hyperparameter matters", and §1–2 argue against
it.** A saturated objective is one that has stopped constraining the answer:
what is left is the manifold of weight functions matching `p(x)`, and its
members span 10.0%–34.1% at particle level, with truth-optimal epochs at
62–80%. Something picks a point on that manifold and it is not the loss, which
cannot separate ~55 of 100 epochs. It is the inductive bias of the optimiser —
initialization, how far `g` travels from `w ≡ 1`, how smooth it is forced to
be, how dispersed its weights are allowed to get. The seed is the proof: a knob
with no semantics at all moves particle mass by 24 points.

So the knobs worth tuning are the ones that move _along_ the manifold, not the
ones that fit the objective harder — §3 shows fitting harder is actively
harmful (degree 1 → 2 improves detector 92.5 → 94.8 and degrades particle
78.5 → 77.0). That prediction held: `lr_g` and `--lambda-dispersion` both move
along it, and both are measured above. What did **not** hold is the hope that
the oracle's ESS gives a target to aim at — see "The dispersion penalty", where
matching it lands on the worst setting for jet mass.

None of this reopens per-measurement selection, which §2 does close. Tuning a
prior once on MC and shipping it fixed is a different operation from selecting
a checkpoint against truth on the measurement itself.

---

## `ceiling.py`

What is the most a hyperparameter sweep could buy? Run this before spending on
one.

```zsh
uv run benchmarks/ceiling.py                       # all twelve jet variables, 1M events
uv run benchmarks/ceiling.py --n-samples 200000    # a quicker read
uv run benchmarks/ceiling.py --var m --var w
uv run benchmarks/ceiling.py --run-dir runs/2026-… # adds C and D
uv run benchmarks/ceiling.py --run-dir runs/2026-… --epoch 43
```

**A, the detector-level BCE floor.** A converged, unweighted classifier
separating `x_sim` from `x_data`, in the architecture `build_discriminator`
gives Deconvolve, fitted on train and scored on val — the same two splits `val_d`
comes from. Deconvolve's `d` scoring far above this floor means `d` is the bottleneck;
scoring at it means the residual is real. Those imply opposite moves on
`lr_g`/`lr_d`/`n_disc_steps`, and `log 2 − BCE` estimates the Jensen–Shannon
divergence only when `d` is near-optimal — so without this number a `val_d` of
0.6918 is uninterpretable.

**B, whether one weight function fixes both levels.** Fit the particle-level
likelihood ratio `w*(z)` from a `z_true` vs `z_gen` classifier, then apply the
same per-event weights at _both_ levels. If both improve, a single weight
function satisfies both and a shortfall is an optimisation failure tuning can
address. If particle improves while detector degrades, `p(x | z)` differs
between the generators and no reweighting of `z_gen` can fix both.

**C, whether the residual is real or `d` gave up.** A and B bound what is
achievable; C audits what a finished run did. Freeze a saved generator,
reweight `x_sim` by its own weights, and converge a _fresh_ discriminator
against `x_data`. Scored through `deconvolve.training.engine.weighted_bce`, so the number is
`val_d` as `history.npz` defines it rather than something close to it. `--epoch`
audits any epoch, not just the selected one, which is what `params.npz` exists
for.

**D, whether the criterion prefers the oracle or the run.** A and C establish
that detector level is saturated; that leaves the question resolution cannot
answer — among weight functions that all match at detector level, does the
criterion _rank_ the truth-optimal one first? Scoring `w*` on the same weighted
MMD the training loop selects with settles it.

C and D read a run directory, and C refuses to run unless that run's
`variables`, `data_seed` and `n_samples` match the loaded sample — a generator
evaluated on permuted columns reports nonsense with a straight face.

A and B fit on train and neither scores on it. A uses train and val, the two
splits `val_d` is built from, so its floor is comparable to a number out of
`history.npz`. B needs a third: a likelihood ratio early-stopped on the events
it is then scored against reports a match it will not reproduce anywhere else,
which is the failure the diagnostic exists to rule out.

This reads `z_true` and hands it to a network, which is why it lives here and
not under `src/deconvolve/`: nothing importable as `deconvolve.*` should be able to do that by
accident. It is legitimate only because the stated goal is to tune against truth
and say so.

## `tilt.py`

What does the most regularised possible generator achieve?

```zsh
uv run benchmarks/tilt.py --config params/2d_correlated.yaml --degree 2
uv run benchmarks/tilt.py --dataset jets --degree 1 --n-samples 1000000
```

Replaces the network with an exponential family `w(z; b) = exp(-b·T(z))`.
`log P(b)` is a cumulant generating function, hence strictly convex with
`∇log P(b) = -E_w[T(z)]`, so the gradient map is injective and exactly one `b`
reproduces any achievable moment. `--degree` walks a principled capacity
ladder: degree 1 matches first moments, degree 2 adds every `z_i z_j`.

The point is a dimension count, not an appeal to smoothness. The unconstrained
problem has a manifold of weight functions matching `p(x)`; a `p`-parameter
family meets it in isolated points, turning an underdetermined problem into an
overdetermined one.

No adversary and no training: `P(b)` cancels in the normalisation, and matching
as many moments as there are parameters is a square root-find whose Jacobian is
`-Cov_w(S(x), T(z))` in closed form.

**Validated against a known answer.** Every `params/` config gives `z_gen` and
`z_true` different covariances, so the exact likelihood ratio is
`exp(quadratic)` — degree 1 provably cannot represent it and degree 2 contains
it:

| config | degree 1    | degree 2         | degree 3     |
| ------ | ----------- | ---------------- | ------------ |
| 2d     | +46.0% (2p) | **+95.3%** (5p)  | +95.0% (9p)  |
| 4d     | +32.1% (4p) | **+93.6%** (14p) | +93.6% (34p) |
| 6d     | +31.2% (6p) | **+95.5%** (27p) | +95.6% (83p) |

Degree 2 captures essentially everything and degree 3 adds nothing, which is
where theory says the ladder saturates. Residuals sit at machine precision.

The `first-moment transfer` block is what it exists to produce on jets.
Boltzmann guarantees a `b` reproducing any achievable `E_w[T(z)]`; it does not
say the `b` found at _detector_ level is that one. That holds only if the
response is mean-preserving and shared between data and MC.

## `response.py`

Do Herwig and Pythia induce the same detector response?

```zsh
uv run benchmarks/response.py --n-samples 200000 --null-repeats 5
uv run benchmarks/response.py --n-samples 200000 --x-var m           # I(S; X_m | Z_1..N)
uv run benchmarks/response.py --n-samples 200000 --x-var m --z-var m --z-var ang2
```

Fits `D_z(z)` and `D_zx(z, x)` and reports `delta = BCE(D_z) − BCE(D_zx)`, which
is `I(S; X | Z)` for calibrated, Bayes-optimal classifiers. Subtracting `I(S;Z)`
is essential: a classifier on `(x, z)` pairs succeeds even under a perfectly
universal response, because `p(z)` differs between generators by construction —
that difference is exactly what unfolding exists to correct.

`--z-var` and `--x-var` set the two column sets independently, which is what
makes the marginalisation hypothesis testable: if `K(x | z, u)` is universal and
only `r_s(x | z) = ∫ K(x|z,u) p_s(u|z) du` is not, conditioning on more of the
particle-level event should shrink the statistic. It does — see §4 above.

Run `--null-repeats` with any headline number. It splits one generator into two
pseudo-domains with **different `p(z)` but identical `p(x|z)`**, so the true
statistic is zero by construction and anything it reports is classifier bias.

`delta` is a **lower** bound: `D_zx` has the harder task, so if it sits further
from Bayes-optimal than `D_z`, the effect is understated.

## `averaging.py`

Is picking one epoch from a set the criterion calls equivalent costing anything?

```zsh
uv run benchmarks/averaging.py --run-dir runs/2026-…
uv run benchmarks/averaging.py --run-dir runs/2026-… --floor 1e-3
```

Roughly 55 of 100 epochs sit within one estimator floor of the detector-MMD
minimum, and their particle-level MMD spans nearly the whole range of the run
(Spearman +0.18 between the two criteria). Each epoch defines a reweighted
distribution `q_e`; the mixture `(1/K) Σ q_e` is a distribution too, and in
weight space it is the mean of the per-epoch normalised weights.

Weights, not parameters — two networks' parameters have no meaningful midpoint,
while the weights they induce live in the space the objective is defined on.
Each epoch is normalised to mean 1 first, or one whose softplus settled at a
larger scale would dominate a mixture it has no more claim on.

The tied set comes from **validation** detector MMD and needs no truth; scoring
is on test. `all-mean` is the control and `particle-best` the ceiling — the
latter reads `z_true` and is not a method. Every strategy reports the criterion
itself, because an average is only interesting if it is still a solution the
criterion accepts.

Requires `params.npz`, so it only reads runs trained after per-epoch parameter
saving landed. **It did not work** — see §"What was ruled out".

## `hparam_collect.py`

Did that arm actually differ from the baseline, or is this the spread?

```zsh
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_2026-08-30T…
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_… --exclude m
uv run benchmarks/hparam_collect.py --arm-dir runs/hp_… --baseline lr_g=0.0001
```

Reads the runs `scripts/submit_hparam.sh` wrote, groups them into arms, and
reports each arm's mean **with the seed spread it was measured against** as a
column rather than a footnote.

**Paired on the initialization seed.** `data_seed` is pinned across a sweep, so
`seed` is the only nuisance axis left, and the same seed in two arms starts
from the same weights. Pairing is most of the available power and it is free —
it was free for every result in "What was ruled out" too.

**Scores the mean over observables, not one of them.** From the six
default-configuration runs:

| response variable       |   SD | n/arm for 2pp, unpaired |
| ----------------------- | ---: | ----------------------: |
| particle jet mass       | 7.16 |                     201 |
| mean over all six       | 1.91 |                      14 |
| mean excluding jet mass | 1.49 |                       9 |

A factor of fourteen in compute rides on that choice. Per-observable values are
printed beside the aggregate either way, and `--exclude m` drops mass from the
score while still reporting it — §4 measures why mass is limited for reasons no
hyperparameter reaches.

**ESS is a column.** `val_ess` at the selected epoch, over the MMD subsample,
so it is directly comparable to §2's oracle at 80.1% (13124 of 16384) and Deconvolve
at 73.3%. It is the mechanism variable for `--lambda-dispersion`, which is how
that coefficient gets calibrated instead of guessed. The `lr_g` sweep shows why
it belongs here: ESS runs 11654 → 12175 → 12566 (71.1% → 74.3% → 76.7%) as
`lr_g` falls, so `lr_g`'s effect _is_ a dispersion effect and the penalty
pushes the same variable directly. Note ESS is the mechanism, not the target:
1e-5 has the highest ESS and no better aggregate.

**What defines an arm is read, not configured.** `varying_keys` diffs the saved
configs, ignoring `seed` (the pairing key) and the outcomes `config.json`
records next to the settings — `best_epoch`, `mmd_test`, the kernel bandwidths.
Without that exclusion every run is its own arm and there is nothing to pair.
So this works unchanged for the knobs that come after `lr_g`.

Read a **trend across levels**, not an argmax. At n=8 against SD 1.9 the best
of three arms is usually the luckiest.

## `mmd_floor.py`

How small an MMD² difference can the selection criterion actually resolve?

```zsh
uv run benchmarks/mmd_floor.py                              # jets, operating point
uv run benchmarks/mmd_floor.py --m 4096 --m 8192 --m 16384  # check the 1/m scaling
uv run benchmarks/mmd_floor.py --repeats 64 --n-samples 400000
```

This number decides **admissibility**, which the dispersion sweep made the
primary question: an arm within the floor of zero is one the criterion cannot
tell from a perfect match, and an arm outside it is one a truth-free pipeline
refuses however well it scores against truth. Before this it was an
extrapolation — ~5e-4 measured at m=8192, scaled by 1/m to the operating point
of `MMD_SUBSAMPLE = 16384` — and extrapolating the constant every admissibility
call rests on is not good enough.

**The null is by construction.** One population is split into two disjoint
halves, so the true MMD² is exactly zero and everything reported is the
estimator's own noise. Comparing `x_data` to `x_sim` would measure a real
discrepancy plus noise with no way to separate them.

**The floor is two-sided.** The unbiased U-statistic is not a distance and goes
negative, so −2e-4 and +2e-4 are equally indistinguishable from a match. The
number to compare against is `sd` — how far a single run wanders — not
`standard_error`, which only says how well this benchmark pinned the mean.

**Watch the `2m/N` warning.** With repeats drawn from too small a pool the
draws overlap and the spread is not the estimator's. Measured on 12-dimensional
Gaussians the null SD follows the theoretical `m^-1` while `2m/N ≤ 0.2`
(exponents −1.00 and −0.93) and flattens to `m^-0.74` at `2m/N = 0.4`. The
report states the fitted exponent so the assumption is visible rather than
trusted. This is why the benchmark draws from **all three splits**: the floor
is a property of the estimator, the distribution and `m`, not of which rows a
model was scored on, and the test split alone cannot reach `2m/N ≤ 0.2` at
m=16384 without more events than the Zenodo release holds.

**Result at the operating point: 1.159e-4 ± 1.0e-5**, against the 2.5e-4 that
had been extrapolated from m=8192 — a factor of 2.2 too loose, so the criterion
is sharper than the analysis had been crediting. Clean: `2m/N = 0.08`, no
overlap warnings, fitted exponent −0.92 against the theoretical −1.

The error bar is quoted because it is used near a boundary. `sd / sqrt(2(n-1))`
is ~9% at the default 64 repeats, and arms sit within a floor or two of the tie
threshold, where a bare number invites reading 1.1 as different from 0.9.

An earlier measurement gave 1.095e-4 at `2m/N = 0.41` on the test split alone,
before the benchmark was changed to draw from all three. That it landed within
6% of the clean value is luck rather than vindication — the overlap regime has
no guarantee attached, which is what the warning is for.

## `boundary.py`

How is the wall clock time distributed between host numpy and the training loop?

```zsh
uv run benchmarks/boundary.py
```

The primary metric is the ratio of host numpy time to training loop time. Training scales with the accelerator; host numpy and the npz cache write do not. On CPU numpy looks cheap. On an A100 the same numpy may be most of the run.

`_wd_per_dim`, `_js_per_dim` and `_triangular_per_dim` are `jnp` and run on device, and the benchmark imports those same helpers, so what is left on the host is the npz write and the `dim x n_bins` divergence reductions.

## `precision.py`

Does float32 cost unfolding accuracy, or is the difference just seed variance?

One run proves nothing: Deconvolve is an adversarial min-max game, so two runs at different seeds in the _same_ dtype already differ by ~1 percentage point per dimension. Run an ensemble in each dtype and compare the two distributions. The script runs two ensembles in each dtype, and compares them.

```zsh
for s in $(seq 0 9); do
    print "Running seed $s"
    uv run benchmarks/precision.py float64 "$s" | tee -a f64.log
    uv run benchmarks/precision.py float32 "$s" | tee -a f32.log
done
uv run benchmarks/compare_precision.py f64.log f32.log 0.5
```

Each run prints one SUMMARY line, which is what the comparison reads.

## `compare_precision.py`

Compare two precision ensembles produced by `precision.py`.

```zsh
uv run benchmarks/compare_precision.py f64.log f32.log [margin_pp]
```

The runs are **paired**: the same `--seed` initializes the same weights in both arms, on the same data. Pairing is most of the available statistical power here, because seed-to-seed variation (sd ~0.4pp) is several times the effect being looked for (~0.2pp), and it cancels in the within-pair difference.

The following are common pitfalls in significance testing of ensembles of paired data:

### 1. Comparing the separation of means to the _pooled standard deviation_

Runs should be compared be the spread of their means, not the spread of individual runs. The right scale is the standard error of the difference of means, smaller by $\sqrt{\frac{1}{n_1} + \frac{1}{n_2}}$.

### 2. Treating paired runs as independent samples

Paired runs that share a common initial condition are not independent samples. Treating them as independent samples will inflate the statistical power of the test.

### 3. Computing "$n$ needed for power $p$" from the _observed_ effect

An effect estimated near $p=0.05$ at small $n$ is inflated, so that figure is far too small. A t-test can only ever fail to find a difference, so the affirmative question -- "is the gap small enough?" -- is answered by TOST against a stated margin.

The parameter `margin_pp` is the margin of equivalence, in percentage points, that sets an upper bound on the effect size that is considered to be negligible. It is a physics judgement and has to be stated, so it is an argument, rather than an computed result.

---

## `gpu_coexistence.py`

Whether a TensorFlow subprocess can use the GPU while a JAX parent holds it.
This stands in front of any decision to bring OmniFold into this repository:
OmniFold needs TensorFlow, TensorFlow ships no wheels for this project's Python
floor, and Keras binds its backend once per interpreter. The quarantine that
answers all three is a PEP 723 script run through `uv run --no-project`, which
provisions Python 3.13 and TensorFlow in an interpreter that cannot import
`deconvolve`. What the quarantine does not settle is the GPU, and that is what this
measures.

```zsh
module load cudatoolkit/12.9                       # mandatory -- see below
uv run --no-project benchmarks/_tf_probe_worker.py # warm the env on a login node
uv run benchmarks/gpu_coexistence.py --json coexistence.json
```

### The answer is yes, in every configuration

One A100-40GB, Perlmutter, September 2026. All five arms pass:

| Arm                   | Parent                                | Free MiB before the worker | Worker |
| --------------------- | ------------------------------------- | -------------------------- | ------ |
| `control`             | never touches JAX                     | 40441 / 40960              | ok     |
| `preallocate-default` | JAX on GPU, shipped defaults          | 9675                       | ok     |
| `preallocate-false`   | `XLA_PYTHON_CLIENT_PREALLOCATE=false` | 39749                      | ok     |
| `mem-fraction-0.4`    | `XLA_PYTHON_CLIENT_MEM_FRACTION=0.4`  | 23829                      | ok     |
| `parent-on-cpu`       | `JAX_PLATFORMS=cpu`                   | 40441                      | ok     |

**No mitigation is needed.** The concern was that JAX preallocates ~75% of the
card on its first device allocation, leaving a TensorFlow worker nothing --- and
the preallocation is real and exactly as documented, 30.7GB of 40.96GB, visible
in the `preallocate-default` row. It simply does not matter here: the worker's
peak is 1.07GB against the 9.4GB that survives, so the default configuration
has nearly an order of magnitude of headroom. `XLA_PYTHON_CLIENT_PREALLOCATE`
and `JAX_PLATFORMS` are recorded above because they were the candidate fixes and
because a larger OmniFold model could still need them, not because anything is
currently broken.

### `module load cudatoolkit/12.9` is mandatory, and its absence is silent

Without it every arm returns `cpu_fallback` --- TensorFlow runs, on the CPU,
reporting nothing a caller would notice. There is no error and no exception; the
baseline would simply be tens of times slower and its numbers would look fine.
That is the failure this benchmark exists to catch, and it is why the worker
pins `tf.device("/GPU:0")` rather than letting TF place the op: soft placement
turns this into a silent pass.

The cause is a version clash, not coexistence. Perlmutter's default environment
leads `LD_LIBRARY_PATH` with four CUDA **13.2** trees, and the `tensorflow`
2.21 wheel is a CUDA **12** build. Exactly one library goes unreachable ---
`libcusolver.so.11` --- and one is enough for TF to skip registering every GPU.

### Do not read the `dlopen` table as the verdict

The worker tries each soname itself with `ctypes.CDLL`, which searches
`LD_LIBRARY_PATH` and the ldconfig cache. TensorFlow additionally reaches its
own pip wheel directories through the RPATH in its extension modules, so the
two do not measure the same thing: on the failing run the worker called **nine**
libraries unreachable while TF's log showed it had opened eight of them and
failed on one. Acting on the worker's column alone fixes the wrong thing.

`missing_libraries`, parsed from TF's own `Could not load dynamic library 'X'`
lines, is the authoritative field. The `dlopen` table is kept because it
separates _absent_ from _merely off the system path_ --- each row is
cross-referenced against the wheels on disk --- but it is labelled as
supporting evidence and nothing more. Getting TF to emit those lines needs
`TF_CPP_VMODULE=dso_loader=1`; `TF_CPP_MIN_LOG_LEVEL` does not reach them, and
the conventional `TF_CPP_MIN_LOG_LEVEL=3` suppresses the summary too, which is
how an earlier version of this benchmark reported a failure it could not
explain.
