# RANv4 Overview

---

## 1. The Problem: Unfolding

Collider detectors smear particle-level ("truth") distributions. We observe reco-level data `x_data`, but we want to know the underlying particle-level distribution `z_true`. This is the **unfolding** (deconvolution) problem.

The difficulty: we never measure `z_true` for real data events. We only have MC simulation where both levels are known — but the MC may not perfectly describe nature (mismodeling of the generator distribution `z_gen`).

---

## 2. Core Algorithm — RAN

**Key insight:** instead of trying to directly infer `z_true`, reweight the MC generator distribution so the resulting reco-level simulation matches the data.

### Variables

| Symbol   | Meaning                                   | Who sees it         |
| -------- | ----------------------------------------- | ------------------- |
| `z_gen`  | MC particle-level (nominal truth)         | Generator input     |
| `x_sim`  | MC reco-level                             | Discriminator input |
| `z_true` | Data particle-level (unknown in practice) | **Nobody**          |
| `x_data` | Data reco-level (observed)                | Discriminator input |

### The two networks

**Generator `g(z)`** — particle-level network

- Input: `z_gen` (MC nominal-level events)
- Output: per-event scalar weights `w = g(z_gen)` via **softplus** (ensures w > 0)
- Weights normalized so `Σ w_MC = N_MC` (preserves total count per class)
- Data events: weight fixed to 1

**Discriminator `d(x)`** — reco-level network

- Input: `x` (reco-level events, both data and reweighted sim)
- Output: probability in \[0,1\] via sigmoid
- Trained to distinguish data (y=1) from reweighted sim (y=0)

### Loss function

Weighted binary cross-entropy:

```text
L = -mean[ w_i * y_i * log(d(x_i)) + w_i * (1-y_i) * log(1 - d(x_i)) ]
```

**Min-max game:**

- `d` minimizes L (correctly distinguish data vs. reweighted sim)
- `g` maximizes L (generate weights that fool `d` — make sim look like data)

### Optimal equilibrium

At Nash equilibrium, the discriminator can no longer distinguish the two distributions, converging to `d(x) = 0.5` everywhere and loss → `log(2) ≈ 0.693`. This is the horizontal reference line shown on the loss curve plots. Convergence to `log(2)` is the diagnostic that training has succeeded.

### Why this works

If `g` successfully fools `d`, then the reweighted reco-level sim distribution matches the data reco distribution. Because the weights depend only on `z_gen` (not `x_sim`), the same weights applied at particle-level give the unfolded truth estimate. This is the key physical insight: a weight function on generator phase space induces a consistent reweighting at both levels.

---

## 3. Critical Constraint: No z_true Leakage

**`g` must never see `z_true`.** In a real experiment, we do not have particle-level data events — only reco-level. The network `g(z)` takes only MC generator-level events as input.

This is enforced structurally: the dataset stores `z` for both classes (where `z_true` goes in for data events, `z_gen` for MC), but the training loop only feeds the z-features of MC events (y=0) to `g`. Data event weights are hard-coded to 1.

### Data Poisoning Verification (`ran leakage-check`, in `src/ran/leakage.py`)

To prove empirically that `g` never sees `z_true`, the check runs two full training trials on the same 1D Gaussian setup and compares results:

**CLEAN run:** `z_true ~ N(0, 1)` — normal data particle-level values.

**POISONED run:** `z_true[:] = -999.0` — every data particle-level value overwritten with a nonsense sentinel. If `g` had any access to `z_true`, the poisoned run would train differently (it can't learn from corrupted input).

The reco-level `x_data` is generated _before_ poisoning, so the data that the discriminator actually sees is unchanged. The poison only corrupts the `z` slot for data events in the dataset.

**Expected result:** both runs should produce statistically identical Wasserstein and triangular discriminator improvements. The generator only computes weights from `z_gen` (MC events, y=0), so the -999 values for data events are never passed through it. If the poisoned run degrades significantly, it indicates a leakage path.

**What it checks:**

- Discriminator cannot use `z_true` (it only gets `x`)
- Generator only consumes `z_gen` (enforced by the `y==0` mask in the training loop)
- The `z` field for data events is structurally present in the dataset but architecturally dead

**To run:**

```bash
ran leakage-check --clean    # clean baseline
ran leakage-check --poison   # poisoned — should match
```

Both arms must share `--seed`, or initialization variance swamps the effect and
they differ even with no leakage. With it fixed, detector-level results are
bit-identical between the two arms.

---

## 4. Network Architecture (`src/ran/models.py`)

Both networks share the same MLP structure:

```text
Input (dim,) → [Dense(64, relu)] × n_layers → Dense(1, activation)
```

- Generator output: **softplus** → w > 0 always
- Discriminator output: **sigmoid** → probability in (0, 1)
- Default: 64 hidden units, 2 layers, float64 throughout (physics requires precision)

Hyperparameters exposed at CLI: `--hidden-units`, `--n-layers`.

---

## 5. Training Loop (`src/ran/train.py`)

### Weight normalization (`_compute_weights`)

```python
w_MC = g(z_gen) * N_MC / sum(g(z_gen))   (MC events)
w_data = 1                                (data events)
```

Keeping `g` out of the discriminator update is important: the discriminator should not backpropagate through the generator when updating its own parameters. Under JAX this needs no `stop_gradient` — `disc_step` computes the weights _before_ calling the differentiated function, so they enter it as plain constants, and `jax.value_and_grad` differentiates only with respect to the discriminator's trainable variables.

### Alternating update schedule

- Every batch: one discriminator step
- Every `n_disc_steps` batches (default 5): one generator step

This is standard GAN practice — the discriminator needs to be slightly "ahead" of the generator to provide a useful gradient signal.

### Early stopping

- Monitor **validation discriminator loss** — higher is better (approaching `log(2)`)
- Patience: 5 epochs with improvement < 1e-4
- Best weights are checkpointed and restored on stop

---

## 6. Data Pipeline (`src/ran/data/`)

### Gaussian toy datasets (`datasets.py`, `config.py`)

Controlled test where the true answer is known analytically. Each YAML in `params/` defines:

- `mu_gen`, `mu_true`: means of the MC and true distributions
- `sigma_gen`, `sigma_true`: spread (scalar → σ²I, vector → diagonal, matrix → full covariance)
- `sigma_detector`: detector smearing (same promotion logic)

Data generation:

```python
z_true ~ N(mu_true, cov_true)       # data particle-level
x_data = z_true + ε, ε ~ N(0, cov_detector)   # data reco

z_gen ~ N(mu_gen, cov_gen)          # MC particle-level
x_sim = z_gen + ε', ε' ~ N(0, cov_detector)   # MC reco
```

Covariance validation: every sigma input is Cholesky-decomposed to verify positive-definiteness before use.

**Caching:** datasets are hashed by their full parameter set + n*samples + seed and stored in `.cache/gaussian*<hash16>.npz`. Reruns with identical config are instant.

### Jet substructure dataset (`jets.py`)

Real W-boson jet data from Zenodo (6 variables). Variables:

| Key     | Observable                | Physical meaning                |
| ------- | ------------------------- | ------------------------------- |
| `m`     | Jet mass (GeV)            | Leading mass scale              |
| `M`     | Constituent multiplicity  | Track count                     |
| `w`     | Jet width                 | Radial spread of energy         |
| `tau21` | N-subjettiness ratio τ₂₁  | 2-prong vs 1-prong substructure |
| `zg`    | Groomed momentum fraction | Soft-drop leading splitting     |
| `sdm`   | Soft-drop jet mass (ln ρ) | Groomed mass in log scale       |

**Standardization:** z-score using MC generator-level (`z_gen`) statistics only. The same (μ, σ) is applied to all four arrays (`z_true`, `x_data`, `z_gen`, `x_sim`) to avoid leakage and preserve correlations. The parameters are saved per-run for inverse-transforming plots.

### Dataset splits

- 70% train / 10% val / 20% test (configurable)
- Train: reshuffled each epoch
- Val/test: fixed, not reshuffled

---

## 7. Config Formats (`params/`)

**1D uncorrelated:**

```yaml
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9 # scalar → 0.81 * I₁
sigma_true: 1.0
sigma_detector: 0.5
```

**2D correlated:**

```yaml
mu_gen: [0.0, 1.0]
mu_true: [0.2, 0.8]
sigma_gen: # full 2×2 matrix
  - [1.0, -0.9]
  - [-0.9, 2.25]
sigma_detector: [0.5, 0.8] # vector → diagonal
```

The sigma promotion (`sigma_to_covariance`) unifies all three input forms into a proper covariance matrix, validated via Cholesky.

---

## 8. Evaluation Metrics (`src/ran/evaluate.py`)

Three distributional distance metrics, computed **before and after reweighting**, at both detector and particle levels:

| Metric                           | Formula                                       | Sensitivity              |
| -------------------------------- | --------------------------------------------- | ------------------------ |
| **Wasserstein-1**                | Earth-mover distance (sorted CDF, O(n log n)) | Magnitude of shift       |
| **Jensen-Shannon divergence**    | (JS distance)² via histograms                 | Shape differences        |
| **Triangular discriminator** (Δ) | Σ (p-q)²/(p+q) × 10³                          | Bin-by-bin shape, scaled |

All computed per-dimension (1D marginals). Improvement reported as percent reduction from baseline.

---

## 9. Baselines

### IBU (`src/ran/baselines/ibu.py`)

Classic frequentist unfolding. 1D per-variable:

1. Build response matrix R\[t,r\] = P(reco=r | truth=t) from MC
2. Apply Bayes iteratively: P(t|r) = R\[t,r\]·P(t) / Σ R\[t',r\]·P(t')
3. Convert unfolded histogram → per-event weights for comparison

**Purity-based binning:** bins are grown greedily from the left until P(same bin at gen and reco) > √0.5 ≈ 0.707. This ensures the response matrix is well-conditioned.

**Limitation:** strictly 1D — cannot capture correlations between variables. RAN handles the joint distribution natively.

---

## 10. Output Plots (`src/ran/plotting.py`)

Each run produces three PDFs:

**`detector_level.pdf`** — x_data vs. x_sim (reweighted and raw), per variable. Shows that the reweighted simulation matches data at reco level.

**`particle_level.pdf`** — z_true vs. z_gen (reweighted and raw), per variable. The key physics result: unfolded MC vs. truth. Note: for real data, z_true is not available — this plot uses MC where both levels are known, acting as a closure test.

**`losses.pdf`** — train/val discriminator and generator loss curves with `log(2)` reference line. Convergence to `log(2)` confirms the adversarial equilibrium was reached.

All three plots overlay IBU curves if their weights are present in the run directory.

---

## 11. Run Lifecycle

```shell
ran train --config params/1d_default.yaml
  └─ parse YAML → generate/load dataset → train (adversarial) → save to runs/<timestamp>/
       ├── config.json          (self-contained: stores full covariance matrices)
       └── artifacts/
            ├── generator.keras, discriminator.keras
            ├── history.npz
            ├── detector_level.pdf
            ├── particle_level.pdf
            ├── losses.pdf
            ├── selection.pdf
            ├── timings.json
            └── metrics.json

ran baseline ibu --run-dir runs/...       → artifacts/metrics_ibu.json, artifacts/ibu_weights.npz
ran train --load-run runs/...             → reload + re-plot with baseline overlays
ran report runs/...                       → artifacts/report.tex, report.pdf
```

The run root holds only what a person opens by hand -- `config.json` and,
later, `report.pdf`. Everything else -- checkpoints, arrays, figures, metrics
and timings JSON -- lives one level down, flat, in `artifacts/`.

`config.json` stores full covariance matrices (not just the original YAML scalars) so runs are self-contained and exactly reproducible without the original config file.

---

## 12. Tech Stack

- **Python 3.13** + **uv** (no pip, lockfile-based)
- **Keras 3 on the JAX backend** — training, `@jax.jit` on each disc/gen/eval step, float32 end to end
- **NumPy** — data generation and evaluation
- **Matplotlib** — publication-quality plots (serif font, ratio panels)
- **Typer** + **Rich** — one CLI command tree, structured logging and metrics tables
- **SLURM** — `scripts/submit.sh` for cluster runs

---

## 13. FAQ

**Q: How do you know the weights learned at reco level give the correct particle-level unfolding?**
The generator weights are a function of `z_gen` (particle-level). When `g` fools `d`, the reweighted reco distribution matches data. Because the detector smearing is applied on top of the particle-level variable, the weight function on particle space is self-consistent at both levels — it's the same weight for each MC event, applied wherever that event appears.

**Q: How is `g` held constant during the disc step?**
During the discriminator update, weights from `g(z)` must be treated as constants — otherwise gradients propagate through `g` during the disc update, mixing generator and discriminator gradients. This is standard GAN hygiene. The TensorFlow implementation used `tf.stop_gradient`; the JAX one does not need it. `disc_step` evaluates the weights outside the function that `jax.value_and_grad` differentiates, and that function is differentiated only with respect to the discriminator's trainable variables, so `g` is structurally excluded rather than gradient-blocked.

**Q: Why does early stopping maximize val D loss rather than minimize it?**
The discriminator loss measures how well it distinguishes data from sim. At the optimum (distributions matched), it reaches `log(2)`. A higher val D loss means the discriminator is doing better — closer to the equilibrium. We save the checkpoint where val D is highest (best convergence), not lowest.

**Q: What's the closure test?**
The Gaussian toy setup provides ground truth. We know exactly what the unfolded answer should be (the true distribution parameters). The particle-level plots directly compare `g`-reweighted `z_gen` against `z_true` — if the shapes match, the algorithm works correctly. For jets, we use the MC simulation itself as a surrogate for closure (treating one MC sample as "data").

**Q: How do you prove z_true doesn't leak into g?**
See Section 3 — the data poisoning check. We run two training trials: one with real `z_true`, one where every data particle-level value is overwritten with -999. Because `x_data` is fixed before poisoning, the discriminator sees identical data in both runs. If `g` had any access to `z_true`, the poisoned run would produce garbage weights. In practice both runs produce the same unfolding quality, confirming the leakage path doesn't exist. It's a stronger check than code inspection because it validates the full data pipeline.

**Q: How does RAN differ from OmniFold?**
OmniFold is iterative (Step 1 at reco, Step 2 at particle level, repeat). RAN is a simultaneous adversarial game — `g` and `d` train together, with `g` directly outputting particle-level weights. RAN also has a cleaner architectural separation: one network per level, trained end-to-end.

**Q: Why normalize weights to mean=1?**
Normalization preserves the total number of events (integral of the distribution). Without it, a generator that simply outputs large weights everywhere would trivially reduce the discriminator's ability to distinguish counts, but not the shape.

**Q: Why use float32?**
The inputs do not justify anything wider: `mass` and `mult` are bit-exact through a float32 round trip and the other four observables lose exactly half a ULP, so there is no structure below float32 to preserve. The two concerns that argued for float64 were tested and did not survive it — weight normalization is stable in float32 out to a 10^24 dynamic range, and the batched reduction lands three orders of magnitude inside the early-stopping threshold, because summing per-batch then across batches is effectively pairwise summation. An ensemble of 20 paired seeds put the two precisions within ±0.5 percentage points of unfolding improvement.

Earlier notes here recorded float32 "showing numerical instability". That predates the move from `keras.ops` to plain `jnp`, and `keras.ops.mean` silently selecting a float32 compute dtype was a real bug in the old code path — the instability is better explained by that than by float32 itself.

The one place precision still shows: `log(1 - d + ε)` loses accuracy once the discriminator saturates past d ≈ 0.9999. That is a regime where the adversarial game has already broken down, and it degrades gracefully (the ε clamp floors it rather than producing NaN), but a loss curve read from a saturated run is less trustworthy in float32.

**Q: What happens with correlated dimensions?**
The generator takes the full `z_gen` vector as input and produces a single scalar weight per event. This means it naturally captures correlations — the weight for one event depends on all its particle-level features jointly. IBU cannot do this (it's 1D per variable). This is a key advantage of the ML approach.

---

## 14. Run Catalogue

Runs are organized into five experimental families. Each family demonstrates a distinct property of the codebase.

---

### Family 1 — Uncorrelated Gaussian, dimensionality scaling (Mar 14)

**What it shows:** RAN scales cleanly from 1D to 6D when the generator and true distributions differ only in mean and variance (no off-diagonal covariance). The simplest closure test.

**Setup:** `mu_gen = [0.5]*d`, `mu_true = [0]*d`, `sigma_gen = 0.9` (scalar → σ²I), `sigma_det = 0.5`. n=500k.

| Run                  | dim | WD improvement (particle level, all dims) |
| -------------------- | --- | ----------------------------------------- |
| `2026-03-14T051912Z` | 1   | 90.3%                                     |
| `2026-03-14T044110Z` | 2   | 95.7%, 94.8%                              |
| `2026-03-14T044210Z` | 3   | 93.0%, 95.3%, 96.8%                       |
| `2026-03-14T044109Z` | 4   | 96.3%, 97.3%, 96.3%, 94.6%                |
| `2026-03-14T060656Z` | 5   | 97.4%, 92.2%, 96.4%, 96.8%, 95.8%         |
| `2026-03-14T061333Z` | 6   | 96.2%, 97.5%, 95.8%, 94.6%, 84.5%, 88.4%  |

**Takeaway:** performance stays consistently above ~85% across all dimensions with uncorrelated structure. All runs have full baseline overlays (IBU).

---

### Family 2 — Low statistics (Mar 16)

**What it shows:** the sample efficiency floor. With only n=10,000 events/class (50× fewer than the standard runs), the algorithm struggles.

**Setup:** same Gaussian structure as Family 1, sigma_det=0.25 (tighter smearing).

| Run                  | dim | WD improvement (particle level) |
| -------------------- | --- | ------------------------------- |
| `2026-03-16T064923Z` | 1   | **−2.1%** (weights add noise)   |
| `2026-03-16T064938Z` | 2   | 18.9%, 22.0% (correlated)       |

**Takeaway:** at 10k the 1D run actually makes the distribution _worse_ — the generator doesn't have enough signal to learn a useful weight function and introduces variance. The 2D correlated run manages ~20% improvement but far below the 500k equivalent. These runs have no baselines — they predated the baseline infrastructure.

**Key talking point:** this motivates the 500k default. The threshold between "learning signal" and "noise amplification" is somewhere between 10k and 500k events for this problem.

---

### Family 3 — Correlated Gaussian (Mar 17–19)

**What it shows:** RAN handling fully off-diagonal covariance matrices — both `sigma_gen` and `sigma_true` are dense (non-diagonal). Tests the generator's ability to learn a joint weight function over correlated particle-level variables.

**Setup:** mu and covariance differ between gen and true; sigma_det is diagonal. n=500k.

**2D correlated** (`2026-03-17T181937Z` / `2026-03-18T235803Z`):

- sigma_gen has off-diagonal = −0.9 (strong negative correlation)
- sigma_true has off-diagonal = −0.702
- Particle-level WD: 94.6%, 89.1%

**4D correlated** (`2026-03-19T000651Z`):

- Full 4×4 covariance for both gen and true
- Particle-level WD: 86.2%, 86.3%, **59.1%**, 96.1%
- dim_2 is the hardest — the covariance structure creates a subspace that's difficult to reweight

**6D correlated** (`2026-03-19T001535Z`):

- Full 6×6 covariance; most complex Gaussian test
- Particle-level WD: 87.6%, 88.6%, 89.4%, 95.8%, 86.5%, 89.6%
- More uniform than 4D — the 6D covariance structure happens to be better conditioned

**Takeaway:** joint reweighting of correlated distributions works well overall, but performance varies per-dimension based on the specific covariance geometry. IBU cannot compete here — it is strictly 1D per variable and ignores correlations entirely.

---

### Family 4 — Real jet data, single variable (Mar 14)

**What it shows:** the first step from Gaussian toy to real physics — training on a single jet substructure variable (mass, `m`) downloaded from Zenodo. Real data means no analytic ground truth; closure is assessed visually.

| Run                  | var | WD detector     | WD particle |
| -------------------- | --- | --------------- | ----------- |
| `2026-03-14T061023Z` | m   | 88.5%           | 82.8%       |
| `2026-03-14T061506Z` | m   | (duplicate run) |             |

**Takeaway:** the algorithm transfers to real jet data immediately without code changes. Single-variable case also lets IBU compete directly (it's 1D), establishing a fair baseline.

---

### Family 5 — Real jet data, all 6 variables (Mar 17–20)

**What it shows:** the full physics result — joint unfolding of all six jet substructure observables simultaneously. RAN learns a single weight function over the 6D particle-level space.

**Best run** (`2026-03-20T155247Z`, full baselines):

| Variable         | WD detector | WD particle | Note                   |
| ---------------- | ----------- | ----------- | ---------------------- |
| m (mass)         | 94.9%       | **16.8%**   | Particle level hardest |
| M (multiplicity) | 94.3%       | 84.7%       |                        |
| w (width)        | 98.2%       | 97.3%       | Best overall           |
| τ₂₁              | 95.5%       | 75.9%       |                        |
| z_g              | 83.2%       | 84.4%       | Lowest detector        |
| ln ρ (sdm)       | 95.0%       | 95.2%       |                        |

**Notable pattern:** detector-level improvements are uniformly high (83–98%) across all variables, but particle-level is much more variable (17–97%). Jet mass (`m`) is the striking outlier: 94.9% at detector but only 16.8% at particle level. This is physically meaningful — the detector smearing for mass is large relative to the gen/true difference, so the reweighted reco distribution matches data easily, but the particle-level correction is subtle and the generator's signal is weaker.

Earlier jets runs:

- `2026-03-17T181755Z` — missing `metrics.json` (evaluation crashed, baselines still present)
- `2026-03-19T172653Z` — same
