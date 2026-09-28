# Hyperparameters

## A. Architecture

Files: `models.py` passed through `train()`.

- Width: no reason $g$ and $d$ must share `hidden_units`; split into `hidden_units_g`/`hidden_units_d`
- Depth: same argument, `n_layers_g`/`n_layers_d`
- Hidden activation family: `relu` (current), `leaky-relu`, `elu`, `gelu`, `swish/silu`, $\tanh$, some of which are not a single function but a real valued family of functions -- some of which are not a single function but a real valued family of functions -- and no reason $g$ and $d$ must use the same one
- $g$'s output activation: this is an open, function-valued axis, not a discrete choice: anything surjecting onto $\mathbb{R}_{\geq 0}$ with tame gradients is a candidate. `softplus` (current), `relu`, `elu`$+1$, $x^2$, $|x|$, `logplus` $:= \log1p\, \circ\,$`softplus`, a learned/scaled variant --- I've seen suggestions for $\log1p\,\circ\,(e\times$`softplus`$)$ in particular as a variant of my original `logplus`. One could even try `exp` but I'd deprioritize it because the weights and gradients are likely to grow too large.
- Normalization layers: `BatchNorm`, `LayerNorm`, `GroupNorm`, none (current) — placement (pre- vs post-activation) and whether $g$/$d$ use the same choice are both sub-axes
- Regularization mechanisms, each independently toggleable per network:
  - Dropout (rate, which layers)
  - Spectral normalization: specifically well-known for stabilizing the discriminator in GANs
  - Weight decay/L2 (separate coefficient for $g$ vs $d$)
  - Gradient penalty on $d$ (WGAN-GP style): this is really a loss-level choice, see below, but the mechanism lives in the architecture/forward pass
  - Entropy regularization on $g$'s weight distribution: discourages collapse to degenerate reweightings
  - Label smoothing: technically a loss-side trick but changes what $d$ is asked to fit
- Skip/residual connections: plain stacked `Dense` (current) vs residual blocks
- Weight initialization scheme: `Glorot/He/orthogonal`, and init scale
- Bias terms: whether each `Dense` layer carries a bias --- I don't expect this to be a major factor in the model's performance, but it's a real degree of freedom.

## B. Loss/adversarial objective

File: `train.py`

- GAN loss family: weighted BCE/non-saturating (current), least-squares (LSGAN), Wasserstein (WGAN --- requires dropping $d$'s sigmoid and swapping to a critic + either weight clipping or gradient penalty), hinge loss
- Weight normalization scheme: current preserves MC count per batch; alternatives include capping/clipping extreme weights (standard in importance-sampling/reweighting literature to control variance), or normalizing over the full epoch rather than per-batch
- Instance noise: injecting noise into $d$'s inputs, classic early-GAN-training stabilizer
- Feature matching: auxiliary generator loss matching $d$'s intermediate activations instead of pure adversarial signal (Salimans et al.)
- Label smoothing target values: soften $y=1$ and $y=0$ to e.g. $0.9$ and $0.1$ respectively.

## C. Optimization

File: `train.py`

- $\textrm{lr}_g$, $\textrm{lr}_d$, and their ratio
- Optimizer family: `Adam` (current), `RMSprop` (the classical GAN optimizer pre-Adam, still often used for W-GANs), `AdamW` (decoupled weight decay), `SGD`+momentum
- `Adam`'s $\beta_1, \beta_2, \epsilon$: separately for $\textrm{opt}_g$ and $\textrm{opt}_d$
- LR schedule: constant (current) vs decay (cosine/step/exponential) vs warmup
- Gradient clipping: norm or value clipping, absent today, a standard GAN-stability lever
- `n_disc_steps`: plus the more general idea of a schedule that changes over training rather than a fixed ratio, and the TTUR trick (two-time-scale update — different fixed LRs achieving a similar effect to more disc steps). And `n_disc_steps` doesn't necessarily need to be an integer - it could be a rational number. I've seen papers use 3 discriminator updates interleaved with 2 generator updates for example.
- EMA of $g$'s weights: maintaining a shadow generator for eval/inference, common in modern GAN training and orthogonal to the raw optimization trajectory

## D. Training loop and model selection

File: `train.py`

- `n_epochs`, `patience`, `min_delta`
- Model-selection criterion: currently "best" = lowest `val_d` (the BCE $d$ achieves on the validation set). Could instead select on distance-to-log(2) (closest to $d$ being maximally confused, which is actually the adversarial equilibrium condition, not minimal BCE), or on a held-out particle-level Wasserstein directly
- Warm-start/pretraining schedule: e.g., pretrain $d$ for several epochs before $g$ starts updating, to avoid potential early-training instability where $d$ is too weak to give useful gradient

## E. Data/sampling

Files: `datasets.py`, `device.py`

- `n_samples`: dataset size (CLI default 500,000)
- `val_fraction/test_fraction`: split ratios, hardcoded 0.1 / 0.2 in RANDataset.**init**, not CLI-exposed. Should we have a fixed val split in the first place or should we instead use cross-validation?
- `eval_batch_size`: DEFAULT_EVAL_BATCH_SIZE = 8192, hardcoded; affects only eval-pass batching, not training
- `seed`: init-weight seed (nuisance axis, per CLAUDE.md's Seeding section)
- `data_seed`: shuffle/split/batch-order seed (nuisance axis)
