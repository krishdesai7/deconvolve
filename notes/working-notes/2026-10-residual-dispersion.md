# Residual dispersion penalty: results

Plan: `notes/superpowers/plans/2026-10-30-residual-dispersion-penalty.md`.
Split script: `benchmarks/dispersion_split.py`.

**Status: Gaussian and jets (one seed) done. Seed check pending (HPC).**

## 1D Gaussian (`params/1d_default.yaml`, seed 446727034)

All three arms use the same seed and data. Variances are on the MC test split, with `E[w|x]` fit by the split script's MLP.

| arm | particle W1 | particle JS | Var(w) | visible | residual | ESS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| total, λ=0 | 0.0157 | 0.00095 | 0.483 | 0.334 | 0.157 | 67.4% |
| total, λ=0.015 | 0.0978 | 0.00243 | 0.272 | 0.196 | 0.077 | 78.6% |
| **residual, λ=0.015** | **0.0353** | 0.00120 | 0.388 | 0.271 | 0.115 | 72.1% |
| ideal w* (analytic) | 0 | 0 | 0.537 | 0.350 | 0.187 | |

**Does the trained h converge?** Yes. On the MC test split, `regressor.keras` has a correlation of 0.9975 with a 400-bin `E[w|x]`. `E[(h − E[w|x])²]` is 0.0014, against a residual of 0.115. So the penalty really is `E[Var(w|x)]`, not something drifting back toward `Var(w)` (the final review's Important item).

**Reading.** Residual mode removes most of the damage: particle W1 goes from 0.098 to 0.035, but λ=0 reaches 0.016. The plan expected visible variance within 0.02 of λ=0's 0.334. It came out 0.271, so that expectation is **not met**. The cause is not h. The ideal weights carry 0.187 of residual themselves, and for a well-posed deconvolution (Gaussian smearing is injective) visible and residual are not independent knobs. w is a single function of z, and its tails carry both, so shrinking the residual also flattens the visible part. The bias is about a third of the total-mode bias, as the analysis predicted before the build. It isn't zero.

## Jets (seed 337443673, 1.6M events, 128×3)

The split comes from `jets_split_residual.json`. "Particle SW" is the joint particle-level sliced-Wasserstein improvement from `metrics.json`.

| run | mode, λ | particle SW | Var(w) | visible | residual | reco gap | ESS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-10-31T013457Z | total, 0 | 77.3% | 0.385 | 0.168 | 0.218 | 0.0156 | 72.2% |
| 2026-10-30T203443Z | **total, 0.015** | **87.8%** | 0.192 | 0.119 | 0.073 | 0.0150 | 83.9% |
| 2026-10-31T175315Z | residual, 0.015 | 82.4% | 0.232 | 0.148 | 0.085 | 0.0155 | 81.2% |
| 2026-10-31T175423Z | residual, 0.05 | 84.0% | 0.157 | 0.118 | 0.039 | 0.0195 | 86.5% |
| oracle w* | | | 0.275 | 0.163 | 0.114 | 0.0291 | 78.4% |
| ω(x) | | | | 0.141 | | | |

**Reading.** Residual mode loses to total λ=0.015 at both λ values. Within residual mode, more λ helps (82.4% at 0.015, 84.0% at 0.05). But the λ=0.05 run shrinks past the truth on every axis:
- its residual is a third of the oracle's;
- its visible part falls below Var(ω), the floor that matching the reco-level ratio requires;
- its reco gap grows, so the discriminator can see it.

It still unfolds better at particle level. So on jets, λ works as ordinary shrinkage against generator estimation noise, not as "remove what the detector can't see". Total mode, which also shrinks the visible part, does better than residual mode, which preserves it.

**Hypothesis (untested).** The best λ is set by estimation variance: the dimension of z, the sample size and generator capacity. It is not set by response non-universality. On the 1D Gaussian the generator has little to overfit and λ=0 wins. With a 12-dimensional z, λ>0 wins.

**Caveats.**
- Each arm is one seed.
- The split references are classifier estimates and move between runs: the oracle residual was 0.166 in `jets_split.json` and 0.114 here. Compare the variance columns only within a single file.

## Seed check (pending, HPC)

Two more seeds each for total λ=0.015 and residual λ=0.05. With the seed-337443673 runs above, that makes three per arm:

```sh
for seed in 20261031 20261101; do
  for arm in total:0.015 residual:0.05; do
    uv run deconvolve train -D jets --n-samples 1600000 --hidden-units 128 --n-layers 3 \
      --seed $seed --dispersion-mode ${arm%%:*} --lambda-dispersion ${arm##*:} --no-plots
  done
done
```

## Verdict (provisional)

Residual mode wins neither benchmark: λ=0 wins the Gaussian and total λ=0.015 wins jets.

- **If the seed check finds no residual-mode advantage beyond the seed spread:** leave `feat/residual-dispersion` unmerged as the record, and land only this note.
- **Next open question either way:** choosing λ without truth. If λ is a bias–variance knob, held-out reco-level MMD (already computed for checkpoint selection) is the natural criterion to test.
