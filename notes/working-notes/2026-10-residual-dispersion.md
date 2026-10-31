# Residual dispersion penalty: results

Plan: `notes/superpowers/plans/2026-10-30-residual-dispersion-penalty.md`.
Split script: `benchmarks/dispersion_split.py`.

**Status: Gaussian done, jets pending (HPC).**

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

## Jets (pending)

Run on the HPC node:

```sh
for lam in 0.015 0.05; do
  uv run deconvolve train -D jets --n-samples 1600000 --hidden-units 128 --n-layers 3 \
    --seed 337443673 --dispersion-mode residual --lambda-dispersion $lam --no-plots
done
uv run python benchmarks/dispersion_split.py runs/2026-10-30T203443Z runs/2026-10-31T013457Z \
  runs/<two new runs> --out jets_split_residual.json
```

Also report the trained h's explained fraction for each residual run, as was done for the Gaussian above.

## Verdict (provisional, Gaussian only)

Keep residual mode **opt-in**. It's a better-aimed regularizer than total mode, but it isn't free on a problem whose truth has invisible structure. Whether it should become the default depends on jets: ship it if it matches or beats total-mode λ=0.015 on the joint particle metric.
