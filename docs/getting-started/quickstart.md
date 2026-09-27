# Quickstart

This page walks through training, evaluating, and reporting an unfolding run, first on a 1D Gaussian toy model, then on a jet substructure dataset.

---

## 1. Train a 1D Gaussian Model

This example trains on a 1D Gaussian toy model with an analytically known truth. Generation is distributed as \(Z_{\text{Gen.}} \sim \mathcal{N}(\mu = 0.5, \sigma = 0.9)\), while Truth is distributed as \(Z_{\text{Truth}} \sim \mathcal{N}(\mu = 0.0, \sigma = 1.0)\).

```shell
deconvolve train --config params/1d_default.yaml
```

This logs per-epoch discriminator/generator loss and validation Maximum Mean Discrepancy, then reports the epoch selected as the best checkpoint. It writes to a run directory `runs/<yyyy-mm-dd>T<hhmmss>Z/`.

### CLI overrides

Any training hyperparameter can be set through the command line arguments, overriding the configured defaults:

```shell
# 128 hidden units, 3 layers, 200 epochs
$ deconvolve train --config params/1d_default.yaml -u128 -l3 -e200

# name the run directory explicitly
$ deconvolve train --config params/1d_default.yaml --run-dir runs/test-run
```

See the [CLI Reference](../user-guide/cli.md) for the full list of command line arguments and options.

---

## 2. Evaluate the Run

```shell
deconvolve evaluate runs/test-run
```

This computes the following three metrics before and after reweighting,

1. **1D Wasserstein-1 distance**:

    \[
    W_1(p, q) = \int_{-\infty}^\infty \left\vert{} \int_{-\infty}^t (p(x) - q(x)) \, \d x \right\vert{} \d t
    \]

2. **Jensen-Shannon divergence**:

    \[
    D_{\text{JS}}(p, q) = \frac{1}{2} \int_{-\infty}^\infty \left( p(x) \ln \frac{p(x)}{m(x)} + q(x) \ln \frac{q(x)}{m(x)} \right) \d x
    \]

    where the mixing distribution is \(m(x) = \frac{1}{2}(p(x) + q(x))\).

3. **Triangular discriminator (Vincze-LeCam divergence)**:

    \[
    \Delta(p, q) = \int_{-\infty}^\infty \frac{(p(x) - q(x))^2}{p(x) + q(x)} \d x
    \]

and writes the result to `<run-dir>/artifacts/metrics.json`.

---

## 3. Generate a Report

```shell
deconvolve report --run-dir <run-dir>
```

This command generates a report with the computed metrics for before and after reweighting, and plots of the probability density function comparisons, loss curves, and the MMD trajectory, and compiles `report.tex`/`report.pdf` under `<run-dir>/artifacts/`.

---

## 4. Jet Substructure

This example trains on a set of twelve jet substructure observables (Zenodo, downloaded and cached under `.cache/` on first use):

```shell
# all twelve variables
deconvolve train -Djets

# a subset, e.g. mass and width
deconvolve train -Djets -vm -vw
```
