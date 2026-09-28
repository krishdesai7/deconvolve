# MMD Model Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace GAN-loss-based checkpoint selection and early stopping with a weighted MMD computed after training, over a retained per-epoch parameter history.

**Architecture:** The epoch loop becomes a `lax.scan` over a fixed `n_epochs` that emits each epoch's model parameters. Selection moves out of the traced program entirely and runs on the host: build a precomputed-kernel MMD cache from a fixed subsample of the validation split, evaluate the weighted MMD for every epoch's generator, restore the argmin. Because selection is post-hoc, the particle-level MMD diagnostic can be computed without ever putting `z_true` inside the trace.

**Tech Stack:** JAX (`lax.scan`, `jax.jit`), Keras 3 on the JAX backend, jaxtyping + beartype on array seams, pytest, Typer CLI.

**Spec:** `docs/superpowers/specs/2026-08-25-mmd-model-selection-design.md`

## Global Constraints

- Python >= 3.13, managed with `uv` (no pip). Run everything as `uv run ...`.
- **float32 end to end.** `EVENT_DTYPE` in `src/ran/rantypes/constants.py` is the single pin. No `astype` on containers, no dtype parameters.
- **No network may ever see `z_true`.** In the interleaved `ZXY` form, `z[y == 1]` is truth and `z[y == 0]` is `z_gen`. `src/ran/train.py` may only index `z` with `y == 0`.
- ruff lint + format, pyrefly `--min-severity info` at zero diagnostics, complexipy max complexity 10. Validate with `uv format --check`, `uv run --locked ruff check`, `uv run --locked pyrefly check --min-severity info`, `uv run --locked complexipy`, `uv run --locked pytest -q`.
- Kebab-case CLI flags. Every knob that changes a run is recorded in `config.json`.
- Multi-scale RBF bandwidths: median heuristic times `(0.5, 1/sqrt2, 1.0, sqrt2, 2.0)`.
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

### Task 1: The MMD estimator

Self-contained, pure, host-side. Knows nothing about training, datasets, or Keras.

**Files:**
- Create: `src/ran/mmd.py`
- Create: `tests/test_mmd.py`
- Modify: `src/ran/rantypes/__init__.py` (no change needed — `mmd.py` imports from it, not the reverse)

**Interfaces:**
- Consumes: `EVENT_DTYPE` from `ran.rantypes`.
- Produces:
  - `MMDCache(k_yy, v_xy, diag_yy, term_xx)` — a NamedTuple of four arrays only, so it is a clean JAX pytree and `weighted_mmd` can be jitted with it as an argument. Bandwidths are deliberately **not** a field; they are static and would be traced as leaves.
  - `squared_distances(a, b, /) -> Float[Array, "n m"]`
  - `median_bandwidth(x, /) -> float`
  - `bandwidths(x, /, *, scales: Sequence[float] = _SCALES) -> tuple[float, ...]`
  - `subsample_indices(seed: int, n: int, m: int, /) -> NDArray[np.intp]`
  - `build_cache(x_data, y_mc, /, *, sigmas: Sequence[float]) -> MMDCache`
  - `weighted_mmd(cache: MMDCache, raw_w, /) -> tuple[Float[Array, ""], Float[Array, ""]]` returning `(mmd2, ess)`
  - `mmd_curve(cache: MMDCache, raw_w, /) -> tuple[NDArray[np.double], NDArray[np.double]]` taking an `(epochs, m)` array

- [ ] **Step 1: Write the failing tests**

Create `tests/test_mmd.py`:

```python
"""Tests for the weighted MMD estimator.

MMD is a divergence -- zero iff the distributions match, monotone in
mismatch -- which is the property the validation BCE does not have and the
reason selection is built on this instead.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from ran.mmd import (
    bandwidths,
    build_cache,
    median_bandwidth,
    mmd_curve,
    squared_distances,
    subsample_indices,
    weighted_mmd,
)


def _samples(n=1024, d=6, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return jnp.asarray(rng.normal(shift, 1.0, (n, d)), dtype=jnp.float32)


def _cache(x, y):
    return build_cache(x, y, sigmas=bandwidths(x))


class TestSquaredDistances:
    def test_matches_the_broadcast_form(self) -> None:
        """The expansion is an optimization, so it must be exact.

        `sum((a[:,None,:] - b[None,:,:])**2, -1)` is the obvious form and
        materializes an (n, m, d) intermediate -- 6.4 GB at m=16384. The
        expansion below is (n, m) only, and this pins that the saving is free.
        """
        a, b = _samples(64, 3, seed=1), _samples(48, 3, shift=0.7, seed=2)
        naive = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
        np.testing.assert_allclose(
            np.asarray(squared_distances(a, b)), np.asarray(naive), atol=1e-4
        )

    def test_self_distance_is_zero_on_the_diagonal(self) -> None:
        a = _samples(32, 4, seed=3)
        np.testing.assert_allclose(
            np.diag(np.asarray(squared_distances(a, a))), 0.0, atol=1e-4
        )


class TestBandwidths:
    def test_median_heuristic_puts_the_kernel_at_exp_minus_one(self) -> None:
        """sigma = sqrt(median/2) makes k(median distance) = exp(-1)."""
        x = _samples(512, 6, seed=4)
        sigma = median_bandwidth(x)
        med = float(jnp.median(squared_distances(x, x)))
        np.testing.assert_allclose(float(jnp.exp(-med / (2 * sigma**2))), np.exp(-1.0), rtol=1e-5)

    def test_five_scales_bracket_the_median(self) -> None:
        x = _samples(256, 6, seed=5)
        sig = bandwidths(x)
        assert len(sig) == 5
        assert sig[0] < sig[2] < sig[4]
        np.testing.assert_allclose(sig[2], median_bandwidth(x), rtol=1e-6)


class TestWeightedMMD:
    def test_identical_distributions_score_near_zero(self) -> None:
        x, y = _samples(1024, 6, seed=6), _samples(1024, 6, seed=7)
        mmd, _ = weighted_mmd(_cache(x, y), jnp.ones(1024))
        assert abs(float(mmd)) < 5e-3

    def test_a_shifted_distribution_scores_far_higher(self) -> None:
        x, y = _samples(1024, 6, seed=8), _samples(1024, 6, shift=0.5, seed=9)
        near, _ = weighted_mmd(_cache(x, _samples(1024, 6, seed=10)), jnp.ones(1024))
        far, _ = weighted_mmd(_cache(x, y), jnp.ones(1024))
        assert float(far) > 10 * abs(float(near))

    def test_exact_importance_weights_undo_a_shift(self) -> None:
        """The known answer: N(0,1)/N(0.5,1) weights must recover the match."""
        x, y = _samples(2048, 6, seed=11), _samples(2048, 6, shift=0.5, seed=12)
        cache = _cache(x, y)
        unweighted, _ = weighted_mmd(cache, jnp.ones(2048))
        logw = -0.5 * jnp.sum(y**2, 1) + 0.5 * jnp.sum((y - 0.5) ** 2, 1)
        weighted, _ = weighted_mmd(cache, jnp.exp(logw - logw.max()))
        assert float(weighted) < 0.2 * float(unweighted)

    def test_matches_a_fully_materialized_reference(self) -> None:
        """The precomputation must be an optimization and nothing more.

        The reference below stores `k_xy` and forms `w[:,None]*w[None,:]*k_yy`
        outright -- 3.22 GB and 30 ms/eval at m=16384 against 1.07 GB and
        2.8 ms. If the two ever disagree, the collapse of `k_xy` to a vector
        is wrong, not merely slower.
        """
        x, y = _samples(512, 6, seed=28), _samples(512, 6, shift=0.4, seed=29)
        sig = bandwidths(x)
        rng = np.random.default_rng(30)
        w = jnp.asarray(rng.uniform(0.1, 3.0, 512), dtype=jnp.float32)

        def _k(a, b):
            d2 = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
            return sum(jnp.exp(-d2 / (2 * s**2)) for s in sig)

        k_xx, k_yy, k_xy = _k(x, x), _k(y, y), _k(x, y)
        q = w / jnp.sum(w)
        sq = jnp.sum(q**2)
        term_xx = (jnp.sum(k_xx) - jnp.trace(k_xx)) / (512 * 511)
        term_yy = (
            jnp.sum(q[:, None] * q[None, :] * k_yy) - jnp.sum(q**2 * jnp.diag(k_yy))
        ) / (1.0 - sq)
        reference = term_xx + term_yy - 2.0 * jnp.sum(q[None, :] * k_xy) / 512

        got, _ = weighted_mmd(build_cache(x, y, sigmas=sig), w)
        np.testing.assert_allclose(float(got), float(reference), rtol=1e-5)

    def test_rescaling_the_raw_weights_changes_nothing(self) -> None:
        """Only normalized weights enter, so callers need not pre-normalize."""
        x, y = _samples(512, 6, seed=13), _samples(512, 6, shift=0.3, seed=14)
        cache = _cache(x, y)
        rng = np.random.default_rng(15)
        w = jnp.asarray(rng.uniform(0.1, 3.0, 512), dtype=jnp.float32)
        a, _ = weighted_mmd(cache, w)
        b, _ = weighted_mmd(cache, w * 1000.0)
        np.testing.assert_allclose(float(a), float(b), rtol=1e-5)

    def test_ess_is_the_inverse_sum_of_squared_normalized_weights(self) -> None:
        x, y = _samples(256, 6, seed=16), _samples(256, 6, seed=17)
        cache = _cache(x, y)
        _, flat = weighted_mmd(cache, jnp.ones(256))
        np.testing.assert_allclose(float(flat), 256.0, rtol=1e-4)
        w = jnp.asarray(np.concatenate([np.ones(1), np.full(255, 1e-6)]), dtype=jnp.float32)
        _, spiked = weighted_mmd(cache, w)
        assert float(spiked) < 2.0

    def test_collapsed_weights_return_infinity_not_a_blow_up(self) -> None:
        """As ESS -> 1 the unbiased estimator is undefined: one effective
        sample cannot estimate E[k(y,y')]. Selection must never pick it."""
        x, y = _samples(256, 6, seed=18), _samples(256, 6, seed=19)
        w = np.zeros(256, dtype=np.float32)
        w[0] = 1.0
        mmd, ess = weighted_mmd(_cache(x, y), jnp.asarray(w))
        assert np.isinf(float(mmd))
        assert not np.isnan(float(mmd))
        np.testing.assert_allclose(float(ess), 1.0, rtol=1e-4)

    def test_float32_matches_a_float64_reference(self) -> None:
        """MMD^2 is a small difference of O(1) terms, so cancellation is the
        worry. XLA's blocked reduction absorbs it; this pins that."""
        x32, y32 = _samples(1024, 6, seed=20), _samples(1024, 6, shift=0.4, seed=21)
        rng = np.random.default_rng(22)
        w = rng.uniform(0.1, 3.0, 1024)
        sig = bandwidths(x32)
        a, _ = weighted_mmd(build_cache(x32, y32, sigmas=sig), jnp.asarray(w, jnp.float32))
        b, _ = weighted_mmd(
            build_cache(np.float64(x32), np.float64(y32), sigmas=sig), jnp.asarray(w)
        )
        np.testing.assert_allclose(float(a), float(b), atol=1e-6)

    def test_monotone_along_the_path_to_the_exact_weights(self) -> None:
        """Outside the resolution floor, MMD must fall as the weights improve.

        The floor is real: below ~5e-4 in MMD^2 at m=8192 the ranking inverts,
        because the empirical MMD is minimized by weights matching the sample
        rather than the distribution. `t` stops at 0.9 to stay above it.
        """
        x, y = _samples(2048, 6, seed=23), _samples(2048, 6, shift=0.5, seed=24)
        cache = _cache(x, y)
        logw = -0.5 * jnp.sum(y**2, 1) + 0.5 * jnp.sum((y - 0.5) ** 2, 1)
        star = jnp.exp(logw - logw.max())
        star = star / jnp.sum(star)
        flat = jnp.full(2048, 1.0 / 2048)
        vals = [float(weighted_mmd(cache, (1 - t) * flat + t * star)[0]) for t in np.linspace(0, 0.9, 10)]
        assert all(b < a for a, b in zip(vals, vals[1:], strict=True)), vals


class TestCurve:
    def test_curve_matches_evaluating_each_row(self) -> None:
        x, y = _samples(512, 6, seed=25), _samples(512, 6, shift=0.3, seed=26)
        cache = _cache(x, y)
        rng = np.random.default_rng(27)
        w = jnp.asarray(rng.uniform(0.1, 3.0, (7, 512)), dtype=jnp.float32)
        mmds, esss = mmd_curve(cache, w)
        assert mmds.shape == (7,) and esss.shape == (7,)
        for i in range(7):
            one_mmd, one_ess = weighted_mmd(cache, w[i])
            np.testing.assert_allclose(mmds[i], float(one_mmd), rtol=1e-5)
            np.testing.assert_allclose(esss[i], float(one_ess), rtol=1e-5)


class TestSubsample:
    def test_is_reproducible_and_within_range(self) -> None:
        a = subsample_indices(42, 1000, 100)
        b = subsample_indices(42, 1000, 100)
        np.testing.assert_array_equal(a, b)
        assert a.shape == (100,)
        assert len(np.unique(a)) == 100
        assert a.min() >= 0 and a.max() < 1000

    def test_a_different_seed_draws_differently(self) -> None:
        assert not np.array_equal(
            subsample_indices(1, 1000, 100), subsample_indices(2, 1000, 100)
        )

    def test_asking_for_more_than_exists_takes_everything(self) -> None:
        idx = subsample_indices(0, 50, 4096)
        assert idx.shape == (50,)
        assert len(np.unique(idx)) == 50
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_mmd.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'ran.mmd'`

- [ ] **Step 3: Write the implementation**

Create `src/ran/mmd.py`:

```python
"""Weighted maximum mean discrepancy, with everything constant precomputed.

This exists because a GAN's loss is not a model-selection signal. It
oscillates around its equilibrium by construction; a flat curve cannot be
told from a stalled one; and `log 2 - BCE` estimates a divergence only when
`d` is optimal, which nothing reports. MMD has the property the loss was
assumed to have: zero iff the distributions match, monotone in mismatch, and
no adversary or optimization involved -- it is a closed-form functional of
the weights.

Only the MC-side weights change between epochs, and almost everything else
collapses:

* `term_xx` is data-side only, so it is a *scalar*, computed once.
* `k_xy` never has to be stored. The data-side weights are uniform and fixed,
  so `mean_i k(x_i, y_.)` is a *vector* of length m. Storing the matrix costs
  1.07 GB at m=16384 and 10x the per-evaluation time, for nothing.
* `k_yy` is the only matrix that must survive.
* `diag(k_yy)` is exactly `len(sigmas)` for a sum of RBFs. It is stored
  anyway so the estimator stays correct if the kernel is ever swapped.

Squared distances use `||a||^2 + ||b||^2 - 2ab^T` rather than the obvious
broadcast subtraction, which would materialize an (n, m, d) intermediate --
6.4 GB at m=16384.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .rantypes import EVENT_DTYPE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Final

    from jaxtyping import Array, Float
    from numpy.typing import NDArray

# Bracketing the median heuristic. A single bandwidth is blind at every other
# scale, and a sum of RBFs is still a characteristic kernel, so this is free.
_SCALES: Final[tuple[float, ...]] = (0.5, 2.0**-0.5, 1.0, 2.0**0.5, 2.0)

# Below this the unbiased estimator's denominator is meaningless.
_MIN_DENOM: Final[float] = 1e-6


class MMDCache(NamedTuple):
    """Everything about a comparison that does not depend on the weights.

    Four arrays and nothing else, so this is a clean pytree and
    `weighted_mmd` can take it as a jitted argument. The bandwidths are
    static and are deliberately not a field: as leaves they would be traced.
    """

    k_yy: Float[Array, "m m"]
    v_xy: Float[Array, " m"]
    diag_yy: Float[Array, " m"]
    term_xx: Float[Array, ""]


def squared_distances(
    a: Float[Array, "n d"], b: Float[Array, "m d"], /
) -> Float[Array, "n m"]:
    """Pairwise squared distances via expansion, never an (n, m, d) tensor."""
    return (
        jnp.sum(a**2, axis=1)[:, None] + jnp.sum(b**2, axis=1)[None, :] - 2.0 * a @ b.T
    )


def median_bandwidth(x: Float[Array, "n d"], /) -> float:
    """`sigma` such that the kernel at the median distance is `exp(-1)`.

    Computed from the data side alone. `x_data` is fixed by `data_seed`, so
    every hyperparameter arm shares an identical kernel; a pooled heuristic
    would drift with the MC side and make arms incomparable.
    """
    return float(jnp.sqrt(jnp.median(squared_distances(x, x)) / 2.0))


def bandwidths(
    x: Float[Array, "n d"], /, *, scales: Sequence[float] = _SCALES
) -> tuple[float, ...]:
    median: float = median_bandwidth(x)
    return tuple(median * s for s in scales)


def subsample_indices(seed: int, n: int, m: int, /) -> NDArray[np.intp]:
    """A fixed, reproducible draw of at most `m` of `n` rows, without replacement."""
    return np.random.default_rng(seed).permutation(n)[: min(m, n)].astype(np.intp)


def _kernel(
    a: Float[Array, "n d"], b: Float[Array, "m d"], sigmas: Sequence[float], /
) -> Float[Array, "n m"]:
    d2: Float[Array, "n m"] = squared_distances(a, b)
    return sum(jnp.exp(-d2 / (2.0 * s**2)) for s in sigmas)


def build_cache(
    x_data: Float[Array, "n d"],
    y_mc: Float[Array, "m d"],
    /,
    *,
    sigmas: Sequence[float],
) -> MMDCache:
    """Precompute every weight-independent term of the comparison."""
    n: int = x_data.shape[0]
    k_xx: Float[Array, "n n"] = _kernel(x_data, x_data, sigmas)
    # The standard unbiased U-statistic: the diagonal is a self-comparison and
    # carries no information about the distribution.
    term_xx: Float[Array, ""] = (jnp.sum(k_xx) - jnp.trace(k_xx)) / (n * (n - 1))
    del k_xx
    k_yy: Float[Array, "m m"] = _kernel(y_mc, y_mc, sigmas)
    return MMDCache(
        k_yy=k_yy,
        v_xy=jnp.mean(_kernel(x_data, y_mc, sigmas), axis=0),
        diag_yy=jnp.diagonal(k_yy),
        term_xx=term_xx,
    )


@jax.jit
def weighted_mmd(
    cache: MMDCache, raw_w: Float[Array, " m"], /
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Unbiased weighted MMD-squared and the effective sample size.

    `raw_w` is the generator's raw output; normalization happens here so no
    caller has to remember which convention applies.

    ESS is returned rather than folded into the score. The biased estimator
    would absorb `sum(w^2)` into the metric and silently penalize weight
    concentration; given that the adversarial objective is linear in `w` and
    maximized at a simplex vertex, concentration is a thing to *measure*, not
    to mix into the number being minimized.
    """
    w: Float[Array, " m"] = raw_w / jnp.sum(raw_w)
    sum_w_sq: Float[Array, ""] = jnp.sum(w**2)
    denom: Float[Array, ""] = 1.0 - sum_w_sq

    # Double `where`: the guarded branch must not be evaluated at denom = 0,
    # because jnp.where computes both sides and a NaN would propagate.
    safe: Float[Array, ""] = jnp.where(denom > _MIN_DENOM, denom, 1.0)
    term_yy: Float[Array, ""] = (
        w @ (cache.k_yy @ w) - jnp.sum(w**2 * cache.diag_yy)
    ) / safe
    mmd2: Float[Array, ""] = cache.term_xx + term_yy - 2.0 * (cache.v_xy @ w)
    return jnp.where(denom > _MIN_DENOM, mmd2, jnp.inf), 1.0 / sum_w_sq


def mmd_curve(
    cache: MMDCache, raw_w: Float[Array, "epochs m"], /
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    """`weighted_mmd` over a stack of per-epoch weight vectors."""
    mmds, esss = jax.vmap(weighted_mmd, in_axes=(None, 0))(cache, raw_w)
    return np.asarray(mmds, dtype=np.double), np.asarray(esss, dtype=np.double)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_mmd.py -q`
Expected: PASS, all tests.

If `test_identical_distributions_score_near_zero` fails marginally, the tolerance is tracking the finite-sample floor, which at n=1024 is larger than at 8192 — check the value is order 1e-3, not that the test is wrong.

- [ ] **Step 5: Check the whole suite and the linters**

```bash
uv format && uv run --locked ruff check && uv run --locked pyrefly check --min-severity info && uv run --locked complexipy && uv run --locked pytest -q
```
Expected: format clean, lint clean, 0 pyrefly diagnostics, no function over complexity 10, all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/ran/mmd.py tests/test_mmd.py
git commit -m "feat: weighted MMD estimator with precomputed kernels

Only the MC-side weights change between epochs. term_xx collapses to a
scalar, k_xy to a length-m vector, and only k_yy stays a matrix -- 1.07 GB
rather than 3.22 GB at m=16384, and 2.8 ms rather than 30 ms per evaluation.

Unbiased, so ESS is reported separately rather than folded into the score.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Move the epoch loop to `scan` and selection to the host

A pure relocation of *where* selection happens. The criterion is still
`dist_log2`, so the selected epoch can be checked against the old in-loop
result. Early stopping goes away here, because a `scan` runs a fixed trip
count — which is what removes `patience` and `min_delta`.

**Files:**
- Modify: `src/ran/train.py` (`RunCarry`, `TrainResult`, `_make_epoch`, `_run`, `train`)
- Modify: `src/ran/workflow.py:219-260` (call site)
- Modify: `src/ran/leakage.py:81-83` (call site)
- Modify: `src/ran/experiments/cubic_sweep.py:167-175` (call site)
- Modify: `benchmarks/precision.py:102-110` (call site)
- Modify: `tests/test_train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: nothing from Task 1 yet.
- Produces:
  - `class EpochParams(NamedTuple)` with fields `g_trainable, g_non_trainable, d_trainable, d_non_trainable`, each a `Variables` (a `list[Array]`) whose arrays are stacked with a leading epoch axis.
  - `RunCarry(state: TrainState, key: PRNGKeyArray)` — two fields.
  - `TrainResult(g, d, history, seed, best_epoch, params)` where `params: EpochParams`.
  - `train(splits, dim, hidden_units, n_layers, seed, n_epochs=100, n_disc_steps=5, lr_g=1e-4, lr_d=1e-4, *, fused=True)` — **`patience`, `min_delta` and `criterion` are gone**.
  - `_restore(g: RANModel, d: RANModel, params: EpochParams, epoch: int, /) -> None` — used by Task 3.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train.py` (and delete `class TestSelectionCriterion` and the `SelectionCriterion`/`_selection_score`/`LOG2` imports it used — those go in Task 3; for now keep `LOG2`):

```python
class TestParameterHistory:
    """A run keeps every epoch's weights, which is what moves selection out."""

    @staticmethod
    def _splits(n: int = 512) -> DatasetSplits:
        rng = np.random.default_rng(31)
        z = rng.normal(size=(2 * n, 1)).astype(np.single)
        x = z + rng.normal(0, 0.4, size=(2 * n, 1)).astype(np.single)
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(z, x), y))

    def test_every_epoch_is_retained(self) -> None:
        result = train(
            self._splits(), dim=1, n_epochs=6, hidden_units=8, n_layers=1, seed=3
        )
        assert len(result.history["val_d"]) == 6
        for leaf in result.params.g_trainable:
            assert leaf.shape[0] == 6
        for leaf in result.params.d_trainable:
            assert leaf.shape[0] == 6

    def test_the_restored_model_holds_the_selected_epochs_weights(self) -> None:
        """`g` must carry epoch `best_epoch`'s parameters, not the last one's."""
        result = train(
            self._splits(), dim=1, n_epochs=6, hidden_units=8, n_layers=1, seed=4
        )
        for live, stacked in zip(
            result.g.trainable_variables, result.params.g_trainable, strict=True
        ):
            np.testing.assert_allclose(
                np.asarray(live.value), np.asarray(stacked[result.best_epoch]), rtol=1e-6
            )

    def test_selection_reproduces_the_old_in_loop_criterion(self) -> None:
        """Relocating selection must not change which epoch it picks.

        The in-loop version compared `-|val_d - log 2|` against the running
        best. With no `min_delta` that is exactly an argmin of the distance,
        so the host-side result must agree with it row for row.
        """
        result = train(
            self._splits(), dim=1, n_epochs=8, hidden_units=8, n_layers=1, seed=5
        )
        curve = np.asarray(result.history["val_d"], dtype=np.float64)
        assert result.best_epoch == int(np.argmin(np.abs(curve - LOG2)))

    def test_a_run_always_uses_its_full_epoch_budget(self) -> None:
        """`scan` has a fixed trip count: no early stop, so no patience."""
        rng = np.random.default_rng(32)
        n = 512
        z = rng.normal(size=(2 * n, 1)).astype(np.single)
        x = rng.normal(size=(2 * n, 1)).astype(np.single)  # x carries nothing
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        splits = RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(z, x), y))
        result = train(splits, dim=1, n_epochs=9, hidden_units=8, n_layers=1, seed=6)
        assert len(result.history["val_d"]) == 9
```

Replace `test_train_restores_best_weights_on_early_stop` entirely — early
stopping no longer exists — and update the two `functools.partial(train, ...)`
call sites in `TestFusion` and elsewhere to drop `patience=99`.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train.py -q`
Expected: FAIL — `TypeError: train() got an unexpected keyword argument` on the
old calls, and `AttributeError: 'TrainResult' object has no attribute 'params'`.

- [ ] **Step 3: Rewrite the loop in `src/ran/train.py`**

Replace `RunCarry`, add `EpochParams`, and rewrite `_make_epoch` and `_run`:

```python
class EpochParams(NamedTuple):
    """Every epoch's model parameters, stacked on a leading epoch axis.

    Both networks, not just `g`: nothing currently reads
    `discriminator.keras`, but an artifact directory whose two models come
    from different epochs is a trap. The non-trainable lists are empty for
    these Dense-only architectures, and are carried anyway so the day someone
    adds BatchNorm this stays correct rather than silently wrong.
    """

    g_trainable: Variables
    g_non_trainable: Variables
    d_trainable: Variables
    d_non_trainable: Variables


class RunCarry(NamedTuple):
    """What crosses an epoch boundary. Everything else is a `scan` output."""

    state: TrainState
    key: PRNGKeyArray
```

```python
def _make_epoch(
    data: DeviceSplits,
    one_pass: Callable[
        [TrainState, PRNGKeyArray],
        tuple[TrainState, Float[Array, ""], Float[Array, ""]],
    ],
    evaluate,
    *,
    n_epochs: int,
) -> Callable[[RunCarry, Int[Array, ""]], tuple[RunCarry, tuple]]:
    """Build the pure ``(RunCarry, epoch) -> (RunCarry, outputs)`` scan body.

    Nothing about model quality is decided here. The loop trains, records, and
    emits; selection is a host-side read of what it emitted, which is what
    keeps `z_true` out of the traced program entirely.
    """

    def _log(
        epoch: Int[Array, ""],
        train_d: Float[Array, ""],
        train_g: Float[Array, ""],
        val_d: Float[Array, ""],
    ) -> None:
        logger.info(
            "Epoch %3d/%d  D: %.4f  G: %.4f  | Val: %.4f",
            int(epoch) + 1,
            n_epochs,
            float(train_d),
            float(train_g),
            float(val_d),
        )

    def epoch(
        carry: RunCarry, epoch_idx: Int[Array, ""]
    ) -> tuple[RunCarry, tuple[Float[Array, " metrics"], EpochParams]]:
        key, subkey = jax.random.split(carry.key)
        state, train_d, train_g = one_pass(carry.state, subkey)
        val_d: Float[Array, ""] = evaluate(state, data.val)
        jax.debug.callback(_log, epoch_idx, train_d, train_g, val_d, ordered=True)
        row: Float[Array, " metrics"] = jnp.stack(arrays=[train_d, train_g, val_d])
        params = EpochParams(
            g_trainable=state.g_trainable,
            g_non_trainable=state.g_non_trainable,
            d_trainable=state.d_trainable,
            d_non_trainable=state.d_non_trainable,
        )
        return RunCarry(state=state, key=key), (row, params)

    return epoch
```

```python
def _run(
    carry: RunCarry, epoch, *, n_epochs: int, fused: bool
) -> tuple[RunCarry, Float[Array, "epochs metrics"], EpochParams]:
    """Drive the epoch loop, either inside XLA or from Python."""
    steps: Int[Array, " epochs"] = jnp.arange(n_epochs, dtype=jnp.int32)
    if fused:
        carry, (rows, params) = cast(
            "tuple[RunCarry, tuple[Array, EpochParams]]",
            jax.jit(lambda c, xs: lax.scan(epoch, c, xs))(carry, steps),
        )
        return carry, rows, params
    step: JitWrapped = jax.jit(epoch)
    rows_out: list[Array] = []
    params_out: list[EpochParams] = []
    for i in range(n_epochs):
        carry, (row, params) = step(carry, steps[i])
        rows_out.append(row)
        params_out.append(params)
    stacked = cast(
        "EpochParams",
        jax.tree.map(lambda *leaves: jnp.stack(leaves), *params_out),
    )
    return carry, jnp.stack(rows_out), stacked


def _restore(
    g: RANModel, d: RANModel, params: EpochParams, epoch: int, /
) -> None:
    """Write one epoch's parameters back into the live Keras models."""
    chosen = cast(
        "EpochParams", jax.tree.map(lambda leaf: leaf[epoch], params)
    )
    _assign(g.trainable_variables, values=chosen.g_trainable)
    _assign(g.non_trainable_variables, values=chosen.g_non_trainable)
    _assign(d.trainable_variables, values=chosen.d_trainable)
    _assign(d.non_trainable_variables, values=chosen.d_non_trainable)
```

Rewrite the tail of `train()` — drop the `patience`/`min_delta`/`criterion`
parameters from the signature, drop `still_running`, and select on the host:

```python
    final, rows, params = _run(carry, epoch, n_epochs=n_epochs, fused=fused)
    history: dict[str, list[float]] = _unpack_history(rows)

    curve: NDArray = np.abs(np.asarray(history["val_d"], dtype=np.double) - LOG2)
    best_epoch: int = int(np.argmin(curve))
    logger.info("Restoring epoch %d of %d", best_epoch + 1, n_epochs)
    _restore(g, d, params, best_epoch)

    logger.info("Init seed %d", seed)
    return TrainResult(g, d, history, seed, best_epoch, params)
```

`RunCarry` at construction becomes `RunCarry(state=state, key=jax.random.key(data.data_seed))`.

Delete the now-unused final test-split BCE line (`test: float = ...`) — it
scored the in-loop criterion's checkpoint and Task 3 replaces it with the
honest test-split MMD.

- [ ] **Step 4: Update every `train()` call site**

Remove `patience=` from all four:

- `src/ran/workflow.py` — drop `patience` from the `train(...)` call and from `run()`'s signature and the `hyperparameters` dict; unpack six fields: `g, d, history, init_seed, best_epoch, _params = train(...)`.
- `src/ran/leakage.py:81` — `train(splits, dim=1, hidden_units=32, n_layers=2, seed=init_seed).g`
- `src/ran/experiments/cubic_sweep.py:167` — delete the `patience=_PATIENCE` argument and the now-unused `_PATIENCE` constant.
- `benchmarks/precision.py:102` — delete `patience=99`.
- `src/ran/cli.py` — drop the `--patience` option and stop forwarding it.

`tests/test_cubic_sweep.py`'s `fake_train` already accepts `**_kwargs` and
`TrainResult`'s new fields are defaulted, so it needs no change — verify.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: PASS. `TestFusion::test_fused_and_eager_runs_agree` is the important
one — it proves the Python-loop path stacks outputs identically to `scan`.

- [ ] **Step 6: Linters, then commit**

```bash
uv format && uv run --locked ruff check && uv run --locked pyrefly check --min-severity info && uv run --locked complexipy && uv run --locked pytest -q
git add -A
git commit -m "refactor: scan the epoch loop and select on the host

A while_loop cannot emit per-epoch outputs without a preallocated buffer,
and early stopping is worth less wall clock than compiling the loop that
implements it -- 0.034s/epoch against a 4.6s compile. Fixing the trip count
lets scan stack every epoch's parameters, which moves selection out of the
traced program.

Selection is still dist_log2 here, so the epoch it picks can be checked
against the in-loop result. patience and min_delta go with the while_loop.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Select on detector-level MMD; delete the BCE criteria

**Files:**
- Modify: `src/ran/train.py` (selection, imports)
- Modify: `src/ran/rantypes/enums.py` (delete `SelectionCriterion`)
- Modify: `src/ran/rantypes/__init__.py` (delete the export)
- Modify: `src/ran/cli.py` (delete `--criterion`, `--min-delta`)
- Modify: `src/ran/workflow.py` (drop `criterion`/`min_delta`)
- Modify: `CLAUDE.md` (Training Loop section)
- Modify: `src/ran/README.md` (`RunCarry` entries)
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `build_cache`, `bandwidths`, `subsample_indices`, `mmd_curve`, `MMDCache` from `ran.mmd` (Task 1); `EpochParams`, `_restore` from Task 2.
- Produces:
  - `MMD_SUBSAMPLE: Final[int] = 16384`
  - `_detector_arrays(zxy, seed, m) -> tuple[EventArray, EventArray, EventArray]` returning `(x_data, x_sim, z_gen)`, all subsampled. **Only ever indexes `z` where `y == 0`.**
  - `_weights_per_epoch(g: RANModel, params: EpochParams, z: EventArray, /) -> Float[Array, "epochs m"]`
  - `TrainResult(g, d, history, seed, best_epoch, params, mmd_test, sigmas)` — the final shape. `mmd_test: float = float("nan")` and `sigmas: tuple[float, ...] = ()` are defaulted so `tests/test_cubic_sweep.py`'s stub stays constructible.
  - `history` gains `val_mmd` and `val_ess` (host additions; `_HISTORY_KEYS` stays three, since those two are not scan columns).
  - Task 4 imports `MMD_SUBSAMPLE` and `_weights_per_epoch` from `ran.train`. Importing a private name across modules matches the existing pattern in `ran/baselines/_shared.py`, which imports `_improvement` and `_wd_per_dim` from `ran.evaluate` — the leading underscore marks "not public API", not "not importable".

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_train.py`:

```python
class TestMMDSelection:
    @staticmethod
    def _splits(n: int = 768) -> DatasetSplits:
        rng = np.random.default_rng(41)
        z = rng.normal(size=(2 * n, 1)).astype(np.single)
        x = z + rng.normal(0, 0.4, size=(2 * n, 1)).astype(np.single)
        y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])
        return RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(z, x), y))

    def test_history_carries_the_mmd_and_ess_curves(self) -> None:
        result = train(self._splits(), dim=1, n_epochs=5, hidden_units=8, n_layers=1, seed=7)
        assert set(result.history) == {
            "train_d", "train_g", "val_d", "val_mmd", "val_ess",
        }
        assert all(len(v) == 5 for v in result.history.values())
        assert all(np.isfinite(result.history["val_mmd"]))
        assert all(e > 1.0 for e in result.history["val_ess"])

    def test_the_restored_epoch_is_the_mmd_argmin(self) -> None:
        result = train(self._splits(), dim=1, n_epochs=7, hidden_units=8, n_layers=1, seed=8)
        curve = np.asarray(result.history["val_mmd"], dtype=np.float64)
        assert result.best_epoch == int(np.argmin(curve))

    def test_selection_no_longer_tracks_the_bce(self) -> None:
        """Not a tautology: it pins that the criterion actually changed.

        If MMD selection silently fell back to the BCE, this passes only by
        coincidence -- so it asserts the two disagree on at least one of
        several seeds rather than on one.
        """
        picks = []
        for seed in (11, 12, 13, 14, 15):
            r = train(self._splits(), dim=1, n_epochs=9, hidden_units=8, n_layers=1, seed=seed)
            bce = int(np.argmin(np.abs(np.asarray(r.history["val_d"]) - LOG2)))
            picks.append((r.best_epoch, bce))
        assert any(m != b for m, b in picks), picks

    def test_the_reported_test_mmd_is_not_the_one_selection_optimized(self) -> None:
        """Selection runs on val; the quoted number comes from test."""
        result = train(self._splits(), dim=1, n_epochs=5, hidden_units=8, n_layers=1, seed=9)
        assert np.isfinite(result.mmd_test)
        assert result.mmd_test != min(result.history["val_mmd"])


def test_training_never_reads_the_truth_rows_of_z() -> None:
    """The MMD subsample must come from `y == 0` rows only.

    `z[y == 1]` is `z_true`. Poisoning it must leave every recorded number
    bit-identical -- the same guarantee `ran leakage-check` makes, asserted
    here at the seam where the MMD subsample is drawn.
    """
    n = 512
    rng = np.random.default_rng(42)
    z = rng.normal(size=(2 * n, 1)).astype(np.single)
    x = z + rng.normal(0, 0.4, size=(2 * n, 1)).astype(np.single)
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])

    poisoned = z.copy()
    poisoned[y == 1] = -999.0

    clean_r = train(
        RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(z, x), y)),
        dim=1, n_epochs=4, hidden_units=8, n_layers=1, seed=17,
    )
    dirty_r = train(
        RANDataset(batch_size=128, seed=0).splits_from_data(ZXY(Events(poisoned, x), y)),
        dim=1, n_epochs=4, hidden_units=8, n_layers=1, seed=17,
    )
    for key in ("val_d", "val_mmd", "val_ess"):
        np.testing.assert_array_equal(clean_r.history[key], dirty_r.history[key])
    assert clean_r.best_epoch == dirty_r.best_epoch
```

Delete `class TestSelectionCriterion` and the `SelectionCriterion` /
`_selection_score` imports.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_train.py -q -k "MMDSelection or truth_rows"`
Expected: FAIL — `KeyError: 'val_mmd'` and `AttributeError: ... 'mmd_test'`.

- [ ] **Step 3: Implement selection in `src/ran/train.py`**

```python
MMD_SUBSAMPLE: Final[int] = 16384


def _detector_arrays(
    zxy: ZXY, seed: int, m: int, /
) -> tuple[EventArray, EventArray, EventArray]:
    """Subsample the detector-level comparison and the generator's input.

    `z` is indexed **only** where `y == 0`. The nature rows of `z` hold
    `z_true`, and this module must never read them -- which is why selection
    is built from `ZXY` rather than from `partition()`, whose `Populations`
    would put truth in scope even if nothing used it.
    """
    nature: NDArray[np.bool] = zxy.y == 1
    mc: NDArray[np.bool] = ~nature
    x_data: EventArray = zxy.x[nature]
    x_sim: EventArray = zxy.x[mc]
    z_gen: EventArray = zxy.z[mc]
    i_d = subsample_indices(seed, x_data.shape[0], m)
    i_m = subsample_indices(seed + 1, x_sim.shape[0], m)
    return x_data[i_d], x_sim[i_m], z_gen[i_m]


def _weights_per_epoch(
    g: RANModel, params: EpochParams, z: EventArray, /
) -> Float[Array, "epochs m"]:
    """`g`'s raw output on a fixed sample, for every retained epoch.

    A plain Python loop rather than `vmap`: the non-trainable lists are empty
    for these architectures, which gives `vmap` no leaf to infer a batch size
    from, and 100 forward passes of a 34k-parameter MLP is milliseconds.
    """

    @jax.jit
    def one(trainable: Variables, non_trainable: Variables) -> Float[Array, " m"]:
        raw, _ = g.stateless_call(
            trainable_variables=trainable,
            non_trainable_variables=non_trainable,
            inputs=jnp.asarray(z),
            training=False,
        )
        return jnp.squeeze(raw, axis=-1)

    n_epochs: int = params.g_trainable[0].shape[0]
    return jnp.stack(
        [
            one(
                [leaf[i] for leaf in params.g_trainable],
                [leaf[i] for leaf in params.g_non_trainable],
            )
            for i in range(n_epochs)
        ]
    )
```

In `train()`, after `_run`:

```python
    # Selection: a fixed subsample of val, and the honest number from test.
    # Both are drawn from `data_seed`, so every hyperparameter arm compares
    # against an identical kernel and identical events.
    x_data, x_sim, z_gen = _detector_arrays(
        splits.val.as_arrays(), splits.train.seed, MMD_SUBSAMPLE
    )
    sigmas: tuple[float, ...] = bandwidths(jnp.asarray(x_data))
    cache: MMDCache = build_cache(
        jnp.asarray(x_data), jnp.asarray(x_sim), sigmas=sigmas
    )
    raw_w: Float[Array, "epochs m"] = _weights_per_epoch(g, params, z_gen)
    mmd, ess = mmd_curve(cache, raw_w)
    history["val_mmd"] = mmd.tolist()
    history["val_ess"] = ess.tolist()

    best_epoch: int = int(np.argmin(mmd))
    logger.info(
        "Restoring epoch %d of %d  (val MMD^2 %.3e, ESS %.0f)",
        best_epoch + 1, n_epochs, mmd[best_epoch], ess[best_epoch],
    )
    _restore(g, d, params, best_epoch)

    # Recomputed on test, because selecting 100 times against one val
    # subsample is exactly the regime where the estimator starts fitting the
    # sample rather than the distribution.
    tx_data, tx_sim, tz_gen = _detector_arrays(
        splits.test.as_arrays(), splits.train.seed + 2, MMD_SUBSAMPLE
    )
    test_cache: MMDCache = build_cache(
        jnp.asarray(tx_data), jnp.asarray(tx_sim), sigmas=sigmas
    )
    mmd_test = float(
        weighted_mmd(test_cache, _weights_per_epoch(g, params, tz_gen)[best_epoch])[0]
    )
    logger.info("Test MMD^2: %.3e  (init seed %d)", mmd_test, seed)

    return TrainResult(g, d, history, seed, best_epoch, params, mmd_test, sigmas)
```

Add `mmd_test: float = float("nan")` and `sigmas: tuple[float, ...] = ()` to
`TrainResult`, and `_HISTORY_KEYS` stays three — `val_mmd`/`val_ess` are host
additions, not scan columns.

- [ ] **Step 4: Delete the BCE criteria**

- `src/ran/rantypes/enums.py` — delete `class SelectionCriterion` entirely.
- `src/ran/rantypes/__init__.py` — delete the `SelectionCriterion` export line.
- `src/ran/train.py` — delete `_selection_score` and the `SelectionCriterion` import. Keep `LOG2`: `tests/test_train.py` still uses it to show the criteria disagree.
- `src/ran/cli.py` — delete the `--criterion` and `--min-delta` options and their forwarding.
- `src/ran/workflow.py` — delete `criterion` and `min_delta` from `run()`'s signature and from the `hyperparameters` dict.
- `tests/test_cli.py` — delete the `criterion` assertion.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest -q`
Expected: PASS.

If `test_selection_no_longer_tracks_the_bce` fails, that is a real signal, not
a flaky test: it means MMD and the BCE agreed on all five seeds. Print both
curves before assuming the implementation is wrong.

- [ ] **Step 6: Update the prose**

In `CLAUDE.md`, replace the whole "What counts as `best` is `--criterion`..."
passage (added in the previous session) with:

```markdown
Selection does not happen in the loop. A GAN's loss is not a proxy for a
single scalar objective monotonically related to model quality: it oscillates
around its equilibrium by construction, a flat curve cannot be told from a
stalled one, and `log 2 - BCE` estimates a divergence only when `d` is
optimal -- which nothing reports. Both criteria once built on it were unsound
and are gone.

Instead the scan emits every epoch's parameters (`EpochParams`, ~27 MB for
100 epochs of both networks), and `train` picks the epoch minimizing a
weighted MMD against a fixed subsample of the validation split. MMD is a
divergence -- zero iff the distributions match, monotone in mismatch, no
adversary and no optimization -- so patience and early stopping would be
sound again. They are gone anyway: `scan` has a fixed trip count, and at
0.034s/epoch against a 4.6s compile, early stopping saved less wall clock
than compiling the loop that implemented it.

Selection is **detector level** (`x_sim` reweighted vs `x_data`), so it needs
no truth and the method stays deployable. The particle-level MMD is computed
too, but on the host in `workflow`, never in the trace -- which is what keeps
`z_true` out of the traced program while still producing the curve. The
number reported for the restored checkpoint comes from a *test* subsample,
not the val one selection minimized.

The estimator has a resolution floor around 5e-4 in MMD^2 at m=8192, scaling
as ~1/m; below it the ranking inverts, because the empirical MMD is minimized
by weights matching the sample rather than the distribution. `benchmarks/`
measures it. `MMD_SUBSAMPLE` is 16384.
```

Also delete the `--patience` and `--criterion` sentences from the Running
section, and update `src/ran/README.md`'s `RunCarry` bullets to the two
surviving fields.

- [ ] **Step 7: Full validation, then commit**

```bash
uv format && uv run --locked ruff check && uv run --locked pyrefly check --min-severity info && uv run --locked complexipy && uv run --locked pytest -q
git add -A
git commit -m "feat: select checkpoints on detector-level weighted MMD

Deletes SelectionCriterion. Both BCE criteria were unsound: max_val rewards
g overshooting a stale d, and dist_log2 is maximized by a collapsed d that
scores log 2 with g doing nothing.

Selection reads y == 0 rows of z only, so the truth rows stay unreachable
from train.py; a poisoning test asserts it. The reported number is
recomputed on test, because selecting N times against one val subsample is
where the estimator starts fitting the sample.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: The particle-level diagnostic, and the run record

**Files:**
- Modify: `src/ran/workflow.py` (`run`, `_save_run`)
- Test: `tests/test_workflow.py` (create if absent — check first with `ls tests/`)

**Interfaces:**
- Consumes: `TrainResult.params`, `TrainResult.sigmas`, `TrainResult.mmd_test` (Task 3); `MMD_SUBSAMPLE` and `_weights_per_epoch` from `ran.train` (Task 3); `bandwidths`, `build_cache`, `mmd_curve`, `subsample_indices` from `ran.mmd` (Task 1).
- Produces: `_particle_curve(splits, result, data_seed) -> tuple[list[float], tuple[float, ...]] | None`; `history["val_mmd_particle"]`; `config.json` keys `mmd_subsample`, `mmd_sigmas_detector`, `mmd_sigmas_particle`, `mmd_test`, `best_epoch`.

- [ ] **Step 1: Write the failing test**

`run()` takes `config: Path | None` for Gaussian mode, so the test writes a
YAML and passes its path. Do **not** add a parameter to `run()` for the
test's benefit.

```python
import json
from pathlib import Path

import numpy as np
from ran.rantypes import DatasetName
from ran.workflow import run

_CONFIG = """\
mu_gen: [0.5]
mu_true: [0.0]
sigma_gen: 0.9
sigma_true: 1.0
sigma_detector: 0.5
"""


def test_particle_curve_is_recorded_when_truth_exists(tmp_path, monkeypatch) -> None:
    """The diagnostic curve lives here, not in `train`, because it needs
    `z_true` -- and putting truth-derived Grams in the trace is exactly what
    `Populations` keeping `truth` outside `Events` exists to prevent.
    """
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "gaussian.yaml"
    config_path.write_text(_CONFIG)

    run(
        128, 2000, config_path, DatasetName.gaussian, (), None, 8, 1, 3, 42,
        n_epochs=4, plots=False,
    )

    run_dir = next((tmp_path / "runs").iterdir())
    history = dict(np.load(run_dir / "history.npz"))
    assert "val_mmd_particle" in history
    assert len(history["val_mmd_particle"]) == 4
    assert np.all(np.isfinite(history["val_mmd_particle"]))

    config = json.loads((run_dir / "config.json").read_text())
    assert config["best_epoch"] >= 0
    assert config["mmd_subsample"] == 16384
    assert len(config["mmd_sigmas_detector"]) == 5
    assert len(config["mmd_sigmas_particle"]) == 5
    # The unsound knobs are gone from the record, not merely unused.
    assert "criterion" not in config
    assert "patience" not in config
    assert "min_delta" not in config
```

Check `ls tests/` first: if `tests/test_workflow.py` does not exist, create it
with the imports above; if it does, append the test and reuse its imports.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_workflow.py -q`
Expected: FAIL — `KeyError: 'val_mmd_particle'`.

- [ ] **Step 3: Implement in `src/ran/workflow.py`**

```python
def _particle_curve(
    splits: DatasetSplits,
    result: TrainResult,
    data_seed: int,
) -> tuple[list[float], tuple[float, ...]] | None:
    """Particle-level MMD per epoch: the diagnostic, never the criterion.

    Returns `None` for a real measurement, which has no truth to score
    against. Selection has already happened by the time this runs, so nothing
    the generator saw depends on it.
    """
    pops: Populations = splits.val.as_arrays().partition()
    if not pops.has_truth:
        return None
    z_true: EventArray = pops.require_truth()
    z_gen: EventArray = pops.mc.z
    i_t = subsample_indices(data_seed + 3, z_true.shape[0], MMD_SUBSAMPLE)
    i_g = subsample_indices(data_seed + 4, z_gen.shape[0], MMD_SUBSAMPLE)
    ref, comp = jnp.asarray(z_true[i_t]), jnp.asarray(z_gen[i_g])
    sigmas = bandwidths(ref)
    curve, _ = mmd_curve(
        build_cache(ref, comp, sigmas=sigmas),
        _weights_per_epoch(result.g, result.params, z_gen[i_g]),
    )
    return curve.tolist(), sigmas
```

Then, in `run()`, replace the `train(...)` / `_save_run(...)` block's tail.
`run()` is already at the complexity ceiling, so this goes in a helper rather
than inline — extract it rather than reaching for a `# noqa`:

```python
def _finish_run(
    splits: DatasetSplits,
    result: TrainResult,
    data_seed: int,
    /,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Merge the particle diagnostic in, and assemble what gets recorded."""
    history: dict[str, list[float]] = dict(result.history)
    particle = _particle_curve(splits, result, data_seed)
    sigmas_particle: tuple[float, ...] = ()
    if particle is not None:
        history["val_mmd_particle"], sigmas_particle = particle
    return history, {
        "hidden_units": ...,  # filled by the caller, which owns these
        "mmd_subsample": MMD_SUBSAMPLE,
        "mmd_sigmas_detector": list(result.sigmas),
        "mmd_sigmas_particle": list(sigmas_particle),
        "mmd_test": result.mmd_test,
        "best_epoch": result.best_epoch,
    }
```

The `...` above is a signal, not a placeholder to leave in: the architecture
and optimization knobs are `run()`'s own arguments, so build that half of the
dict at the call site and merge, rather than threading eight parameters into
this helper:

```python
        result = train(splits, dim, hidden_units, n_layers, seed, ...)
        history, mmd_record = _finish_run(splits, result, data_seed)
        run_dir = _save_run(
            result.g, result.d, history,
            batch_size=batch_size, n_samples=n_samples, dim=dim,
            dataset=dataset, init_seed=result.seed, data_seed=data_seed,
            gaussian_params=gaussian_params, variables=variables,
            hyperparameters={
                "hidden_units": hidden_units,
                "n_layers": n_layers,
                "n_epochs": n_epochs,
                "n_disc_steps": n_disc_steps,
                "lr_g": lr_g,
                "lr_d": lr_d,
                **mmd_record,
            },
        )
```

Delete the `"hidden_units": ...` line from `_finish_run`'s returned dict once
the call site above owns it.

- [ ] **Step 4: Run tests, lint, commit**

```bash
uv format && uv run --locked ruff check && uv run --locked pyrefly check --min-severity info && uv run --locked complexipy && uv run --locked pytest -q
git add -A
git commit -m "feat: record the particle-level MMD curve as a diagnostic

Computed in workflow, not train: it needs z_true, and the whole reason
selection moved to the host is so truth-derived Grams never enter the
traced program.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: The selection figure

**Files:**
- Modify: `src/ran/plotting.py` (add `plot_selection`)
- Modify: `src/ran/workflow.py` — `_draw_figures` gains a `best_epoch: int` parameter, and `run()` passes `result.best_epoch` (or the value read back from `config.json` on the `--load-run` path, where no `TrainResult` exists)
- Test: `tests/test_plotting.py` (add `plot_selection` to its existing `from ran.plotting import ...`)

**Interfaces:**
- Consumes: `history` with `val_mmd` and `val_ess`, optionally `val_mmd_particle`.
- Produces: `plot_selection(history: dict[str, list[float]], best_epoch: int, save_path: Path = Path("plots/selection.pdf")) -> None`.

- [ ] **Step 1: Write the failing test**

```python
def test_selection_plot_survives_a_missing_particle_curve(tmp_path) -> None:
    """A real measurement has no truth, so the particle curve is optional."""
    history = {
        "train_d": [0.69] * 5, "train_g": [0.69] * 5, "val_d": [0.69] * 5,
        "val_mmd": [0.05, 0.03, 0.01, 0.02, 0.04],
        "val_ess": [900.0, 850.0, 800.0, 700.0, 600.0],
    }
    out = tmp_path / "selection.pdf"
    plot_selection(history, best_epoch=2, save_path=out)
    assert out.exists() and out.stat().st_size > 0

    history["val_mmd_particle"] = [0.09, 0.07, 0.06, 0.06, 0.07]
    plot_selection(history, best_epoch=2, save_path=out)
    assert out.exists()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_plotting.py -q -k selection`
Expected: FAIL — `ImportError: cannot import name 'plot_selection'`.

- [ ] **Step 3: Implement `plot_selection`**

Add to `src/ran/plotting.py`, following `plot_losses`'s structure — `Figure`
plus `FigureCanvasPdf`, never pyplot:

```python
def plot_selection(
    history: dict[str, list[float]],
    best_epoch: int,
    save_path: Path = Path("plots/selection.pdf"),
) -> None:
    """The two MMD curves and the epoch selection landed on.

    Detector-level MMD is the criterion; particle-level is the diagnostic.
    Where they diverge -- detector still falling while particle turns up -- is
    the ill-posedness made visible, and it is the plot that answers whether
    truth-free selection costs anything. The particle curve is absent for a
    real measurement, which has no truth to score against, so it is optional.

    ESS shares the figure because the adversarial objective is linear in the
    weights and therefore maximized at a simplex vertex: a falling MMD bought
    by a collapsing effective sample size is not an improvement.
    """
    epochs: NDArray[np.uintc] = np.arange(len(history["val_mmd"]), dtype=np.uintc)

    figure: Figure = Figure(figsize=(8, 5))
    figure.canvas = FigureCanvasPdf(figure)
    ax: Axes = figure.add_subplot(111)

    ax.plot(
        epochs,
        np.array(history["val_mmd"], dtype=np.double),
        label="Detector MMD$^2$ (criterion)",
        color="C0",
        lw=2,
    )
    if "val_mmd_particle" in history:
        ax.plot(
            epochs,
            np.array(history["val_mmd_particle"], dtype=np.double),
            label="Particle MMD$^2$ (diagnostic)",
            color="C3",
            ls="--",
            lw=2,
        )
    ax.axvline(
        best_epoch, color="k", ls=":", lw=1, label=f"selected (epoch {best_epoch + 1})"
    )
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"MMD$^2$")

    ess: Axes = ax.twinx()
    ess.plot(
        epochs,
        np.array(history["val_ess"], dtype=np.double),
        color="C7",
        lw=1,
        alpha=0.6,
    )
    ess.set_ylabel("Effective sample size", color="C7")
    ess.tick_params(axis="y", labelcolor="C7")

    ax.legend(loc="best")
    figure.tight_layout()
    figure.savefig(save_path)
    logger.info("Saved %s", save_path)
```

Then wire it into `_draw_figures` in `src/ran/workflow.py`, which needs
`best_epoch` passed through:

```python
    plot_selection(history, best_epoch, save_path=run_dir / "selection.pdf")
```

- [ ] **Step 4: Run tests, lint, commit**

```bash
uv format && uv run --locked ruff check && uv run --locked pyrefly check --min-severity info && uv run --locked complexipy && uv run --locked pytest -q
git add -A
git commit -m "feat: plot the selection curves

Detector and particle MMD against epoch with the selected epoch marked.
Divergence between them is the ill-posedness made visible.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## After the plan: validation on real data

These are measurements, not code, and they run on the cluster where the jet
cache lives. They are the point of the whole exercise — do not skip them.

1. **Measure the resolution floor.** One weight vector against ~10 independent
   val subsamples at m ∈ {4096, 8192, 16384}. Confirms the ~1/m scaling and
   fixes the floor in absolute units.
2. **Check the dynamic range.** Detector MMD across epochs against that floor.
   *This is the risk the detector-level decision was taken with open eyes:* the
   flagship run already reaches 89–98% detector improvement, so the curve may
   be flat relative to the floor, in which case selection is choosing noise. If
   so: raise `MMD_SUBSAMPLE`, move to tiled-exact over the full 50k val rows, or
   reconsider the level.
3. **Compare the three criteria** against one retained history — MMD, `max_val`,
   `dist_log2` — before the deleted code is out of reach in git history. Records
   what the change bought.
4. Then, and only then, the 10-seed baseline and the hyperparameter sweep.
