# Vectorized Histogram Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return normalized histograms as feature-by-bin matrices and compute Jensen-Shannon and triangular divergences across all features with axis-wise operations, including plain one-dimensional inputs.

**Architecture:** `_normalized_histograms` remains responsible for feature-specific bin construction with `numpy.histogram`, but materializes its results into two `(dimensions, n_bins)` matrices. `_js_per_dim` and `_triangular_per_dim` consume those matrices directly and reduce over the bin axis.

**Tech Stack:** Python 3.13, NumPy 2.5, SciPy 1.18, pytest 9, Ruff, Pyrefly

## Global Constraints

- Preserve the existing metric definitions, `comp`-only weighting semantics, per-feature shared bin edges, and zero-histogram normalization behavior.
- Treat a plain one-dimensional sample array as one feature and return metric arrays with shape `(1,)`.
- Continue using `numpy.histogram` per feature; do not add custom flattened-bin arithmetic.
- Do not add dependencies or broaden input validation.
- Preserve all pre-existing uncommitted edits in `src/ran/evaluate.py` and `src/README.md`.
- Restore the runtime-compatible CLI and Gaussian-config signatures that regressed in commit `22ced91` before changing histogram metrics.
- Work in the clean isolated task worktree and commit each reviewed task independently.

---

### Task 1: Restore runtime-compatible CLI and config signatures

**Files:**
- Modify: `src/ran/cli.py:3-19,104`
- Modify: `src/ran/data/config.py:64-66`
- Modify: `tests/test_cli.py:68-111`
- Test: `tests/test_config.py`

**Interfaces:**
- Preserves: Typer command construction and every existing CLI option.
- Restores: `gaussian_config_from_run_config(params, dim=...)` keyword compatibility.
- Preserves: the committed `workflow.run` contract, which now receives `DatasetName` and may be called positionally.

- [ ] **Step 1: Verify the existing regression tests fail**

Run:

```bash
uv run --locked pytest -q tests/test_cli.py tests/test_config.py
```

Expected: CLI tests fail because Typer cannot resolve `Annotated` at runtime, and six config tests fail because `dim` became positional-only.

- [ ] **Step 2: Restore runtime annotation dependencies for Typer**

Move `Path` and `Annotated` out of the `TYPE_CHECKING` block in `src/ran/cli.py`:

```python
from pathlib import Path
from typing import Annotated
```

Remove the now-empty `TYPE_CHECKING` import and block. Typer calls `inspect.signature(..., eval_str=True)`, so annotations it consumes must be available in module globals even with postponed annotations enabled.

- [ ] **Step 3: Restore a Typer-supported purity-threshold annotation**

Change only the CLI-facing annotation:

```python
purity_threshold: float = DEFAULT_PURITY_THRESHOLD,
```

Retain the existing `purity_threshold = np.double(purity_threshold)` conversion inside the command before calling the IBU evaluator.

- [ ] **Step 4: Restore the Gaussian config loader's keyword-compatible signature**

Change the signature to:

```python
def gaussian_config_from_run_config(
    params: Mapping[str, Any], dim: int
) -> GaussianConfig:
```

Do not change its implementation or its existing callers.

- [ ] **Step 5: Align the CLI forwarding test with the committed workflow contract**

The latest workflow contract accepts `DatasetName` rather than `str`, and the CLI now calls it positionally. Replace the `lambda **kwargs` fake with a signature-matching fake that records named values, then assert the enum:

```python
def fake_run(
    batch_size,
    n_samples,
    config,
    dataset,
    variables,
    load_run,
    hidden_units,
    n_layers,
    patience,
    seed,
    data_seed,
) -> None:
    calls.append(
        {
            "batch_size": batch_size,
            "n_samples": n_samples,
            "config": config,
            "dataset": dataset,
            "variables": variables,
            "load_run": load_run,
            "hidden_units": hidden_units,
            "n_layers": n_layers,
            "patience": patience,
            "seed": seed,
            "data_seed": data_seed,
        }
    )


fake_workflow.__dict__["run"] = fake_run
```

Change `assert calls[0]["dataset"] == "jets"` to:

```python
assert calls[0]["dataset"] is cli.DatasetName.jets
```

This test adjustment preserves the intentional workflow type refactor while continuing to verify Typer conversion, repeated-variable collection, `Path` preservation, and seed forwarding.

- [ ] **Step 6: Verify focused tests pass**

Run:

```bash
uv run --locked pytest -q tests/test_cli.py tests/test_config.py
```

Expected: all tests in both files pass with pristine output.

- [ ] **Step 7: Run focused static checks and commit**

Run:

```bash
uv run --locked ruff check src/ran/cli.py src/ran/data/config.py tests/test_cli.py tests/test_config.py
uv format --check src/ran/cli.py src/ran/data/config.py tests/test_cli.py tests/test_config.py
uv run --locked pyrefly check --min-severity info src/ran/cli.py src/ran/data/config.py
git diff --check
```

Expected: every command exits zero. Commit the two production files and the aligned CLI test:

```bash
git add src/ran/cli.py src/ran/data/config.py tests/test_cli.py
git commit -m "fix: restore runtime-compatible command signatures"
```

---

### Task 2: Matrix histogram contract and axis-wise divergences

**Files:**
- Create: `tests/test_evaluate_metrics.py`
- Modify: `src/ran/evaluate.py:139-193`
- Modify: `src/README.md:58-86`

**Interfaces:**
- Consumes: `ref: NDArray[T]`, `comp: NDArray[T]`, optional comparison-sample `weights: NDArray[T]`, and `n_bins: int`.
- Produces: `_normalized_histograms(...) -> tuple[NDArray[np.double], NDArray[np.double]]`, with both arrays shaped `(dimensions, n_bins)`.
- Produces: `_js_per_dim(...) -> NDArray[np.double]` and `_triangular_per_dim(...) -> NDArray[np.double]`, each shaped `(dimensions,)`.

- [ ] **Step 1: Add focused tests for the matrix contract, one-dimensional support, weighting, and both metric formulas**

Create `tests/test_evaluate_metrics.py` with expectations derived from literal two-bin distributions:

```python
import numpy as np

from ran.evaluate import (
    _js_per_dim,
    _normalized_histograms,
    _triangular_per_dim,
)


def test_normalized_histograms_treats_1d_samples_as_one_feature() -> None:
    p, q = _normalized_histograms(
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 1.0]),
        n_bins=2,
    )

    assert p.shape == (1, 2)
    assert q.shape == (1, 2)
    np.testing.assert_allclose(p, [[0.5, 0.5]])
    np.testing.assert_allclose(q, [[0.25, 0.75]])


def test_normalized_histograms_weights_only_comparison_samples() -> None:
    ref = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    comp = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    p, q = _normalized_histograms(
        ref,
        comp,
        weights=np.array([1.0, 1.0, 2.0]),
        n_bins=2,
    )

    np.testing.assert_allclose(p, [[0.5, 0.5], [0.5, 0.5]])
    np.testing.assert_allclose(q, [[0.25, 0.75], [0.5, 0.5]])


def test_js_per_dim_reduces_over_histogram_bins() -> None:
    ref = np.array([[0.0, 0.0], [0.0, 1.0]])
    comp = np.array([[1.0, 0.0], [1.0, 1.0]])

    result = _js_per_dim(ref, comp, n_bins=2)

    np.testing.assert_allclose(result, [np.log(2.0), 0.0], atol=1e-15)


def test_triangular_per_dim_reduces_over_histogram_bins() -> None:
    ref = np.array([[0.0, 0.0], [0.0, 1.0]])
    comp = np.array([[1.0, 0.0], [1.0, 1.0]])

    result = _triangular_per_dim(ref, comp, n_bins=2)

    np.testing.assert_allclose(result, [2000.0, 0.0], atol=1e-12)
```

These tests catch: accidental two-dimensional indexing of 1-D inputs, returning an iterator instead of matrices, applying weights to `ref`, reducing over features instead of bins, forgetting to square Jensen-Shannon distance, and dividing by zero in empty shared bins.

- [ ] **Step 2: Run the focused tests and verify the red state**

Run:

```bash
uv run --locked pytest -q tests/test_evaluate_metrics.py
```

Expected: FAIL. The 1-D test fails at `ref[:, i]`, while the matrix-contract test cannot use the current iterator as two feature-by-bin arrays. Record that these are contract failures rather than import or fixture errors.

- [ ] **Step 3: Materialize normalized histograms into two matrices**

Replace `_normalized_histograms` with:

```python
def _normalized_histograms[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
    n_bins: int = 100,
) -> tuple[NDArray[np.double], NDArray[np.double]]:
    ref_2d: NDArray[T] = ref.reshape(-1, 1) if ref.ndim == 1 else ref
    comp_2d: NDArray[T] = comp.reshape(-1, 1) if comp.ndim == 1 else comp
    dim: int = _dim(ref_2d)
    p: NDArray[np.double] = np.empty(shape=(dim, n_bins), dtype=np.double)
    q: NDArray[np.double] = np.empty(shape=(dim, n_bins), dtype=np.double)

    for i in range(dim):
        r: NDArray[T] = ref_2d[:, i]
        c: NDArray[T] = comp_2d[:, i]
        bins: NDArray[np.double] = np.linspace(
            start=min(r.min(), c.min()),
            stop=max(r.max(), c.max()),
            num=n_bins + 1,
        )
        h_ref: NDArray[np.intp] = np.histogram(a=r, bins=bins)[0]
        h_comp: NDArray[np.intp | T] = np.histogram(
            a=c,
            bins=bins,
            weights=weights,
        )[0]
        p[i] = h_ref / (h_ref.sum() or 1.0)
        q[i] = h_comp / (h_comp.sum() or 1.0)

    return p, q
```

- [ ] **Step 4: Convert Jensen-Shannon and triangular divergence to axis-wise operations**

Replace the two consumers with:

```python
def _js_per_dim[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
    n_bins: int = 100,
) -> NDArray[np.double]:
    p, q = _normalized_histograms(ref, comp, weights, n_bins)
    return np.square(jensenshannon(p, q, axis=1))


def _triangular_per_dim[T: np.floating = np.double](
    ref: NDArray[T],
    comp: NDArray[T],
    weights: NDArray[T] | None = None,
    n_bins: int = 100,
) -> NDArray[np.double]:
    """Triangular discriminator (Vincze-LeCam divergence) per dimension."""
    p, q = _normalized_histograms(ref, comp, weights, n_bins)
    denominator: NDArray[np.double] = p + q
    difference: NDArray[np.double] = p - q
    terms: NDArray[np.double] = np.divide(
        difference**2,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return np.sum(terms, axis=1) * 1e3
```

Retain the existing mathematical explanation in `_triangular_per_dim` below the summary line; only the implementation and annotations change.

- [ ] **Step 5: Run the focused tests and verify the green state**

Run:

```bash
uv run --locked pytest -q tests/test_evaluate_metrics.py
```

Expected: `4 passed`, with no warnings.

- [ ] **Step 6: Update the private helper documentation**

In `src/README.md`, change the `_normalized_histograms` signature and prose to say it returns a tuple of two `(dimensions, n_bins)` arrays rather than yielding per-feature pairs. Change `_js_per_dim`'s documented return annotation from `list[float]` to `NDArray[np.double]`. Keep this edit limited to those existing sections; do not add a new `_triangular_per_dim` documentation section.

- [ ] **Step 7: Run focused static checks**

Run:

```bash
uv run --locked ruff check src/ran/evaluate.py tests/test_evaluate_metrics.py
uv format --check src/ran/evaluate.py tests/test_evaluate_metrics.py
uv run --locked pyrefly check --min-severity info src/ran/evaluate.py tests/test_evaluate_metrics.py
```

Expected: all commands exit zero. If formatting is required, run `uv format` on only the two Python files, inspect the diff, and rerun these commands.

- [ ] **Step 8: Run the complete validation suite**

Run:

```bash
just check
```

Expected: formatting, Ruff, Pyrefly, dependency validation, complexity checks, and the complete pytest suite all pass without new warnings. If a failure is unrelated to these changes, capture the exact pre-existing failure and still rerun the focused tests to demonstrate this refactor remains green.

- [ ] **Step 9: Review and commit the final patch**

Run:

```bash
git diff --check
git diff -- src/ran/evaluate.py src/README.md tests/test_evaluate_metrics.py
git status --short
```

Confirm that the new changes are limited to the approved helper contract, the two consumers, their tests, and their documentation. Then commit the task:

```bash
git add src/ran/evaluate.py src/README.md tests/test_evaluate_metrics.py
git commit -m "refactor: vectorize histogram divergences"
```
