# Cubic Response Sweep (RAN vs OmniFold) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a SLURM-array sweep that, for each value of a cubic detector-distortion parameter `s ∈ linspace(0, 20, 25)`, trains both the real RAN and the real OmniFold to unfold `z_gen` back to `z_truth`, and plots Wasserstein(`z_truth`, `z_unfolded`) vs `s` for both methods.

**Architecture:** A new `ran/experiments/cubic_sweep.py` module reuses the existing `train()` (RAN) and a newly-extracted `omnifold_unfold()` (OmniFold). Each SLURM array task runs one `s` via `run_point` and writes one JSON; a dependent `collect` job concatenates the JSONs into `results.npz` + a PDF. Two small, behavior-preserving edits to existing code make the training entry points callable on in-memory arrays.

**Tech Stack:** Python 3.13 (uv), TensorFlow/Keras, the `omnifold` package, scipy (`wasserstein_distance`), matplotlib, python-fire, SLURM.

---

## Implementation deviations (as built)

- **No local model training in tests.** The default `uv run pytest` must never train RAN or OmniFold — that is cluster work. Task 4's smoke test was replaced with a fast **stubbed** wiring test (`monkeypatch` fakes `train` and `omnifold_unfold`). Task 2's real-OmniFold smoke test is gated behind `RAN_RUN_SLOW=1` (skipped by default). Task 7's "real mini sweep" was replaced by running the stubbed suite plus a real `collect` CLI exercise over stub JSONs (no training).
- **`run_point` gained an OmniFold hang-guard.** OmniFold computes `validation_steps = (0.2 * NTRAIN) // batch_size`; at small N this floors to 0 and `model.fit` hangs forever. `run_point` caps the OmniFold batch size (`of_batch = max(1, min(omnifold_batch_size, 2*n_samples//5))`) so there is always ≥1 validation step. No-op at the real n=500k scale. Added param: `omnifold_batch_size=512`.

---

## File Structure

- **Create** `ran/experiments/__init__.py` — empty package marker.
- **Create** `ran/experiments/cubic_sweep.py` — `response`, `make_particles`, `unfolded_wasserstein`, `run_point`, `collect`, fire CLI.
- **Modify** `ran/data/datasets.py` — add public `RAN_Dataset.splits_from_arrays`.
- **Modify** `ran/baselines/omnifold.py` — extract reusable `omnifold_unfold`, call it from `_run_and_evaluate`.
- **Create** `scripts/submit_sweep.sh` — login-node launcher submitting the array + dependent collect job.
- **Create** `tests/test_cubic_sweep.py` — unit + smoke tests for the new module.
- **Modify** `tests/test_datasets.py` — add a `splits_from_arrays` test.
- **Create** `tests/test_omnifold.py` — smoke test for `omnifold_unfold`.

---

## Task 1: `splits_from_arrays` on RAN_Dataset

**Files:**

- Modify: `ran/data/datasets.py` (add method after `generate_gaussian_dataset`)
- Test: `tests/test_datasets.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_datasets.py`:

```python
import numpy as np
from ran.data.datasets import RAN_Dataset


def test_splits_from_arrays_builds_three_nonempty_splits():
    n = 200
    z = np.random.default_rng(0).normal(size=(2 * n, 1))
    x = np.random.default_rng(1).normal(size=(2 * n, 1))
    y = np.concatenate([np.ones(n, dtype=np.ubyte), np.zeros(n, dtype=np.ubyte)])

    splits = RAN_Dataset(batch_size=32).splits_from_arrays(z, x, y)

    for ds in (splits.train, splits.val, splits.test):
        features, labels = next(iter(ds))
        assert set(features.keys()) == {"z", "x"}
        assert features["z"].shape[-1] == 1
        assert labels.shape[0] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_datasets.py::test_splits_from_arrays_builds_three_nonempty_splits -v`
Expected: FAIL with `AttributeError: 'RAN_Dataset' object has no attribute 'splits_from_arrays'`

- [ ] **Step 3: Write minimal implementation**

In `ran/data/datasets.py`, add this method to `RAN_Dataset` immediately after `generate_gaussian_dataset` (the body mirrors that method's final two lines):

```python
    def splits_from_arrays(
        self,
        z: npt.NDArray[np.double],
        x: npt.NDArray[np.double],
        y: npt.NDArray[np.ubyte],
    ) -> DatasetSplits:
        """Build train/val/test splits directly from in-memory (z, x, y) arrays.

        z (particle level) and x (detector level) must have matching first
        dimension; y is the per-event class label (1 = data, 0 = MC).
        """
        self.dataset = self._build_dataset(z, x, y)
        self.splits = self._split_dataset(self.dataset)
        return self.splits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_datasets.py::test_splits_from_arrays_builds_three_nonempty_splits -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ran/data/datasets.py tests/test_datasets.py
git commit -m "feat: add RAN_Dataset.splits_from_arrays for in-memory datasets"
```

---

## Task 2: Extract `omnifold_unfold` from the baseline

**Files:**

- Modify: `ran/baselines/omnifold.py` (add function; refactor `_run_and_evaluate` to call it)
- Test: `tests/test_omnifold.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_omnifold.py`:

```python
import numpy as np
import pytest


def test_omnifold_unfold_returns_mean_normalized_weights():
    from ran.baselines.omnifold import omnifold_unfold

    rng = np.random.default_rng(0)
    n = 500
    z_gen = rng.normal(-1.0, 1.0, size=(n, 1)).astype(np.float32)
    x_sim = z_gen + 0.1 * rng.normal(size=(n, 1)).astype(np.float32)
    x_data = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

    w = omnifold_unfold(x_data, x_sim, z_gen, niter=1, epochs=2, batch_size=128)

    assert w.shape == (n,)
    assert np.all(np.isfinite(w))
    np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_omnifold.py -v`
Expected: FAIL with `ImportError: cannot import name 'omnifold_unfold'`

- [ ] **Step 3: Add the `omnifold_unfold` function**

In `ran/baselines/omnifold.py`, add this function above `_run_and_evaluate` (it lifts the `DataLoader`/`MultiFold`/`Unfold`/`reweight` logic currently inlined there):

```python
def omnifold_unfold(
    x_data: npt.NDArray,
    x_sim: npt.NDArray,
    z_gen: npt.NDArray,
    z_target: npt.NDArray | None = None,
    niter: int = 3,
    epochs: int = 50,
    batch_size: int = 512,
) -> npt.NDArray[np.float64]:
    """Train OmniFold on in-memory arrays; return mean-normalized gen weights.

    Trains on (data reco = x_data, MC reco = x_sim, MC gen = z_gen), then
    reweights z_target (defaults to z_gen) through the gen-level model. Returns
    a 1D weight array, normalized so its mean is 1.
    """
    def _as2d(a: npt.NDArray) -> npt.NDArray:
        a = np.asarray(a, dtype=np.float32)
        return a[:, None] if a.ndim == 1 else a

    x_data = _as2d(x_data)
    x_sim = _as2d(x_sim)
    z_gen = _as2d(z_gen)
    z_target = z_gen if z_target is None else _as2d(z_target)
    dim = x_data.shape[1]

    data_dl = DataLoader(reco=x_data)
    mc_dl = DataLoader(reco=x_sim, gen=z_gen)

    unfold = MultiFold(
        "omnifold_baseline",
        MLP(dim), MLP(dim),
        data_dl, mc_dl,
        niter=niter,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
    )
    unfold.Unfold()

    w = unfold.reweight(z_target, unfold.model2).astype(np.float64).ravel()
    return w / w.mean()
```

- [ ] **Step 4: Refactor `_run_and_evaluate` to call it**

In `ran/baselines/omnifold.py`, inside `_run_and_evaluate`, replace the block that builds `data_dl`, `mc_dl`, the `MultiFold`, calls `unfold.Unfold()`, and computes `w` via `unfold.reweight(...)`. The current code is:

```python
    dim = x_data.shape[1]

    data_dl = DataLoader(reco=x_data)
    mc_dl = DataLoader(reco=x_mc, gen=z_mc)

    unfold = MultiFold(
        "omnifold_baseline",
        MLP(dim), MLP(dim),
        data_dl, mc_dl,
        niter=niter,
        epochs=epochs,
        batch_size=512,
        verbose=False,
    )
    unfold.Unfold()

    # Evaluate on test split only
    z_test, x_test, y_test = _collect_test_data(splits.test)
    z_data_t = z_test[y_test == 1]
    x_data_t = x_test[y_test == 1]
    z_mc_t = z_test[y_test == 0]
    x_mc_t = x_test[y_test == 0]

    # Get OmniFold weights for test MC via the trained gen-level model
    w = unfold.reweight(z_mc_t.astype(np.float32), unfold.model2).astype(np.float64)
    w = w / w.mean()
```

Replace it with (train on full arrays, reweight the test subset via `z_target`):

```python
    # Evaluate on test split only
    z_test, x_test, y_test = _collect_test_data(splits.test)
    z_data_t = z_test[y_test == 1]
    x_data_t = x_test[y_test == 1]
    z_mc_t = z_test[y_test == 0]
    x_mc_t = x_test[y_test == 0]

    w = omnifold_unfold(
        x_data, x_mc, z_mc,
        z_target=z_mc_t,
        niter=niter,
        epochs=epochs,
    )
```

This preserves behavior: training data is unchanged, and weights are still computed for the test MC events through `model2`.

- [ ] **Step 5: Run the new test and the existing import to verify**

Run: `uv run pytest tests/test_omnifold.py -v`
Expected: PASS (may take ~30–60s due to TF + OmniFold import and a 2-epoch train)

Run: `uv run python -c "import ran.baselines.omnifold as m; print(hasattr(m, 'omnifold_unfold'), hasattr(m, '_run_and_evaluate'))"`
Expected: `True True`

- [ ] **Step 6: Commit**

```bash
git add ran/baselines/omnifold.py tests/test_omnifold.py
git commit -m "refactor: extract omnifold_unfold for in-memory array unfolding"
```

---

## Task 3: Pure helpers in `cubic_sweep.py` (`response`, `make_particles`, `unfolded_wasserstein`)

**Files:**

- Create: `ran/experiments/__init__.py`
- Create: `ran/experiments/cubic_sweep.py`
- Test: `tests/test_cubic_sweep.py`

- [ ] **Step 1: Create the package marker**

Create `ran/experiments/__init__.py` with empty content:

```python

```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_cubic_sweep.py`:

```python
import json
import numpy as np
from scipy.stats import wasserstein_distance

from ran.experiments.cubic_sweep import (
    response,
    make_particles,
    unfolded_wasserstein,
)


def test_response_identity_at_zero():
    z = np.linspace(-3, 3, 100)
    np.testing.assert_array_equal(response(0.0, z), z)


def test_response_monotonic_for_positive_s():
    z = np.linspace(-3, 3, 1000)
    out = response(5.0, z)
    assert np.all(np.diff(out) > 0)


def test_make_particles_shapes_and_means():
    z_truth, z_gen = make_particles(50_000, seed=123)
    assert z_truth.shape == (50_000,)
    assert z_gen.shape == (50_000,)
    assert abs(z_truth.mean() - 0.0) < 0.05
    assert abs(z_gen.mean() - (-1.0)) < 0.05


def test_unfolded_wasserstein_uniform_weights_equals_unweighted():
    rng = np.random.default_rng(0)
    z_truth = rng.normal(0, 1, 5000)
    z_gen = rng.normal(-1, 1, 5000)
    w = np.ones_like(z_gen)
    got = unfolded_wasserstein(z_truth, z_gen, w)
    expected = wasserstein_distance(z_truth, z_gen)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_cubic_sweep.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ran.experiments.cubic_sweep'`

- [ ] **Step 4: Write the module with the three helpers**

Create `ran/experiments/cubic_sweep.py`:

```python
"""Sweep the cubic detector-distortion strength s and compare RAN vs OmniFold.

For each s in linspace(0, 20, n_points), apply a deterministic non-linear
detector response r(s, z) = z + s * z**3 to fixed particle-level samples
(z_truth ~ N(0,1), z_gen ~ N(-1,1)), unfold z_gen back toward z_truth with both
RAN and OmniFold, and record Wasserstein(z_truth, z_unfolded) for each.

Usage:
    python -m ran.experiments.cubic_sweep run_point --s_index=0 --sweep_dir=...
    python -m ran.experiments.cubic_sweep collect --sweep_dir=...
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import json
from pathlib import Path

import fire
import numpy as np
import numpy.typing as npt
from scipy.stats import wasserstein_distance


def response(s: float, z: npt.NDArray[np.double]) -> npt.NDArray[np.double]:
    """Deterministic non-linear detector response r(s, z) = z + s * z**3."""
    return z + s * z**3


def make_particles(
    n_samples: int, seed: int = 42
) -> tuple[npt.NDArray[np.double], npt.NDArray[np.double]]:
    """Draw fixed particle-level samples: z_truth ~ N(0,1), z_gen ~ N(-1,1)."""
    rng = np.random.default_rng(seed)
    z_truth = rng.normal(0.0, 1.0, size=n_samples)
    z_gen = rng.normal(-1.0, 1.0, size=n_samples)
    return z_truth, z_gen


def unfolded_wasserstein(
    z_truth: npt.NDArray[np.double],
    z_gen: npt.NDArray[np.double],
    weights: npt.NDArray[np.double],
) -> float:
    """Wasserstein distance between z_truth and the weighted z_gen distribution."""
    return float(
        wasserstein_distance(
            np.asarray(z_truth).ravel(),
            np.asarray(z_gen).ravel(),
            v_weights=np.asarray(weights).ravel(),
        )
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_cubic_sweep.py -v`
Expected: PASS (4 passed)

- [ ] **Step 6: Commit**

```bash
git add ran/experiments/__init__.py ran/experiments/cubic_sweep.py tests/test_cubic_sweep.py
git commit -m "feat: add cubic_sweep helpers (response, make_particles, unfolded_wasserstein)"
```

---

## Task 4: `run_point` — train RAN + OmniFold for one s

**Files:**

- Modify: `ran/experiments/cubic_sweep.py` (add imports + `run_point`)
- Test: `tests/test_cubic_sweep.py`

- [ ] **Step 1: Write the failing smoke test**

Append to `tests/test_cubic_sweep.py`:

```python
def test_run_point_writes_finite_metrics(tmp_path):
    from ran.experiments.cubic_sweep import run_point

    out = run_point(
        s_index=0,
        sweep_dir=tmp_path,
        n_samples=800,
        n_points=25,
        ran_epochs=2,
        omnifold_niter=1,
        omnifold_epochs=2,
    )

    assert out["s_index"] == 0
    assert out["s"] == 0.0
    assert np.isfinite(out["ran_wd"])
    assert np.isfinite(out["omnifold_wd"])

    written = json.loads((tmp_path / "s_00.json").read_text())
    assert written == out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cubic_sweep.py::test_run_point_writes_finite_metrics -v`
Expected: FAIL with `ImportError: cannot import name 'run_point'`

- [ ] **Step 3: Add imports and `run_point`**

In `ran/experiments/cubic_sweep.py`, add these imports after the existing `from scipy.stats import wasserstein_distance` line:

```python
from ran.data.datasets import RAN_Dataset
from ran.train import train
from ran.baselines.omnifold import omnifold_unfold
```

Then add this function after `unfolded_wasserstein`:

```python
def run_point(
    s_index: int,
    sweep_dir: str | Path,
    n_samples: int = 500_000,
    n_points: int = 25,
    seed: int = 42,
    batch_size: int = 1024,
    ran_epochs: int = 100,
    omnifold_niter: int = 3,
    omnifold_epochs: int = 50,
) -> dict:
    """Train RAN and OmniFold at one sweep point and write s_{index}.json.

    Particle samples are drawn once (fixed seed) so only s varies across points.
    RAN uses g(z_gen) weights; OmniFold uses its gen-level model weights. Both
    z_unfolded distributions are the z_gen samples reweighted accordingly.
    """
    sweep_dir = Path(sweep_dir)
    sweep_dir.mkdir(parents=True, exist_ok=True)

    s = float(np.linspace(0.0, 20.0, n_points)[s_index])
    z_truth, z_gen = make_particles(n_samples, seed=seed)
    x_data = response(s, z_truth)
    x_sim = response(s, z_gen)

    # --- RAN: data events carry z_truth (y=1, weight fixed to 1), sim events
    # carry z_gen (y=0, reweighted by g). Matches generate_gaussian_dataset. ---
    z = np.concatenate([z_truth, z_gen]).reshape(-1, 1).astype(np.double)
    x = np.concatenate([x_data, x_sim]).reshape(-1, 1).astype(np.double)
    y = np.concatenate(
        [np.ones(n_samples, dtype=np.ubyte), np.zeros(n_samples, dtype=np.ubyte)]
    )
    splits = RAN_Dataset(batch_size=batch_size).splits_from_arrays(z, x, y)
    g, _, _ = train(splits, dim=1, n_epochs=ran_epochs)

    raw = g(z_gen.reshape(-1, 1).astype(np.double)).numpy().ravel()
    w_ran = raw * len(raw) / raw.sum()

    # --- OmniFold: reweight z_gen toward z_truth via reco-level unfolding ---
    w_of = omnifold_unfold(
        x_data.reshape(-1, 1),
        x_sim.reshape(-1, 1),
        z_gen.reshape(-1, 1),
        niter=omnifold_niter,
        epochs=omnifold_epochs,
    )

    # Guard against non-finite weights (e.g. a saturated OmniFold classifier at
    # large s) so the Wasserstein call cannot crash; bad weights -> 0 mass.
    w_ran = np.where(np.isfinite(w_ran), w_ran, 0.0)
    w_of = np.where(np.isfinite(w_of), w_of, 0.0)

    ran_wd = unfolded_wasserstein(z_truth, z_gen, w_ran)
    of_wd = unfolded_wasserstein(z_truth, z_gen, w_of)

    out = {"s_index": int(s_index), "s": s, "ran_wd": ran_wd, "omnifold_wd": of_wd}
    (sweep_dir / f"s_{s_index:02d}.json").write_text(json.dumps(out, indent=2))
    print(f"s={s:.4f}  RAN={ran_wd:.6f}  OmniFold={of_wd:.6f}")
    return out
```

- [ ] **Step 4: Run the smoke test to verify it passes**

Run: `uv run pytest tests/test_cubic_sweep.py::test_run_point_writes_finite_metrics -v`
Expected: PASS (slow — trains a tiny RAN and a 1-iter OmniFold; allow ~1–2 min)

- [ ] **Step 5: Commit**

```bash
git add ran/experiments/cubic_sweep.py tests/test_cubic_sweep.py
git commit -m "feat: add run_point to train RAN + OmniFold per sweep point"
```

---

## Task 5: `collect` — concatenate results and plot

**Files:**

- Modify: `ran/experiments/cubic_sweep.py` (add `collect` + fire CLI)
- Test: `tests/test_cubic_sweep.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_cubic_sweep.py`:

```python
def test_collect_writes_results_and_plot(tmp_path):
    from ran.experiments.cubic_sweep import collect

    for i, s in enumerate([0.0, 10.0]):
        rec = {"s_index": i, "s": s, "ran_wd": 0.1 * (i + 1), "omnifold_wd": 0.2 * (i + 1)}
        (tmp_path / f"s_{i:02d}.json").write_text(json.dumps(rec))

    collect(sweep_dir=tmp_path, n_points=2)

    assert (tmp_path / "results.npz").exists()
    assert (tmp_path / "wasserstein_vs_s.pdf").exists()

    data = np.load(tmp_path / "results.npz")
    np.testing.assert_array_equal(data["s"], [0.0, 10.0])
    np.testing.assert_allclose(data["ran"], [0.1, 0.2])
    np.testing.assert_allclose(data["omnifold"], [0.2, 0.4])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cubic_sweep.py::test_collect_writes_results_and_plot -v`
Expected: FAIL with `ImportError: cannot import name 'collect'`

- [ ] **Step 3: Add `collect` and the fire CLI**

In `ran/experiments/cubic_sweep.py`, add this function after `run_point`:

```python
def collect(sweep_dir: str | Path, n_points: int = 25) -> None:
    """Gather all s_*.json into results.npz and a Wasserstein-vs-s PDF."""
    sweep_dir = Path(sweep_dir)
    records = [
        json.loads(f.read_text()) for f in sorted(sweep_dir.glob("s_*.json"))
    ]
    if not records:
        raise FileNotFoundError(f"No s_*.json files found in {sweep_dir}")
    records.sort(key=lambda r: r["s"])

    present = {r["s_index"] for r in records}
    missing = sorted(set(range(n_points)) - present)
    if missing:
        print(f"WARNING: missing s_index values (failed/incomplete tasks): {missing}")

    s = np.array([r["s"] for r in records])
    ran = np.array([r["ran_wd"] for r in records])
    omnifold = np.array([r["omnifold_wd"] for r in records])
    np.savez(sweep_dir / "results.npz", s=s, ran=ran, omnifold=omnifold)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(s, ran, "o-", label="RAN")
    ax.plot(s, omnifold, "s-", label="OmniFold")
    ax.set_xlabel(r"$s$ (cubic distortion strength)")
    ax.set_ylabel(r"Wasserstein($z_\mathrm{truth}$, $z_\mathrm{unfolded}$)")
    ax.set_title("Unfolding performance vs detector distortion")
    ax.legend()
    fig.tight_layout()
    fig.savefig(sweep_dir / "wasserstein_vs_s.pdf")
    plt.close(fig)
    print(
        f"Wrote {sweep_dir / 'results.npz'} and "
        f"{sweep_dir / 'wasserstein_vs_s.pdf'} ({len(records)} points)"
    )
```

Then add the fire CLI at the very end of the file:

```python
if __name__ == "__main__":
    fire.Fire({"run_point": run_point, "collect": collect})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cubic_sweep.py::test_collect_writes_results_and_plot -v`
Expected: PASS

- [ ] **Step 5: Verify the CLI dispatch works**

Run: `uv run python -m ran.experiments.cubic_sweep collect --help`
Expected: fire help text for `collect` (mentions `sweep_dir` and `n_points`), exit without error.

- [ ] **Step 6: Commit**

```bash
git add ran/experiments/cubic_sweep.py tests/test_cubic_sweep.py
git commit -m "feat: add collect step (results.npz + Wasserstein-vs-s plot) and CLI"
```

---

## Task 6: SLURM launcher `scripts/submit_sweep.sh`

**Files:**

- Create: `scripts/submit_sweep.sh`

- [ ] **Step 1: Write the launcher script**

Create `scripts/submit_sweep.sh` (run on the login node as `bash scripts/submit_sweep.sh`, NOT via `sbatch`; it computes a shared sweep dir, submits the 25-task array, then a dependent collect job). It follows the account/constraint conventions from `scripts/submit.sh`:

```bash
#!/bin/bash
# Launch the cubic-response RAN-vs-OmniFold sweep as a SLURM array job.
# Run on the login node:  bash scripts/submit_sweep.sh
set -euo pipefail

PROJECT_DIR=/global/u1/k/kdesai/RANv4
N_POINTS=25
SWEEP_DIR="${PROJECT_DIR}/runs/cubic_sweep_$(date -u +%Y-%m-%dT%H%M%SZ)"
mkdir -p "${SWEEP_DIR}"
echo "Sweep dir: ${SWEEP_DIR}"

# One array task per s value (indices 0..N_POINTS-1).
ARRAY_JOB=$(sbatch --parsable \
  --qos=regular --constraint=gpu --gpus=1 --account=m3246_g --time=02:00:00 \
  --array=0-$((N_POINTS - 1)) \
  --output="${SWEEP_DIR}/slurm-%A_%a.log" \
  --wrap="cd ${PROJECT_DIR} && uv run -m ran.experiments.cubic_sweep run_point --s_index=\${SLURM_ARRAY_TASK_ID} --sweep_dir=${SWEEP_DIR} --n_points=${N_POINTS}")
echo "Array job: ${ARRAY_JOB}"

# Collect after all array elements finish (afterany: run even if some failed).
COLLECT_JOB=$(sbatch --parsable \
  --qos=regular --constraint=gpu --gpus=1 --account=m3246_g --time=00:20:00 \
  --dependency=afterany:"${ARRAY_JOB}" \
  --output="${SWEEP_DIR}/slurm-collect-%j.log" \
  --wrap="cd ${PROJECT_DIR} && uv run -m ran.experiments.cubic_sweep collect --sweep_dir=${SWEEP_DIR} --n_points=${N_POINTS}")
echo "Collect job: ${COLLECT_JOB} (afterany:${ARRAY_JOB})"
echo "Results will land in ${SWEEP_DIR}/ (results.npz, wasserstein_vs_s.pdf)"
```

- [ ] **Step 2: Make it executable and syntax-check it**

Run: `chmod +x scripts/submit_sweep.sh && bash -n scripts/submit_sweep.sh && echo OK`
Expected: `OK` (no syntax errors; `bash -n` does not submit anything)

- [ ] **Step 3: Commit**

```bash
git add scripts/submit_sweep.sh
git commit -m "feat: add SLURM launcher for the cubic-response sweep"
```

---

## Task 7: Full local verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest tests/ -v`
Expected: all tests pass, including `test_datasets.py`, `test_omnifold.py`, `test_cubic_sweep.py`, and the pre-existing `test_config.py`.

- [ ] **Step 2: End-to-end mini sweep (2 points, tiny N) on CPU**

Run:

```bash
uv run -m ran.experiments.cubic_sweep run_point --s_index=0 --sweep_dir=/tmp/sweep_smoke --n_samples=800 --n_points=2 --ran_epochs=2 --omnifold_niter=1 --omnifold_epochs=2
uv run -m ran.experiments.cubic_sweep run_point --s_index=1 --sweep_dir=/tmp/sweep_smoke --n_samples=800 --n_points=2 --ran_epochs=2 --omnifold_niter=1 --omnifold_epochs=2
uv run -m ran.experiments.cubic_sweep collect --sweep_dir=/tmp/sweep_smoke --n_points=2
```

Expected: two `s_0*.json` files, then `results.npz` + `wasserstein_vs_s.pdf` in `/tmp/sweep_smoke`, and a "Wrote ... (2 points)" line with no missing-index warning.

- [ ] **Step 3: Clean up the smoke artifacts**

Run: `rm -rf /tmp/sweep_smoke`
Expected: no error.

---

## Notes for the implementer

- **Why data events carry `z_truth`:** this matches `generate_gaussian_dataset`, which concatenates `z_true` (data) and `z_gen` (MC) into the `z` feature. The generator's loss only flows through MC events (`_compute_weights` zeroes the generator-dependent term for `y=1`), so `g` never trains on `z_truth` — the "no network sees `z_true`" constraint holds. Do not "fix" this by removing `z_truth` from the dataset.
- **The metric is a weighted Wasserstein, not a resampling.** `z_unfolded` is always the original `z_gen` points carrying per-event weights (`v_weights`). Never draw new samples to represent the unfolded distribution.
- **OmniFold weight degeneracy at large `s` is expected signal, not a bug.** As detector overlap collapses, OmniFold's Step-1 classifier saturates and its weights become poorly conditioned — that degradation is exactly what the sweep is meant to surface. `omnifold_unfold` mean-normalizes; if NaNs appear, investigate but treat large-`s` instability as part of the result.
- **GPU vs CPU:** tests and the mini sweep run on CPU. The real sweep (`n_samples=500_000`) is GPU work submitted via `scripts/submit_sweep.sh`.
