# Residual Dispersion Penalty Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in `--dispersion-mode residual` that penalises only the part of `g`'s weight variance the detector cannot see, `λ·E[(w − h(x))²]`, with `h(x) ≈ E[w|x]` learned alongside `g` and `d`.

**Architecture:** A third network `h(x)` (the *regressor*) rides in `TrainState` and is updated on every discriminator step by MSE against `g`'s normalised MC weights, held constant. In residual mode `g`'s penalty becomes the masked MC mean of `(w − h(x))²` with `h` held constant, which is the exact gradient of `E[Var(w|x)]` at the optimal `h` because `E[w − E[w|x] | x] = 0`. Total mode (the default) builds no `h` and must stay bit-for-bit what it is today.

**Tech Stack:** JAX (`lax.scan`, `stateless_call`/`stateless_apply`), Keras 3 on the JAX backend, Typer, pytest.

**Spec:** No separate spec. The design came out of the 2026-10-30 investigation of `runs/2026-10-27T222108Z`, and the evidence is in `dispersion_split.py` (repo root) and `jets_split.json`:

- Var(w) = Var(E[w|x]) + E[Var(w|x)], meaning visible + residual. Only the visible part is constrained by `d`.
- 1D Gaussian (universal response): the current total-variance penalty at λ = 0.015 cuts visible 0.330 → 0.196 and residual 0.157 → 0.077 relative to λ = 0. Particle W1 is 0.097 against 0.016 at λ = 0.
- Jets: λ = 0 → 0.015 cuts residual 0.218 → 0.073 (−67%) and visible 0.167 → 0.120 (−28%), with detector-level match unchanged and particle level much better. Whether the visible cut helps jets is the open question this mode answers.

## Global Constraints

- No network sees `z_true`. `h` takes `x` and is fit on MC rows only (`(1 − y) * mask`). Its targets are `g(z_gen)` weights, which are already MC-only.
- `--dispersion-mode` defaults to `total`, and a total-mode run is bit-for-bit identical to the pre-change code (same seeds give the same `history`).
- The history stays three weighted-BCE columns (`train_d`, `train_g`, `val_d`). The penalty never enters a recorded number (`notes/agent/training-loop.md`).
- `fused=True` and `fused=False` must agree in both modes (`TestFusion`).
- Float32 everywhere (`EVENT_DTYPE`, `notes/agent/precision.md`).
- Imports stay at module top, with no lazy imports (the user's standing preference).
- `uv run` for all Python. `just validate` and `just test-fast` must pass at the end of every task.
- Out of scope: `uncertainty freeze/run`, `leakage-check`, and `benchmarks/`. They keep calling `train` with its default (total) mode.

## Review Focus

1. **Nature rows leaking into `h`'s fit.** Data rows carry `w = 1` by construction, so fitting them would drag `h` toward 1 wherever data is dense. Expected: `h`'s loss ignores rows with `y = 1` and padded rows (Task 1 tests `weight_residual`; Task 2 tests the regressor loss through `disc_step`).
2. **An untrained `h` at the start of training.** For the first steps `h` is a random function of `x`. Expected: a constant offset in `h` has no effect on `g`'s gradient, because the weights are normalised to mean 1 (Task 1, `test_constant_offset_in_h_does_not_move_g`).
3. **Default runs drifting.** Building `h` would consume RNG and change `TrainState`'s pytree. Expected: total mode builds no `h`, carries empty lists, and reproduces a golden pre-change history (Task 2, `test_total_mode_matches_the_pre_change_golden`).
4. **Old run directories.** `params.npz` without `h_*` keys and `config.json` without `dispersion_mode` must still reload. Expected: empty `h` lists and total mode (Task 2, `test_load_params_without_regressor_keys`; Task 3, `test_reload_of_a_run_without_dispersion_mode`).
5. **λ = 0 in residual mode.** `h` trains but must be inert. Expected: the `g` and `d` trajectories equal total mode at λ = 0 (Task 2, `test_residual_mode_at_zero_lambda_matches_total_mode`).

---

### Task 1: `weight_residual`, the residual penalty as a pure function

**Files:**
- Modify: `src/deconvolve/training/engine.py` (beside `weight_dispersion`, ~line 285)
- Modify: `src/deconvolve/training/__init__.py` (export it)
- Test: `tests/test_train.py` (new class `TestWeightResidual`, beside the `weight_dispersion` tests)

**Interfaces:**
- Produces: `weight_residual(w: Float[Array | NDArray[np.single], " n"], h: Float[Array | NDArray[np.single], " n"], y: ..., mask: ..., /) -> Float[Array, ""]`. This is the masked MC-row mean of `(w − h)²`, with the same masking and the same `EPS` guard as `weight_dispersion`. The docstring carries the visible/residual decomposition and the exact-gradient argument from the Architecture paragraph.

- [ ] **Step 1: Write the failing tests** in `TestWeightResidual`:

```python
def test_matching_h_has_no_residual(self) -> None:
    # w == h on every MC row -> 0.0
def test_constant_h_at_one_equals_weight_dispersion(self) -> None:
    # w normalised to mean 1 (via normalize_weights on lognormal raw, seed 0, n=4096),
    # h = ones  ->  weight_residual(w, h, y, mask) == approx(weight_dispersion(w, y, mask), rel=1e-6)
def test_ignores_the_nature_rows(self) -> None:
    # y=[1,1,0,0], w=[1,1,0.5,1.5], h=[9,9,1,1] -> 0.25
def test_ignores_padded_rows(self) -> None:
    # w=[0.5,1.5,0.5,1.5,0,0], h=[1,1,1,1,7,7], mask=[1,1,1,1,0,0] -> 0.25
def test_constant_offset_in_h_does_not_move_g(self) -> None:
    # raw ~ lognormal(seed 1, n=256); h ~ normal(seed 2)
    # f(raw, h) = weight_residual(normalize_weights(raw, y, mask), h, y, mask)
    # jax.grad(f)(raw, h) == approx(jax.grad(f)(raw, h + 3.0), abs=1e-5)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_train.py::TestWeightResidual -v`
Expected: FAIL with `ImportError: cannot import name 'weight_residual'`

- [ ] **Step 3: Implement `weight_residual`** in `engine.py`, `@jaxtyped(typechecker=beartype)`, following `weight_dispersion`'s body shape.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_train.py::TestWeightResidual -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/deconvolve/training/engine.py src/deconvolve/training/__init__.py tests/test_train.py
git commit -m "feat(training): add weight_residual, the invisible part of weight variance"
```

---

### Task 2: Train `h` in the engine and switch `g`'s penalty by mode

**Files:**
- Modify: `src/deconvolve/coretypes/enums.py` (add `DispersionMode`)
- Modify: `src/deconvolve/training/models.py` (add `build_regressor`)
- Modify: `src/deconvolve/training/engine.py`: `EpochParams`, `TrainResult`, `TrainState`, `_make_steps`, `_initial_state`, `_restore`, `_make_epoch` (emit `h` params), `train`
- Modify: `tests/typecheck/keras_protocol_contracts.py:68` (new `_make_steps` signature)
- Test: `tests/test_train.py` (`TestTrainSteps._setup`, new `TestResidualPenalty`, `TestFusion`)

**Interfaces:**
- Consumes: `weight_residual` (Task 1).
- Produces:
  - `class DispersionMode(StrEnum): total = auto(); residual = auto()`, with a docstring naming what each one penalises.
  - `build_regressor(dim: int = 1, hidden_units: int = 64, n_layers: int = 2) -> DeconvolveModel`. Same body as `build_generator`, with softplus output (so `E[w|x] > 0`) and `name="regressor"`. Its input is `x`.
  - `TrainState` gains `h_trainable: Variables`, `h_non_trainable: Variables` and `opt_h: Variables`, all `[]` in total mode.
  - `EpochParams` gains `h_trainable` and `h_non_trainable` (`[]` in total mode). `save_params`/`load_params` need no change: `load_params` already yields `[]` for a field with no keys. Add a test pinning that.
  - `TrainResult` gains `h: DeconvolveModel | None = None` after `sigmas`.
  - `_make_steps(g, d, opt_g, opt_d, lambda_dispersion: float, dispersion_mode: DispersionMode, h: DeconvolveModel | None, opt_h: StatelessOptimizer | None)`. All arguments are required; `h`/`opt_h` are `None` exactly when the mode is `total`, and it raises `ValueError` otherwise.
  - `train(..., lambda_dispersion: float = 0.015, dispersion_mode: DispersionMode = DispersionMode.total, log_every: int = 1, *, fused: bool = True)`.

Behaviour the steps must have, in residual mode:
- `disc_step` updates `d` exactly as now and **also** takes one `opt_h` step on the same batch. The loss is `weight_residual(stop_gradient(w), h(x), y, mask)`, differentiated in `h` only, where `w` comes from `_weights(..., training=False)` as `d` already uses.
- `gen_step`'s penalty term becomes `lambda_dispersion * weight_residual(w, stop_gradient(h(x)), y, mask)`. The returned (recorded) loss is still `adversarial`.
- `h`'s optimizer is Adam at `lr_d`. Don't add a flag for it.
- `train` builds `h` and `opt_h` **after** `g` and `d`, and only in residual mode, so the initial weights of `g` and `d` don't depend on the mode. `_restore` writes `h`'s selected epoch back when present. Selection still reads `g` only.

- [ ] **Step 1: Capture the golden history before touching the engine.** On the current code, run `TestDispersionPenalty._run(lambda_dispersion=0.015, epochs=5)` and paste its `history["train_g"]` and `history["val_d"]` into `TestResidualPenalty` as module constants `_GOLDEN_TRAIN_G` and `_GOLDEN_VAL_D`.

Run: `uv run python -c "import sys; sys.path.insert(0, 'tests'); from test_train import TestDispersionPenalty as T; r = T()._run(0.015, epochs=5); print(r.history['train_g'], r.history['val_d'])"`

- [ ] **Step 2: Write the failing tests**

```python
# TestResidualPenalty (marked slow; reuses TestDispersionPenalty._splits)
def test_total_mode_matches_the_pre_change_golden(self) -> None:
    # train(..., lambda_dispersion=0.015, n_epochs=5) default mode
    # history["train_g"] == approx(_GOLDEN_TRAIN_G, rel=1e-6); same for val_d
def test_total_mode_builds_no_regressor(self) -> None:
    # result.h is None; every leaf list in result.params.h_trainable == []
def test_residual_mode_at_zero_lambda_matches_total_mode(self) -> None:
    # dispersion_mode=residual, λ=0  vs  total, λ=0 (n_epochs=5)
    # train_g, train_d, val_d histories equal (approx rel=1e-6)
def test_residual_mode_trains_the_regressor(self) -> None:
    # residual, λ=0.015, n_epochs=20, lr_g=1e-2
    # On the MC test rows: mean((h(x) - w)^2) < var(w)   -- h explains some of w
    # and h's selected-epoch weights differ from its epoch-0 weights
def test_residual_penalty_tames_the_weights(self) -> None:
    # residual λ=100 vs λ=0, n_epochs=40, lr_g=1e-2: min(val_ess) strictly larger
def test_recorded_loss_stays_on_the_bce_scale(self) -> None:
    # residual λ=100: all(0 < v < 2 for v in history["train_g"])
def test_load_params_without_regressor_keys(self, tmp_path) -> None:
    # save_params a total-mode EpochParams, load_params -> h_trainable == [] and h_non_trainable == []

# TestTrainSteps (extend _setup with a `dispersion_mode` arg that builds h/opt_h when residual)
def test_residual_disc_step_updates_d_and_h_not_g(self) -> None:
def test_regressor_loss_ignores_nature_rows(self) -> None:
    # one disc_step on a batch, then the same batch with every nature row's x replaced by 1e3:
    # resulting h_trainable equal (approx) -- nature rows contribute no gradient to h

# TestFusion
# parametrize test_fused_and_eager_runs_agree over DispersionMode (residual with λ=0.015)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `uv run pytest tests/test_train.py -k "Residual or residual" -v`
Expected: FAIL with `ImportError: cannot import name 'DispersionMode'` (and `TypeError` on `train(..., dispersion_mode=...)` once that resolves)

- [ ] **Step 4: Implement** the enum, `build_regressor`, the state/params/result fields and the step behaviour above. Update `TestTrainSteps._setup` and `keras_protocol_contracts.py` for the new `TrainState` fields and `_make_steps` signature. Extend `TrainState`'s and `EpochParams`' docstrings by one line on `h`.

- [ ] **Step 5: Run the train tests, including slow ones**

Run: `uv run pytest tests/test_train.py -v`
Expected: all pass, including the unchanged `TestDispersionPenalty` and the golden test

- [ ] **Step 6: Run full validation**

Run: `just validate && just test-fast`
Expected: clean

- [ ] **Step 7: Commit**

```bash
git add src/deconvolve/coretypes/enums.py src/deconvolve/training/ tests/test_train.py tests/typecheck/keras_protocol_contracts.py
git commit -m "feat(training): learn h(x)=E[w|x] and penalise only the residual in residual mode"
```

---

### Task 3: `--dispersion-mode` on `deconvolve train`, recorded and documented

**Files:**
- Modify: `src/deconvolve/cli.py`, `train_command` (new option after `lambda_dispersion`)
- Modify: `src/deconvolve/workflows/train.py`: `run`, `_pipeline` (pass it to `train`, record it in `hyperparameters`), `_write_run_dir` (save `regressor.keras` when `result.h` is not `None`)
- Modify: `docs/user-guide/cli.md` (`deconvolve train` section usage line and table), `docs/user-guide/configuration.md` (the `[train]` key table), `README.md` (the train option table), `notes/agent/cli-and-running.md` (one sentence beside `--lambda-dispersion`), `notes/agent/training-loop.md` (one paragraph: `h` rides in `TrainState`, is trained on disc steps, and is never selected on)
- Test: `tests/test_cli.py`, `tests/test_workflow.py`

**Interfaces:**
- Consumes: `DispersionMode`, `TrainResult.h`, and `train(..., dispersion_mode=...)` (Task 2).
- Produces:
  - CLI option `--dispersion-mode <total|residual>`, default `total`. Help text: `"What the dispersion penalty measures: total weight variance, or only the part the detector cannot see."` It is layerable like the other train options, so the TOML key is `dispersion-mode` and the env var is `DECONVOLVE_TRAIN_DISPERSION_MODE`.
  - `config.json` gains `"dispersion_mode": "total" | "residual"` beside `lambda_dispersion`.
  - In residual mode, `artifacts/regressor.keras` exists next to `generator.keras`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py, beside the lambda_dispersion default assertion (~line 148)
assert calls[0]["dispersion_mode"] == DispersionMode.total
def test_dispersion_mode_flag_reaches_train(...) -> None:
    # invoke ["train", "--dispersion-mode", "residual"] with the existing monkeypatched run
    # -> calls[0]["dispersion_mode"] == DispersionMode.residual

# tests/test_workflow.py, using the module's existing tiny-run fixture
def test_residual_run_records_mode_and_saves_regressor(tmp_path) -> None:
    # config.json["dispersion_mode"] == "residual"; (artifacts / "regressor.keras").exists()
def test_total_run_saves_no_regressor(tmp_path) -> None:
    # config.json["dispersion_mode"] == "total"; not (artifacts / "regressor.keras").exists()
def test_reload_of_a_run_without_dispersion_mode(tmp_path) -> None:
    # write a run, delete "dispersion_mode" from its config.json, `--load-run` it -> succeeds
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/test_workflow.py -k dispersion -v`
Expected: FAIL (`KeyError: 'dispersion_mode'` / no such option)

- [ ] **Step 3: Implement** the option, the plumbing and the `regressor.keras` save. Then update the five docs files.

- [ ] **Step 4: Run the tests, including the docs-sync test**

Run: `uv run pytest tests/test_cli.py tests/test_workflow.py tests/test_cli_docs.py tests/test_config_cli.py -v`
Expected: all pass

- [ ] **Step 5: Full validation**

Run: `just validate && just test-fast`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/deconvolve/cli.py src/deconvolve/workflows/train.py docs/ README.md notes/agent/ tests/
git commit -m "feat(cli): add --dispersion-mode to train"
```

---

### Task 4: The experiment that decides whether this ships as the default

There's no code here. Its deliverable is a results note. Run the Gaussian part locally (CPU is fine, ~1 min per run) and the jets part on the HPC interactive node.

**Files:**
- Create: `notes/working-notes/2026-10-residual-dispersion.md`

- [ ] **Step 1: 1D Gaussian, three arms, same seed**

```sh
for arm in "total 0" "total 0.015" "residual 0.015"; do set -- $arm
  uv run deconvolve train -D gaussian --config params/1d_default.yaml --seed 446727034 \
    --dispersion-mode $1 --lambda-dispersion $2 --no-plots --run-dir runs/resid-gauss-$1-$2
done
uv run python dispersion_split.py runs/resid-gauss-* --no-references
```

Expected: `residual 0.015` has particle W1 close to `total 0` (≈ 0.016) rather than to `total 0.015` (≈ 0.097), and its visible variance is within 0.02 of `total 0`'s 0.330.

- [ ] **Step 2: Jets, residual arm at two strengths** (the residual is a third of the total, so 0.015 penalises less than before)

```sh
for lam in 0.015 0.05; do
  uv run deconvolve train -D jets --n-samples 1600000 --hidden-units 128 --n-layers 3 \
    --seed 337443673 --dispersion-mode residual --lambda-dispersion $lam --no-plots
done
uv run python dispersion_split.py runs/2026-10-30T203443Z runs/2026-10-31T013457Z runs/<two new runs> --out jets_split_residual.json
```

- [ ] **Step 3: Write the note.** Include both tables, each arm's particle-level W1 per observable and the joint sliced-Wasserstein improvement, and a verdict:
  - **Ship as default:** residual mode matches or beats total-mode λ = 0.015 on the jets joint particle metric and fixes the Gaussian.
  - **Keep opt-in:** jets get worse. Record that the visible shrinkage was doing useful work, probably damping the fit to the non-universal response.

  Changing the default is a separate change: it needs the `benchmarks/README.md` evidence and the `test_cli.py` default comment updated.

- [ ] **Step 4: Commit**

```bash
git add notes/working-notes/2026-10-residual-dispersion.md
git commit -m "notes: residual dispersion penalty, Gaussian and jets results"
```
