# Targeted Keras Protocols Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give Pyrefly an explicit, project-owned contract for the Keras model, variable, and optimizer operations used by RAN.

**Architecture:** Define three runtime-light structural protocols in `ran.rantypes`, then annotate model construction, JAX training, persistence, and inference against them. A non-pytest static fixture exercises the contracts directly; existing integration tests continue to prove that real Keras objects conform at runtime.

**Tech Stack:** Python 3.13, `typing.Protocol`, Keras 3.15 with the JAX backend, JAX 0.11, Pyrefly 1.2, pytest 9.1, Ruff 0.16.

## Global Constraints

- Do not import Keras from `ran.rantypes.types`; importing RAN's type package must not select a Keras backend.
- Do not add wrappers, adapters, runtime protocol checks, Keras stubs, tensor-shape typing, or new dependencies.
- Keep the contracts limited to operations already used by RAN.
- Preserve all runtime training, evaluation, plotting, and persistence behavior.
- Use Pyrefly failures as the red phase for type-only changes and real Keras tests as runtime conformance evidence.

---

### Task 1: Define and Export the Structural Contracts

**Files:**
- Create: `tests/typecheck/keras_protocol_contracts.py`
- Modify: `src/ran/rantypes/types.py:1-30`
- Modify: `src/ran/rantypes/__init__.py:20-24`
- Modify: `src/ran/rantypes/schema.py:15-31`

**Interfaces:**
- Consumes: existing `Variables = list[JaxArray]` alias.
- Produces: `KerasVariable`, `RANModel`, and `StatelessOptimizer`, re-exported from `ran.rantypes`; `TrainResult.g` and `.d` typed as `RANModel`.

- [ ] **Step 1: Write the failing static contract fixture**

```python
from __future__ import annotations

from typing import assert_type

from jax._src.basearray import Array as JaxArray
from numpy.typing import ArrayLike

from ran.rantypes import KerasVariable, RANModel, StatelessOptimizer, Variables


def check_protocol_surfaces(
    model: RANModel,
    optimizer: StatelessOptimizer,
    variable: KerasVariable,
    inputs: ArrayLike,
    state: Variables,
) -> None:
    assert_type(model.trainable_variables, list[KerasVariable])
    assert_type(model.non_trainable_variables, list[KerasVariable])
    assert_type(model(inputs), JaxArray)
    output, next_state = model.stateless_call(state, state, inputs, training=True)
    assert_type(output, JaxArray)
    assert_type(next_state, Variables)
    model.save("model.keras")

    assert_type(variable.value, JaxArray)
    variable.assign(state[0])

    optimizer.build(model.trainable_variables)
    assert_type(optimizer.variables, list[KerasVariable])
    trainable, optimizer_state = optimizer.stateless_apply(state, state, state)
    assert_type(trainable, Variables)
    assert_type(optimizer_state, Variables)
```

- [ ] **Step 2: Run Pyrefly to verify the fixture fails for missing exports**

Run: `uv run pyrefly check tests/typecheck/keras_protocol_contracts.py`

Expected: nonzero exit with missing-import diagnostics for `KerasVariable`, `RANModel`, and `StatelessOptimizer`.

- [ ] **Step 3: Add the minimal protocol definitions**

Add runtime `Protocol` imports and type-only `PathLike`/`ArrayLike` imports in `types.py`, then define:

```python
class KerasVariable(Protocol):
    @property
    def value(self) -> JaxArray: ...

    def assign(self, value: JaxArray) -> None: ...


class RANModel(Protocol):
    @property
    def trainable_variables(self) -> list[KerasVariable]: ...

    @property
    def non_trainable_variables(self) -> list[KerasVariable]: ...

    def __call__(self, inputs: ArrayLike) -> JaxArray: ...

    def stateless_call(
        self,
        trainable_variables: Variables,
        non_trainable_variables: Variables,
        inputs: ArrayLike,
        *,
        training: bool,
    ) -> tuple[JaxArray, Variables]: ...

    def save(self, filepath: str | PathLike[str]) -> None: ...


class StatelessOptimizer(Protocol):
    @property
    def variables(self) -> list[KerasVariable]: ...

    def build(self, variables: list[KerasVariable]) -> None: ...

    def stateless_apply(
        self,
        optimizer_variables: Variables,
        gradients: Variables,
        trainable_variables: Variables,
    ) -> tuple[Variables, Variables]: ...
```

Re-export all three names from `rantypes/__init__.py`. Replace the type-only Keras import in `schema.py` with `RANModel` and use it for both fields of `TrainResult`.

- [ ] **Step 4: Run the static fixture and focused runtime test**

Run: `uv run pyrefly check tests/typecheck/keras_protocol_contracts.py`

Expected: exit 0.

Run: `uv run pytest -q tests/test_train.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the contracts**

```bash
git add src/ran/rantypes tests/typecheck/keras_protocol_contracts.py
git commit -m "feat: define targeted Keras protocols"
```

### Task 2: Type the Model Builders and JAX Training Boundary

**Files:**
- Modify: `tests/typecheck/keras_protocol_contracts.py`
- Modify: `src/ran/models.py:1-25`
- Modify: `src/ran/train.py:18-26,83-88,244-310`

**Interfaces:**
- Consumes: `KerasVariable`, `RANModel`, and `StatelessOptimizer` from Task 1.
- Produces: model builders returning `RANModel`; training helpers consuming protocol types; typed optimizer updates without casts.

- [ ] **Step 1: Extend the fixture with training-boundary checks**

```python
from ran.models import build_discriminator, build_generator
from ran.train import _make_steps


def check_training_boundary(model: RANModel, optimizer: StatelessOptimizer) -> None:
    assert_type(build_generator(), RANModel)
    assert_type(build_discriminator(), RANModel)
    _make_steps(model, model, optimizer, optimizer)
```

- [ ] **Step 2: Run Pyrefly to verify the existing nominal annotations fail**

Run: `uv run pyrefly check tests/typecheck/keras_protocol_contracts.py`

Expected: nonzero exit because builders return `keras.Model` and `_make_steps` requires nominal Keras model and optimizer types.

- [ ] **Step 3: Move construction and training annotations onto the protocols**

In `models.py`, enable postponed annotations, import `RANModel` under
`TYPE_CHECKING`, and use it as both builder return types.

In `train.py`, import the three protocol names under `TYPE_CHECKING`; change
`_make_steps` to accept two `RANModel` and two `StatelessOptimizer` values;
change `_assign` to accept `list[KerasVariable]`; declare the local models and
optimizers with protocol types. Remove the two `cast("tuple[Variables,
Variables]", ...)` calls around `stateless_apply`, leaving their calls and
tuple unpacking intact.

- [ ] **Step 4: Run static and real-training verification**

Run: `uv run pyrefly check tests/typecheck/keras_protocol_contracts.py`

Expected: exit 0.

Run: `uv run pytest -q tests/test_train.py tests/test_workflow.py`

Expected: all tests pass.

- [ ] **Step 5: Commit the training migration**

```bash
git add src/ran/models.py src/ran/train.py tests/typecheck/keras_protocol_contracts.py
git commit -m "refactor: type JAX training against protocols"
```

### Task 3: Propagate the Model Contract Through Consumers

**Files:**
- Modify: `tests/typecheck/keras_protocol_contracts.py`
- Modify: `src/ran/workflow.py:14-24,107-109,157-160,255-261`
- Modify: `src/ran/evaluate.py:20-31,128-130`
- Modify: `src/ran/plotting.py:10-20,67-68,293-330`
- Modify: `src/ran/leakage.py:18-35,72-74`

**Interfaces:**
- Consumes: `RANModel` from Task 1 and protocol-returning builders/results from Task 2.
- Produces: checked callable/saveable model surfaces in persistence, evaluation, plotting, and leakage code.

- [ ] **Step 1: Extend the fixture with downstream-consumer checks**

```python
from pathlib import Path

from numpy.typing import NDArray

from ran.evaluate import _get_weights as evaluate_weights
from ran.plotting import _get_weights as plotting_weights
from ran.rantypes import TrainResult
from ran.workflow import _load_artifacts


def check_model_consumers(
    model: RANModel,
    inputs: NDArray,
    result: TrainResult,
) -> None:
    evaluate_weights(model, inputs)
    plotting_weights(model, inputs)
    assert_type(result.g, RANModel)
    loaded, _ = _load_artifacts(Path("run"))
    assert_type(loaded, RANModel)
```

- [ ] **Step 2: Run Pyrefly to verify nominal consumer annotations fail**

Run: `uv run pyrefly check tests/typecheck/keras_protocol_contracts.py`

Expected: nonzero exit where `RANModel` is passed to consumers that still require `keras.Model`, and where `_load_artifacts` still returns `keras.Model`.

- [ ] **Step 3: Replace consumer-facing nominal annotations**

Import `RANModel` under `TYPE_CHECKING` in each consumer. Replace direct
`keras.Model` annotations on parameters, return values, and locals in
`workflow.py`, `evaluate.py`, `plotting.py`, and `leakage.py`. Keep runtime
Keras imports wherever model loading or other runtime Keras operations still
use them.

- [ ] **Step 4: Run focused and complete verification**

Run: `uv run pyrefly check`

Expected: 0 errors, retaining only the pre-existing unrelated suppression.

Run: `uv run pytest -q tests/test_train.py tests/test_workflow.py tests/test_completion_logging.py tests/test_cubic_sweep.py`

Expected: all tests pass.

Run: `uv run ruff format --check src/ran tests/typecheck/keras_protocol_contracts.py && uv run ruff check src/ran tests/typecheck/keras_protocol_contracts.py`

Expected: exit 0.

Run: `uv run pytest -q`

Expected: complete suite passes.

- [ ] **Step 5: Commit the consumer migration**

```bash
git add src/ran/workflow.py src/ran/evaluate.py src/ran/plotting.py src/ran/leakage.py tests/typecheck/keras_protocol_contracts.py
git commit -m "refactor: propagate RAN model protocol"
```

### Task 4: Review the Complete Branch

**Files:**
- Review: all changes since `master`

**Interfaces:**
- Consumes: completed protocol implementation and verification evidence.
- Produces: review findings resolved or explicitly documented.

- [ ] **Step 1: Inspect branch scope**

Run: `git diff --stat master...HEAD && git diff --check master...HEAD`

Expected: only the design, plan, static fixture, type definitions, and annotation migrations are present; no whitespace errors.

- [ ] **Step 2: Request an independent code review**

Provide the reviewer with `master` as the base, `HEAD` as the target, the approved design document, and the requirement that runtime behavior remain unchanged.

- [ ] **Step 3: Resolve every critical or important finding**

For each valid finding, first extend the static fixture or existing runtime tests so the issue reproduces, then make the smallest correction and rerun the relevant verification command.

- [ ] **Step 4: Run final branch verification**

Run: `uv run pyrefly check && uv run ruff format --check src/ran tests/typecheck/keras_protocol_contracts.py && uv run ruff check src/ran tests/typecheck/keras_protocol_contracts.py && uv run pytest -q`

Expected: every command exits 0; Pyrefly reports 0 errors with only the pre-existing suppression.
