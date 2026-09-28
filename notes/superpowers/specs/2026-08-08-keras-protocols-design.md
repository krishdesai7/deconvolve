# Targeted Keras Protocols Design

## Goal

Replace permissive uses of untyped Keras model, variable, and optimizer
surfaces with small structural contracts that Pyrefly can check throughout
RAN's JAX training and inference paths.

The change is type-only. It must not wrap Keras objects, alter runtime model
construction, change training behavior, or attempt to describe the complete
Keras API.

## Scope

The contracts cover only operations used by RAN:

- Models are callable, support `stateless_call`, expose trainable and
  non-trainable variables, and can be saved.
- Variables expose their JAX value and accept assignment of a JAX value.
- Optimizers can be built, expose their variables, and support
  `stateless_apply`.

Keras layer construction in `ran.models` remains outside the typed boundary.
Calls such as `keras.Input`, `keras.layers.Dense`, and `keras.Model` originate
in untyped Keras source and cannot be made safe by a consumer-side protocol.

## Architecture

Add three runtime-light protocols to `ran.rantypes.types`:

1. `KerasVariable` describes `value` and `assign`.
2. `RANModel` describes the combined model surface used across training,
   evaluation, plotting, persistence, and experiments.
3. `StatelessOptimizer` describes the optimizer surface used by the JAX
   training loop.

The module must stay safe to import before a Keras backend is selected. Keras
itself therefore must not be imported by the protocol module. JAX, NumPy, and
path types used only in annotations remain guarded by `TYPE_CHECKING`, relying
on postponed annotations and Python 3.13 lazy type aliases.

`RANModel` is deliberately project-specific rather than a general Keras model
abstraction. One combined model contract avoids generic or intersection-type
machinery that would add complexity without improving the current call sites.

## Type Flow

Model builders continue constructing real `keras.Model` instances but declare
the project-facing result as `RANModel`. `TrainResult.g` and `TrainResult.d`
carry that contract to downstream consumers. Training helpers accept
`RANModel` and `StatelessOptimizer`, so Pyrefly checks stateless calls and
updates against explicit JAX-array state.

`keras.saving.load_model` remains an untyped boundary. Its result is assigned
to `RANModel`, making the local trust boundary explicit while providing a
checked surface everywhere after loading.

## Error Handling and Runtime Behavior

The protocols introduce no runtime validation and use no `runtime_checkable`
decorator. Existing integration tests are the runtime conformance check: they
train real Keras models, perform stateless updates, assign state back, and save
and reload the result.

If a future Keras release changes one of these runtime operations, the existing
tests should fail even though Keras's own untyped return values remain
permissive at the protocol boundary.

## Testing

Use a static contract fixture to force Pyrefly to check representative model,
variable, and optimizer operations. Before adding the protocols, checking that
fixture must fail because the protocol names do not exist. After implementation
it must pass and reveal the intended JAX-array and state types.

Then run:

- `uv run pyrefly check`
- focused training, workflow, evaluation, plotting, and experiment tests
- the complete `uv run pytest -q` suite
- Ruff formatting and lint checks for modified files

The existing runtime tests remain authoritative for compatibility with actual
Keras objects; no mock-only runtime conformance layer is added.

## Non-Goals

- Supplying stubs for all of Keras.
- Checking keyword arguments passed to Keras layer constructors.
- Encoding tensor shapes in the type system.
- Distinguishing generator and discriminator state with nominal types.
- Adding wrappers, adapters, casts at every call site, or runtime protocol
  checks.
