# Training Loop

`src/deconvolve/training/engine.py` is hand-rolled, since the two-optimizer min-max game does not
fit `Model.fit` — but it is not a Python loop over batches. **A whole run
compiles to one XLA program.** Model state lives in JAX pytrees (`TrainState`)
for the duration, updates go through `stateless_call`/`stateless_apply`, and the
values are written back into the Keras models at the end, so the returned
objects are ordinary saveable `keras.Model`s.

The nesting, innermost out:

1. `lax.scan` over the `n_disc_steps` discriminator batches of one group.
2. One generator update per group, on the group's first batch.
3. `lax.scan` over the groups — one epoch. Then a `lax.scan` over the
   pre-batched val split, accumulating `(masked total, count)` and dividing
   once, so validation is a true mean rather than a mean of per-batch means.
4. `lax.scan` over the epochs, carrying `RunCarry` — just the state and the
   PRNG key — and emitting every epoch's `(train_d, train_g, val_d)` row
   plus its full `EpochParams`: both networks' trainable and non-trainable
   variables, stacked on a leading epoch axis. A fixed trip count is what
   lets `scan` stack that output at all; a `while_loop` cannot emit
   per-epoch arrays without a preallocated buffer of its own.

The history has three columns, not four: every one is the weighted BCE on the
same scale (`_make_pass` negates `g_loss` back before recording it), and
validation measures that BCE in exactly one place, so both networks are scored
by a single number. A `val_g` column could only repeat `val_d` — which is what
it used to hold, and what drew two identical curves on `losses.pdf`. Runs saved
before this carry the extra key; nothing reads it, and `val_d` kept its name, so
they still reload.

Nothing about model quality is decided inside the trace. Per-epoch logging
goes through `jax.debug.callback(..., ordered=True)` so the Rich handler
still sees it from inside the loop, and that is the loop's only side effect;
selection is a host-side read of what `scan` already emitted, once the run
is over.

Selection does not happen in the loop. A GAN's loss is not a proxy for a
single scalar objective monotonically related to model quality: it oscillates
around its equilibrium by construction, a flat curve cannot be told from a
stalled one, and `log 2 - BCE` estimates a divergence only when `d` is
optimal — which nothing reports. Both criteria once built on it were unsound
and are gone.

Instead the scan emits every epoch's parameters (`EpochParams`, ~27 MB for
100 epochs of both networks), and `engine.train` picks the epoch minimizing a
weighted MMD (`training/mmd.py`) against a fixed subsample of the validation split. MMD is a
divergence — zero iff the distributions match, monotone in mismatch, no
adversary and no optimization — so patience and early stopping would be
sound again. They are gone anyway: `scan` has a fixed trip count, and at
0.034s/epoch against a 4.6s compile, early stopping saved less wall clock
than compiling the loop that implemented it.

Selection is **detector level** (`x_sim` reweighted vs `x_data`), so it needs
no truth and the method stays deployable. The particle-level MMD is computed
too, but on the host in `workflows.train`, never in the trace — which is what keeps
`z_true` out of the traced program while still producing the curve. The
number reported for the restored checkpoint comes from a _test_ subsample,
not the val one selection minimized.

The estimator has a resolution floor around 5e-4 in MMD^2 at m=8192, scaling
as ~1/m; below it the ranking inverts, because the empirical MMD is minimized
by weights matching the sample rather than the distribution. `benchmarks/`
measures it. `MMD_SUBSAMPLE` is 16384.

Loss math is plain `jnp` — `keras.ops` bought backend-agnosticism this module no
longer has, since `lax.scan` and `jax.random` are both native.
`stateless_call`/`stateless_apply` are the only Keras calls inside the trace.

**`engine.train(fused=False)` is the debugging path.** It runs the identical `_epoch`
function from an ordinary Python `while` — still one XLA program per epoch, but
with breakpoints, readable tracebacks and host-side control flow. It is also the
reference the fused path is tested against (`tests/test_train.py::TestFusion`),
so the two must never diverge into separate implementations.
