# RAN: Reweighting Adversarial Networks

`ran` is a library for training and evaluating reweighting adversarial networks (RANs).

Importing anything under `ran` first pins the Keras 3 backend to JAX and disables JAX's 64-bit mode. Both settings are read once, when `jax`/`keras` are first imported, so they must be in place before any submodule imports either.

`ran` is float32 end to end. The pin is `EVENT_DTYPE` in :mod:`deconvolve.coretypes.constants`, with its annotation twin `EventArray` in :mod:`deconvolve.coretypes.types`; `JAX_ENABLE_X64=0` and the `dtype=` arguments in :mod:`deconvolve.training.models` follow from it.

`setdefault` throughout, so that the environment can be explicitly overridden.

Import the submodule needed (`from deconvolve.workflows.train import run`); the CLI re-exports below are the sole exception, and they defer their own imports into the command bodies.

Only `cli.py` sits beside `__init__.py` and `__main__.py`; everything else lives in a subpackage: `config/` (layered CLI configuration: discovery and merge, the command spec, `config show`), and by pipeline stage `training/` (models, the fused loop, MMD), `evaluation/` (metrics and plots), `workflows/` (the orchestration behind `deconvolve train` and `deconvolve leakage-check`), `reporting/` (the LaTeX dossier and its template) and `instrumentation/` (timing and logging setup). Dependencies point one way: `workflows` imports `training`, `evaluation` and `baselines`, and none of those import `workflows`. Keep it that way — `workflows/train.py` and `workflows/leakage.py` are the orchestration layer precisely because they need both `training` and `evaluation`; putting either inside `training/` or `evaluation/` instead would make one of those packages import the other's sibling, closing a cycle. `workflows/__init__.py` re-exports `run` and `run_leakage_check` so `from deconvolve.workflows import run` works without knowing which submodule it lives in.

## module `evaluation.evaluate`

Computes distance metrics on test sets for completed runs: per-dimension 1D Wasserstein distances, Jensen-Shannon divergences and triangular discriminators, both before and after reweighting.

**Every one of them runs on device.** The `jnp` implementation is vectorized across dimensions, so one dispatch computes every column, and only the per-dimension reduced scalars cross back to the host. Measured at 100k-vs-100k in 6D: ~0.28s of compute plus a one-time XLA compile.

Scores stay float64. Only the reductions over the full sample happen in float32, and each is arranged so its error is relative to the answer rather than to the largest intermediate.

Usage:

```bash
    deconvolve evaluate                          # all runs in runs/
    deconvolve evaluate runs/2026-...  # single run
    deconvolve evaluate --force                  # recompute existing
```

### `apply_to_runs(run_dir: Path, evaluate_one: Callable[[Path], object], description: str, log: Logger) -> None`

Apply `evaluate_one` to a single run directory, or to every run inside one.

A directory is a run if it holds a config.json; otherwise it is treated as a parent directory of runs. In the multi-run case one failure is logged and skipped rather than abandoning the remaining runs.

**Arguments:**

- `run_dir: Path` The directory to evaluate.
- `evaluate_one: Callable[[Path], object]` A function to apply to each run directory.
- `description: str` A description of the evaluation to log.
- `log: Logger` A logger to use for logging.

**Returns:**
None

### `_load_splits(config: dict) -> DatasetSplits`

Load the dataset splits from the config.

**Arguments:**

- `config: dict` The config to load the dataset splits from.

**Returns:**

- `DatasetSplits` The loaded dataset splits.

### `_bin_edges(ref: EventArray | JaxArray, comp: EventArray | JaxArray, n_bins: int) -> NDArray[np.single]`

`(dim, n_bins + 1)` uniform edges spanning each dimension's combined range, built on the host with `np.linspace` so both histograms bin against exactly the same numbers.

They are **float32 on purpose**: `JAX_ENABLE_X64=0` truncates a float64 array on its way into a traced function, so float64 edges would be re-rounded at the boundary and the bin a value lands in would stop matching the edges the host computed.

### `_counts(x: JaxArray, edges: JaxArray, weights: JaxArray) -> JaxArray`

Weighted bin counts per column, `(dim, n_bins)`. `searchsorted(..., "right") - 1` reproduces `np.histogram`'s placement against explicit edges, and the clip is its closed last bin, where the maxima land.

The weights are **centered before they are scattered** and the mean added back through the exact count. Scattering them raw sums ~200 magnitudes per bin in float32, while centering leaves the scatter summing residuals an order of magnitude smaller, and the integer count is exact in float32 out to 2\*\*24. The mean carries its own error and does not matter: it multiplies every bin of a column by the same factor, which `_normalize` divides straight back out.

### `_cdf_gap_integral(ref: JaxArray, comp: JaxArray, weights: JaxArray) -> JaxArray`

$\int |F_{ref} - F_{comp}| \, dt$ per column: the 1D Wasserstein-1 distance, vectorized over columns so one dispatch does every dimension.

The two CDFs are **never accumulated separately**. Each climbs to 1 while their difference stays at the order of the distance being measured, so subtracting them afterwards cancels away most of a float32 mantissa. Cumulatively summing the signed weights instead keeps the running value at the size of the answer, which makes the float32 error relative to it rather than to 1, and costs one scan instead of two.

### `_wd_per_dim(ref: EventArray, comp: EventArray, weights: EventArray | JaxArray | None = None, n_bins: int = 100) -> NDArray[np.double]`

1D Wasserstein distance per dimension. `weights` reweights `comp` only, and is normalized, so scaling all of them by a constant cannot change a distance.

### `_normalized_histograms(ref: EventArray, comp: EventArray, weights: EventArray | JaxArray | None = None, n_bins: int = 100) -> tuple[NDArray[np.double], NDArray[np.double]]`

Returns two $(dimensions, n_bins)$ arrays containing the $(p, q)$ probability histograms for each dimension of `ref`/`comp`. Both histograms share one binning per dimension, `n_bins` uniform bins over the combined range, so the divergence metrics are comparable across dimensions. `weights` reweights `comp` only, and an all-zero histogram is left unnormalized rather than divided by zero.

**Arguments:**

- `ref: EventArray` The reference data.
- `comp: EventArray` The comparison data.
- `weights: EventArray | JaxArray | None` The weights to use for the comparison data.
- `n_bins: int` The number of bins to use for the histograms.

**Returns:**

- `tuple[NDArray[np.double], NDArray[np.double]]` The $(p, q)$ probability histograms for each dimension of `ref`/`comp`

### `_js_per_dim(ref: EventArray, comp: EventArray, weights: EventArray | JaxArray | None = None, n_bins: int = 100) -> NDArray[np.double]`

Returns JS divergence (squared JS distance) per dimension.

### `_triangular_per_dim(ref: EventArray, comp: EventArray, weights: EventArray | JaxArray | None = None, n_bins: int = 100) -> NDArray[np.double]`

Triangular discriminator (Vincze-LeCam divergence) per dimension.

$$\Delta(p,q) = \sum_{i=1}^{n} \frac{(p_i - q_i)^2}{p_i + q_i} \times 10^3$$

where $p_i, q_i$ are histogram probability masses. The bin-width factor cancels analytically, so this works directly on normalized histograms.

### `class MetricSet(NamedTuple)`

Every metric `metrics.json` records -- `wasserstein`, `jensenshannon`, `triangular` -- one entry per dimension. The field order is the order the keys are written in.

### `_metrics_per_dim(ref: EventArray, comp: EventArray, weights: EventArray | JaxArray | None = None, n_bins: int = 100) -> MetricSet`

All three metrics in one device pass, and the path `evaluate_run` takes.

The two divergences read the **same** pair of histograms, since `_js_per_dim` and `_triangular_per_dim` would each otherwise build their own -- two identical scatters over the full sample, for two reductions over `dim x n_bins` values. The single-metric helpers stay for the callers that want one number: `leakage`, the IBU baseline, and the benchmarks.

### `evaluate_run(run_dir: Path = RUN_DIR, force: bool = False) -> None`

Compute distance metrics for completed runs.

**Arguments:**

- `run_dir: Path` Path to a single run or a directory containing multiple runs.
- `force: bool` Recompute even if metrics.json already exists.

**Returns:**

- `None`

## module `evaluation.plotting`

### `plot_detector_level(test_dataset: ArrayDataset, g: keras.Model, save_path: Path = Path("plots/detector_level.pdf"), var_info: list[VarInfo] | None = None, ibu_weights: list[NDArray[np.double]] | None = None) -> None:`

Generate detector level plots.

**Arguments:**

- `test_dataset: ArrayDataset` The test dataset.
- `g: keras.Model` The generator model.
- `save_path: Path` The path to save the plot to.
- `var_info: list[VarInfo] | None` The per-variable plot config.
- `ibu_weights: list[NDArray[np.double]] | None` The per-variable list of per-event IBU weights for MC events.

**Returns:**

- `None`

### `plot_particle_level(test_dataset: ArrayDataset, g: keras.Model, save_path: Path = Path("plots/particle_level.pdf"), var_info: list[VarInfo] | None = None, ibu_weights: list[NDArray[np.double]] | None = None) -> None:`

Generate particle level plots.

**Arguments:**

- `test_dataset: ArrayDataset` The test dataset.
- `g: keras.Model` The generator model.
- `save_path: Path` The path to save the plot to.
- `var_info: list[VarInfo] | None` The per-variable plot config.
- `ibu_weights: list[NDArray[np.double]] | None` The per-variable list of per-event IBU weights for MC events.

**Returns:**

- `None`

## :mod:`deconvolve.training.engine`

Adversarial training loop for RAN, on Keras 3 with the JAX backend, as a single fused XLA program.

The min-max game needs two optimizers driven at different cadences against a shared loss, which does not fit `Model.fit`, so this module implements a hand-rolled loop. It follows the standard Keras 3 + JAX pattern: model state lives in JAX pytrees (:class:`TrainState`) for the duration of training, updates go through `stateless_call`/`stateless_apply`, and each step is a single jitted function. Values are written back into the Keras models at the end so the returned objects are ordinary, saveable `keras.Model`s.

The training loop is not a Python loop over batches: the dataset is moved to device once (:mod:`deconvolve.data.device`), one epoch is a `lax.scan` over grouped batch indices, and the epoch loop is a `lax.scan` with a fixed trip count, so a whole run compiles to one program and the batch gathers fuse into the first `Dense`. Every epoch's parameters are retained so a checkpoint can be selected on the host once the scan is done, by detector-level MMD against a validation subsample.

The loss math is plain `jnp`. `stateless_call`/`stateless_apply` are the only Keras calls inside the trace. `lax.scan` and `jax.random` are both native JAX.

:func:`train(fused=False)` runs the very same epoch function from an ordinary Python `while`. It is still one XLA program per epoch, but it keeps breakpoints,
readable tracebacks and host-side logging, which can be helpful for debugging when a run goes wrong.

### :data:`_HISTORY_KEYS`

Every column is the weighted BCE on the same scale (`_make_pass` negates `g_loss` back before recording it, so `train_g` is the BCE at the generator's batch rather than the objective g descends). Validation measures the BCE in exactly one place: `eval_step` runs once per epoch and both networks are scored by that number.

### :class:`TrainResult(NamedTuple)`

All mutable training state, as a JAX pytree. Returns package of a model training, unpacked as `(g, d, history, seed)`.

Held outside the `keras.Model`s so jitted steps stay pure and no host/device sync happens between steps.

**Fields:**

- :attr:`TrainResult.g: DeconvolveModel` The generator model.
- :attr:`TrainResult.d: DeconvolveModel` The discriminator model.
- :attr:`TrainResult.history: dict[str, list[float]]` The training history. Carries `train_d`, `train_g`, `val_d` (the three `lax.scan` columns) plus `val_mmd` and `val_ess` (host additions computed from the retained per-epoch parameters, once the scan is done).
- :attr:`TrainResult.seed: int` The random seed used for training.
- :attr:`TrainResult.best_epoch: int` Which epoch's parameters were restored: the argmin of `history["val_mmd"]`.
- :attr:`TrainResult.params: EpochParams` Every epoch's parameters, stacked on a leading epoch axis -- what makes host-side selection possible at all.
- :attr:`TrainResult.mmd_test: float` The weighted MMD at `best_epoch`, recomputed on a held-out test subsample rather than read off the validation curve selection minimized.
- :attr:`TrainResult.sigmas: tuple[float, ...]` The RBF bandwidths `bandwidths()` chose from the validation subsample, reused for the test-side cache so both numbers share one kernel.

### :class: `RunCarry(NamedTuple)`

What crosses an epoch boundary in the `lax.scan` over epochs. Everything else
-- the per-epoch `(train_d, train_g, val_d)` row and the full `EpochParams`
-- is a `scan` output, not carried state.

**Fields:**

- :attr:`RunCarry.state: TrainState` The current training state.
- :attr:`RunCarry.key: PRNGKeyArray` The random key, split once per epoch.

### :func:`normalize_weights(Float[Array | NDArray, " n"], Real[Array | NDArray, " n"], Float[Array | NDArray, " n"]) -> Float[Array, " n"]`

Per-batch weights: fixed at 1 for nature, renormalized to count for MC.

`mask` is 1 for a real event and 0 for a padding row, and it enters every sum so a padded eval batch gives exactly the value the unpadded one would. On the training path the mask is all ones and this is the plain form.

`raw_w` is the raw generator output for every event in the batch. Data events (y=1) are pinned to weight 1; MC events (y=0) are rescaled so their weights sum to the MC event count, preserving the per-class normalization.

The y=1 entries of `raw_w` are multiplied by (1 - y) = 0 in both the sum and the result, so `g`'s output on data rows, which are `z_true`, never reaches the loss or its gradient.

**Arguments:**

- `raw_w: Float[Array | NDArray, " n"]` The raw generator output for every event in the batch, already squeezed to one dimension.
- `y: Real[Array | NDArray, " n"]` The target labels for every event in the batch. `Real` rather than `Float` because the pipeline carries them as `uint8` while the tests pass floats.
- `mask: Float[Array | NDArray, " n"]` The mask for every event in the batch.

**Returns:**

- `Float[Array, " n"]` The per-batch normalized weights.

### :func:`bce_sums(Float[Array | NDArray, " n"], Real[Array | NDArray, " n"], Float[Array | NDArray, " n"], Float[Array | NDArray, " n"]) -> tuple[Float[Array, ""], Float[Array, ""]]`

Masked weighted BCE, unnormalized, paired with the count it divides by, so a scan can accumulate across batches and divide once with `jnp.sum(...) / n` rather than a mean.

**Arguments:**

- `d_out: Float[Array | NDArray, " n"]` The discriminator output for every event in the batch, already squeezed to one dimension.
- `y: Real[Array | NDArray, " n"]` The target labels for every event in the batch.
- `w: Float[Array | NDArray, " n"]` The weights for every event in the batch.
- `mask: Float[Array | NDArray, " n"]` The mask for every event in the batch.

**Returns:**

- `tuple[Float[Array, ""], Float[Array, ""]]` The unnormalized weighted BCE and the count it divides by.

### :func:`weighted_bce(d_out, y, w)`

Weighted binary cross-entropy, reduced with `jnp.sum(...) / n` rather than a mean -- `keras.ops.mean` is not used anywhere in this module, since for float64 input it picks a float32 compute dtype internally and returns a result carrying ~1e-8 relative error, which would silently undo the project's precision policy (see the top-level README's Precision section). `ops.sum` is unaffected.

**Arguments:**

- `d_out: Float[Array | np.ndarray, " n"]` The discriminator output for every event in the batch, already squeezed to one dimension.
- `y: Real[Array | np.ndarray, " n"]` The target labels for every event in the batch.
- `w: Float[Array | np.ndarray, " n"]` The weights for every event in the batch.

**Returns:**

- `Float[Array, ""]` The scalar loss; the shape annotation catches a dropped reduction.

### :func:`_make_steps(DeconvolveModel, DeconvolveModel, StatelessOptimizer, StatelessOptimizer) -> tuple[TrainStep, TrainStep, EvalStep]`

Build the jitted disc/gen/eval steps, closing over the models.

The models are captured rather than passed so jit sees only array arguments; each returned function is traced once per input shape.

**Arguments:**

- `g: DeconvolveModel` The generator model.
- `d: DeconvolveModel` The discriminator model.
- `opt_g: StatelessOptimizer` The generator optimizer.
- `opt_d: StatelessOptimizer` The discriminator optimizer.

**Returns:**

- `tuple[JitWrapped, JitWrapped, JitWrapped]` The jitted discriminator, generator, and evaluation steps.

### def `_run_epoch(state: TrainState, train_ds: ArrayDataset, disc_step: JitWrapped, gen_step: JitWrapped, n_disc_steps: int) -> tuple[TrainState, float, float]`

One pass over the training split, returning the new state and mean losses.

`d` updates every batch and `g` every `n_disc_steps`-th batch, i.e. the usual adversarial cadence, giving the discriminator a head start each round. The generator loss is negated back to d's sign convention so the two curves stay directly comparable in the history. Losses are reduced to plain floats so every history series has one element type (`np.mean` would give `np.floating`).

**Arguments:**

- `state: TrainState` The current training state.
- `train_ds: ArrayDataset` The training dataset.
- `disc_step: JitWrapped` The jitted discriminator step.
- `gen_step: JitWrapped` The jitted generator step.
- `n_disc_steps: int` The number of discriminator steps per generator step.

**Returns:**

- `tuple[TrainState, float, float]` The new state and mean losses.

### def `train(splits: DatasetSplits, dim: int, hidden_units: int, n_layers: int, seed: int | None, n_epochs: int = 100, n_disc_steps: int = 5, lr_g: float = 1e-4, lr_d: float = 1e-4, *, fused: bool = True) -> TrainResult`

Train the generator and discriminator, then select a checkpoint.

This seeds weight initialization _only_. The train/val/test split and the
per-epoch batch order come from the dataset's own seed (`DeconvolveDataset`), which draws from an independent generator. Varying `seed` across runs therefore estimates training/initialization variance at fixed data, i.e., the usual HEP model-uncertainty ensemble, while varying the dataset seed instead would fold in split variance.

The networks are Dense-only with no dropout or batch norm and Adam is deterministic, so the two seeds together fully determine a run (up to non-deterministic GPU reductions).

There is no `patience` or `min_delta`: `n_epochs` is a fixed `lax.scan` trip
count, and every epoch's parameters are retained so selection can happen once
the scan is done, reading a detector-level MMD curve rather than tracking a
running best inside the trace.

**Arguments:**

- `splits: DatasetSplits` The dataset splits.
- `dim: int` The dimension of the data.
- `hidden_units: int` The number of hidden units in the networks.
- `n_layers: int` The number of layers in the networks.
- `seed: int | None` The random weight-initialization seed to use for training. `None` draws one from system entropy. Either way the value used is returned, so a run stays reproducible after the fact without having to decide up front that it is worth reproducing.
- `n_epochs: int` The number of epochs to train for.
- `n_disc_steps: int` The number of discriminator steps per generator step.
- `lr_g: float` The learning rate for the generator.
- `lr_d: float` The learning rate for the discriminator.
- `fused: bool` Whether to run the whole epoch loop as one `lax.scan` (the default) or drive the identical per-epoch function from a Python `while` for debugging; the two must agree bit-for-bit.

**Returns:**

- `TrainResult` The training result.

## module `workflows.leakage`

Quick leakage check: poison z_true and verify training is unaffected.

Sets z_true to a silly value in one arm and compares against a clean arm; if any
network can see z_true, the two diverge.

Both arms must use the same `init_seed` or the comparison is meaningless: with
random initialization the run-to-run spread swamps the effect being tested.

## module `workflows.train`

### def `_prepare_gaussian(config: Path | None, saved_config: GaussianConfig | None, batch_size: int, n_samples: int, data_seed: int) -> tuple[DatasetSplits, int, GaussianConfig]`

Build Gaussian splits from a reloaded run's config, or from a YAML file. Returns the splits, the dimensionality, and the parsed Gaussian params. The last is so a fresh run can record them in its own config.json.

**Arguments:**

- `config: Path | None` The path to the configuration file to use for training.
- `saved_config: GaussianConfig | None` The saved Gaussian configuration.
- `batch_size: int` The batch size to use for training.
- `n_samples: int` The number of samples to use for training.
- `data_seed: int` The data seed to use for training.

**Returns:**

- `tuple[DatasetSplits, int, GaussianConfig]` The splits, the dimensionality, and the parsed Gaussian params.

### `def _save_run(g: keras.Model, d: keras.Model, history: dict[str, list[float]], *, batch_size: int, n_samples: int, dim: int, dataset: str, init_seed: int, data_seed: int, gaussian_params: GaussianConfig | None, variables: tuple[str, ...]) -> Path`

Write models, history and config to a fresh timestamped run directory.

Gaussian params are stored as covariance matrices so runs are self-contained and reloadable without the original YAML. `init_seed` is the resolved weight-init seed, never None, so a run drawn from entropy is still reproducible via --seed after the fact.

**Arguments:**

- `g: keras.Model` The generator model.
- `d: keras.Model` The discriminator model.
- `history: dict[str, list[float]]` The training history.
- `batch_size: int` The batch size to use for training.
- `n_samples: int` The number of samples to use for training.
- `dim: int` The dimension of the data.
- `dataset: str` The dataset to use for training.
- `init_seed: int` The initial seed to use for training.
- `data_seed: int` The data seed to use for training.
- `gaussian_params: GaussianConfig | None` The Gaussian configuration.
- `variables: tuple[str, ...]` The variables to use for training, in column order.

**Returns:**

- `Path` The path to the run directory.

### `def run(batch_size: int, n_samples: int, config: Path | None, dataset: DatasetName, variables: tuple[str, ...], load_run: Path | None, hidden_units: int, n_layers: int, seed: int | None, data_seed: int) -> None`

Main entry point.

**Arguments:**

- `batch_size: int` The batch size to use for training.
- `n_samples: int` The number of samples to use for training.
- `config: Path | None` The path to the configuration file to use for training.
- `dataset: DatasetName` The dataset to use for training.
- `variables: tuple[str, ...]` The variables to use for training, in column order.
- `load_run: Path | None` The path to the run to load.
- `hidden_units: int` The number of hidden units in the networks.
- `n_layers: int` The number of layers in the networks.
- `seed: int | None` The random weight-initialization seed to use for training. `None` draws one from system entropy. Either way the value used is returned, so a run stays reproducible after the fact without having to decide up front that it is worth reproducing.
- `data_seed: int` The data seed to use for training.

**Returns:**

- `None`

### :func:`_use_compilation_cache() -> None`

Point XLA's persistent cache at :data:`COMPILE_CACHE_DIR`.

Compilation is the largest single time cost in a short run: `benchmarks/boundary.py` on an A100 measures 4.60s of compile time against 0.034s per epoch, so a 100-epoch run spends half its wall clock in XLA. The cache keys on lowered HLO rather than on Python identity, so the fresh `jax.jit(lambda ...)` in :func:`_run` hits it regardless. It lives on disk, so an ensemble of N interpreters compiling the same architecture pays the cost once instead of N times.

It sets `min_compile_time_secs` to zero rather than leaving JAX's default of 1.0s, which would leave RAN's cache entirely empty: the run compiles a few dozen executables that total 4.6s and no single one of them clears a second.

Whatever the caller configured wins, so `JAX_COMPILATION_CACHE_DIR`, or a `jax.config.update` before :func:`train` still overrides this, and an unwritable directory costs a warning from JAX rather than the run.

The path is resolved before it is handed over: JAX opens the cache once and keeps the string, so the default's leading `.` would follow any later `chdir` and turn every write into a `FileNotFoundError` -- which JAX also reports as a warning rather than an error, so the run would go on quietly recompiling. Resolving pins it to the directory the datasets came from.
