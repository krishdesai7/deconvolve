# Data

## Config

Configuration files for Gaussian datasets are YAML files with the following structure:

```yaml
mu_gen: float |list[float]
mu_true: float | list[float]
sigma_gen: float | list[float] | list[list[float]]
sigma_true: float | list[float] | list[list[float]]
sigma_detector: float | list[float] | list[list[float]]
```

### `deconvolve.data.config::sigma_to_covariance`

Promote `sigma` (scalar, vector, or matrix) to a (dim, dim) covariance matrix, where `dim` is the dimension of the data.

- scalar -> σ²I
- (dim,) vector -> diag(σ²)
- (dim, dim) matrix -> used as-is

Validates positive-definiteness via Cholesky decomposition.

### `deconvolve.data.config::parse_gaussian_config`

Parse a Gaussian YAML config file and return a dictionary with the following keys:

- `dim: int` - The dimension of the data.
- `mu_gen: NDArray[np.double]` The mean of the generated data.
- `mu_true: NDArray[np.double]` The mean of the true data.
- `cov_gen: NDArray[np.double]` The covariance matrix of the generated data.
- `cov_true: NDArray[np.double]` The covariance matrix of the true data.
- `cov_detector: NDArray[np.double]` The covariance matrix of the detector.

### `deconvolve.data.config::gaussian_config_from_run_config`

Rebuild a `GaussianConfig` from the `gaussian_params` block of a run's `config.json`, across every format runs/ has ever held - two deprecated formats and one current format. Two share their key names:

- `cov_gen` covariance matrices, written since the type refactor.
- `sigma_gen` (list) covariance matrices under the old name -- master's
  `__main__` stored `cov_gen` as `sigma_gen`.
- `sigma_gen` (scalar) a raw sigma, from before that, needing promotion.

Routing the `sigma_*` spelling through `sigma_to_covariance` resolves the ambiguity without guessing: a raw scalar promotes to σ²I, a vector to diag(σ²), and an already-formed matrix passes through unchanged (checked for shape, symmetry and positive-definiteness on the way). So both readings land on the same covariance, and the `cov_*` spelling needs no promotion at all.

## Two representations

The same events are described two ways, at opposite ends of the pipeline.

`Populations` is the physics form: `mc` (an `Events` pair of generated particle level `mc.z` and simulated detector level `mc.x`, aligned per event), `data` (the measurement), and `truth` (the particle-level answer key). Sources produce it, and analysis consumes it. `truth` sits outside `mc` on purpose, so a function handed the simulation cannot reach the one array no network may see.

`ZXY` is the transport form: an `Events` pair plus a per-event label, `y = 1` for nature and `y = 0` for MC. It gets shuffled, split, batched and trained on.

Every dataset here is a closure test, so `truth` is always known; a real measurement is the case where it is not. `Populations.create(mc, data)` covers that by filling `truth` with `TRUTH_SENTINEL`, and `has_truth` distinguishes the two. The stand-in is a number (-2^15) and not NaN because `interleave` puts `truth` into the nature rows of `z`, which the generator forward-passes: `normalize_weights` discards those rows by multiplying by `1 - y = 0`, and that annihilates a number but not a NaN. Metrics computed against a sentinel `truth` are finite and meaningless, so the particle-level comparisons read the answer key through `require_truth()`, which returns it or refuses. `has_truth` is the same question without the exception.

`Populations.interleave()` converts to the transport form (nature rows first, then MC) and `ZXY.partition()` converts back. Only the second composition is lossless: interleaving a partitioned sample discards the shuffled row order. Weight vectors are indexed against a `Populations`, never against a `ZXY`, so nothing should round-trip.

`DatasetSplits.select(Split.TRAIN | Split.VAL)` concatenates the requested splits into one `ZXY`. The split an event came from is a property of the query, not of the event, so it is not recorded on the result.

## Datasets

### Function `_draw_gaussian`

Draw the four Gaussian populations: `(z_true, z_gen, x_data, x_sim)`.

Runs on the default device. Sharing a node is handled by the launchers, which give each step exactly one visible GPU via `srun --gpus-per-task=1`, so a sibling run cannot have the card swallowed out from under it.

The draw pins its matmul precision to `HIGHEST`. Two dots produce this sample — the `@` against the Cholesky smear, and one inside `multivariate_normal(method="svd")` — and XLA runs both at TF32 on an A100 by default, which would make the sample a function of the hardware as well as of the config and the seed. The pin makes a `.npz` drawn on a login node and one drawn on a GPU node the same sample. The cache key is otherwise a pure function of the physics config, so `_RNG_VERSION` carries `jax-v2` to keep a pre-pin file from being silently reused.

No `check_valid` equivalent is needed: `parse_gaussian_config` has already asserted positive-definiteness with a Cholesky factorization.

**Arguments**:

- `seed: int` Seed for the random number generator.
- `mu_true: NDArray[np.double]` The mean of the true data.
- `mu_gen: NDArray[np.double]` The mean of the generated data.
- `cov_true: NDArray[np.double]` The covariance matrix of the true data.
- `cov_gen: NDArray[np.double]` The covariance matrix of the generated data.
- `cov_detector: NDArray[np.double]` The covariance matrix of the detector.
- `n_samples: int` Number of samples to draw.

**Returns**:

- `tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double], NDArray[np.double]]` The four Gaussian populations.

### `class ArrayDataset`

An in-memory `ZXY` with deterministic minibatching. One host-resident split of (z, x, y), plus how it should be batched.

This is a container, not an iterator. Batch order is drawn on device, per epoch, by `deconvolve.data.device.train_indices` --- so `batch_size` and `seed` are carried here as the split's own parameters and read by
`DeviceSplits.from_splits`, but nothing iterates this object.

Every split holds a view onto one shared pair of base arrays; slicing is done with indexing at batch time, so splitting costs no extra memory.

#### Fields

- `data: ZXY` The split's events and labels. `data.z` and `data.x` reach through to the underlying `Events`.
- `batch_size: int` Events per batch.
- `seed: int` Seed for the reshuffling generator.

#### Properties

- `size: int` Number of events in the dataset.
- `dtype: np.dtype[np.single]` The dtype of the data.

#### Methods

- `len() -> int` Number of batches per pass.
- `as_arrays() -> ZXY` Return the whole split as flat labelled arrays, in stored order.

### `class DeconvolveDataset`

Dataset class for RAN.

#### Fields

- `batch_size: int` Events per batch.
- `seed: int` Random seed.
- `cache_dir: str | Path` Cache directory.
- `val_fraction: float` Validation fraction.
- `test_fraction: float` Test fraction.

#### Properties

- `dataset: ZXY` The events in shuffled order.
- `splits: DatasetSplits` Dataset splits.

#### Methods

##### `_build_dataset`

**Arguments**:

- `data: ZXY` The events as `interleave` produced them.

**Returns**:

- `ZXY` The same events in shuffled order.

**Description**:

Interleave the nature and MC halves with one fixed-seed permutation.

`interleave` stacks nature (y=1) on MC (y=0); the splits below are contiguous slices, so they would otherwise be single-class. This shuffle happens once and is not repeated per epoch -- it defines the event ordering the splits cut into.

##### `_split_dataset`

**Arguments**:

- `dataset: ZXY` The events in shuffled order.

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Cut the shuffled arrays into contiguous train/val/test splits.

Test is taken off the end, validation off the end of what remains, so train occupies the front. Only the training split reshuffles between epochs.

##### `generate_gaussian_dataset`

**Arguments**:

- `config_path: Path | None = None` Path to a YAML config file.
- `params: dict | None = None` Dict with keys mu_gen, mu_true, sigma_gen, sigma_true, sigma_detector.
- `n_samples: int = 10**6` Number of samples per class (data and MC).

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Parse the config file or inline params, generate the dataset, and split it into train/val/test splits. Exactly one of config_path or params must be provided.

##### `splits_from_data`

**Arguments**:

- `data: ZXY` One labelled in-memory sample, usually straight from `Populations.interleave()`.

**Returns**:

- `DatasetSplits` Dataset splits.

**Description**:

Shuffle one labelled sample and cut it into train/val/test.

`ZXY` has already checked that the particle- and detector-level arrays are row-aligned and that every label is zero or one, so this does no validation of its own.

## module `device`

Device-resident training data, the third form alongside `Populations`/`ZXY`.

`Populations` is the physics form and `ZXY` the transport form; both are host NumPy, because they feed Matplotlib, npz I/O and the IBU baseline. This module is the training form: `DeviceSplits.from_splits` is the single host-to-device transfer of a run, and after it no batch crosses the boundary again.

All three forms live under `deconvolve.data` because they are all the dataset, just at different points in its trip to the accelerator. Both splits are laid out for a single fused XLA program: the train split stays flat and is gathered by index inside a `lax.scan`, so XLA fuses the gather into the first `Dense`; the eval splits are pre-batched with a mask, so evaluation scans with no gather at all and still sees every event exactly once.

Evaluation is forward-only and its batching is not part of the training contract, so it uses a wider batch than training to keep the scan short.

### dataclass`TrainSplit`

A flat, device-resident split, gathered by index inside the scan. Labels are promoted from `np.ubyte` to the compute dtype once, here, rather than in every `1 - y` inside the trace.

#### Fields

- `z: Float[Array, "n d"]` The particle-level data.
- `x: Float[Array, "n d"]` The detector-level data.
- `y: Float[Array, "n"]` The labels.

#### Properties

- `n: int` The number of events in the split.

#### class method `from_zxy`

**Arguments**:

- `data: ZXY` The events as `interleave` produced them.

**Returns**:

- `TrainSplit` The train split.

### dataclass`EvalSplit`

A pre-batched device-resident split; padding rows carry `mask == 0`.

#### Fields

- `z: Float[Array, "nb bs d"]` The particle-level data.
- `x: Float[Array, "nb bs d"]` The detector-level data.
- `y: Float[Array, "nb bs"]` The labels.
- `mask: Float[Array, "nb bs"]` The mask.

#### Properties

- `n_batches: int` The number of batches in the split.

#### class method `from_zxy`

**Arguments**:

- `data: ZXY` The events as `interleave` produced them.
- `batch_size: int` Events per batch.

**Returns**:

- `EvalSplit` The eval split class instance.

### `class DeviceSplits(NamedTuple)`

The whole dataset on device, plus the seed that drives batch order.

#### Fields

- `train: TrainSplit` The train split.
- `val: EvalSplit` The validation split.
- `test: EvalSplit` The test split.
- `data_seed: int` The seed that drives batch order.

#### class method `from_splits`

Move a host `DatasetSplits` to device. The one H2D transfer.

`batch_size` is accepted for symmetry but unused: the train split stays flat, and its batching is decided per epoch by `train_indices`.

**Arguments**:

- `splits: DatasetSplits` The dataset splits.
- `batch_size: int | None = None` Events per batch.
- `eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE` Evaluation batch size.

**Returns**:

- `DeviceSplits` The device splits class instance.

### function `grouping`

Split one pass over `n` events into `(groups, disc steps per group)`.

`n_disc_steps` is clamped to the number of whole batches available. A split too small to fill one group still trains --- it becomes a single group with every batch in it, and one generator update.

**Arguments**:

- `n: int` The number of events.
- `batch_size: int` Events per batch.
- `n_disc_steps: int` Number of discriminator steps per group.

**Returns**:

- `tuple[int, int]` The number of groups and the number of discriminator steps per group.

### function `train_indices`

One epoch's batch order, grouped so the update rhythm is structural.

Each group is `n_disc_steps` discriminator batches; the generator updates once per group, on the group's first batch. The tail that does not fill a whole group is dropped. Because the permutation is redrawn every epoch, it is a different random tail each pass.

**Arguments**:

- `key: PRNGKeyArray` The random key.
- `n: int` The number of events.
- `batch_size: int` Events per batch.
- `n_disc_steps: int` Number of discriminator steps per group.

**Returns**:

- `Int[Array, "groups disc batch"]` The batch indices.

### function `gather`

Pull one batch out of the flat split, fused into the first matrix multiplication.

**Arguments**:

- `split: TrainSplit` The train split.
- `idx: Int[Array, "b"]` The batch indices.

**Returns**:

- `tuple[Float[Array, "b d"], Float[Array, "b d"], Float[Array, "b"]]` The particle-level data, the detector-level data and the labels.

## Download

One-time download of jet substructure data from Zenodo (record 3548091).

Downloads Pythia26 and Herwig Z+jets Delphes datasets (17 .npz files each), extracts 6 substructure variables, saves per-variable .npz files to `CACHE_DIR` (`.cache/`, or wherever `DECONVOLVE_CACHE_DIR` points), and deletes the raw downloads

### Degenerate jets

Two of the six observables are undefined for a small number of jets. A jet of one constituent has no width, and for $\beta = 1$ the width is $\tau_1$, so $\tau_{21} = \frac{\tau_2}{\tau_1} = \frac00$. This is the case for around $100-300$ jets per array. A jet that soft drop grooms down to a single prong has $m_{\text{sd}} = 0$, so $\ln \rho = \ln \left( \frac{m_{\text{sd}}^2}{p_T^2} \right) = -\infty$. This is the case for a few hundred more jets per array.

`_get_var` computes each observable only where it is defined and fills the rest with a stated value: `_ONE_PRONG_TAU21` (zero, matching <span style="font-variant: small-caps;">OmniFold</span>) and `LOG_RHO_FLOOR` ($-14$, the bottom of the range `JET_OBS` plots $\ln\rho$ over).

The usual alternative is to nudge the denominator or the log argument by an epsilon, and it is worse in three ways.

1. It hides the convention inside a number that reads like a rounding allowance.
2. It depends on the dtype the raw arrays happen to arrive in. For example $10^{-50}$ (used in <span style="font-variant: small-caps;">OmniFold</span>) is below the smallest `float32` denormal, so if arrays were stored as `float32`, it would round away and hand back `NaN` for exactly the jets it was meant to protect.
3. An epsilon scaled to the data, such as $10^{-12} \times \text{mean}(p_T^2)$ (used in <span style="font-variant: small-caps;">OmniFold</span>), is a _different_ epsilon for each of the four arrays, which puts the floor of $\ln\rho$ in a different place for nature than for MC. Several hundred jets per array sit on that floor and thousands more are compressed against it, so the discriminator gets handed a spike whose position differs between the classes for reasons that have nothing to do with physics. The four arrays are two samples that get compared to each other; an observable that means something slightly different in each is not a comparison.

For $\beta = 1$ the jet width is $\tau_1$, so $\tau_{21} = \frac{\tau_2}{\tau_1}$. A jet of one constituent has neither: both vanish and the ratio is 0/0. This code assigns zero, matching <span style="font-variant: small-caps;">OmniFold</span>'s published convention, though it is a convention rather than a measurement -- and not obviously the right limit, since zero is what a cleanly two-pronged jet approaches, which a one-constituent jet is not.

### `deconvolve.data.download::_get_var`

Extract a substructure variable from raw arrays.

Two of the six are undefined for a jet the detector or the groomer has left with nothing to measure: $\tau_{21}$ is $\frac00$ when the jet has one constituent, and $\ln\rho = -\infty$ when soft drop leaves no groomed mass. Both are handled by computing the observable only where it exists and filling the rest with a declared value, `_ONE_PRONG_TAU21` and `LOG_RHO_FLOOR`.

**Arguments**:

- `data: dict[str, NDArray]` Raw arrays.
- `var: str` Substructure variable.
- `ptype: str` Particle type.

**Returns**:

- `NDArray[np.double]` Substructure variable array.

### `deconvolve.data.download::_fetch_generator`

Fetch every shard for one generator and concatenate the keys needed. Appends each shard path to `all_raw_paths` so the caller can delete the raw downloads once the per-variable caches have been written.

**Arguments**:

- `gen: str` Generator.
- `cache_dir: Path` Cache directory.
- `progress: Progress` Progress object.
- `all_raw_paths: list[Path]` List of raw paths.

**Returns**:

- `dict[str, NDArray]` Raw arrays.

## Jets

Load jet substructure data for RAN training.

Checks `CACHE_DIR` (`.cache/`, or wherever `DECONVOLVE_CACHE_DIR` points) for per-variable `.npz` files. If missing, invokes `download_jet_data` to fetch from Zenodo. Loads, subsamples, z-score standardizes (using MC gen-level statistics only), and builds the train/val/test splits via `DeconvolveDataset`.

### `deconvolve.data.jets::load_jet_dataset`

Load jet substructure data and return DatasetSplits.

Each selected substructure variable is z-score standardized using the MC gen-level (`z_gen`) mean and std. The same parameters are applied to all four arrays (`z_true`, `x_data`, `z_gen`, `x_sim`) to avoid information leakage and preserve correlations.

**Arguments**:

- `n_samples: int = 500_000` Number of events to use per class (data and MC).
- `batch_size: int = 1024` Batch size for the returned splits.
- `cache_dir: Path = CACHE_DIR` Directory containing per-variable `.npz` files. Defaults to `.cache`, relocatable with `DECONVOLVE_CACHE_DIR`.
- `variables: Sequence[str] = SUBSTRUCTURE_VARIABLES` Which substructure variables to use, **in column order**.

  The order is load-bearing, not cosmetic: column `i` is filled from `variables[i]`, `_save_run` records that order in `config.json`, and a later `deconvolve evaluate` or `deconvolve baseline ibu` must reproduce it to label the columns — or to hand a trained generator its own features. A `set` or `frozenset` is refused outright, because its iteration order depends on per-process randomized string hashes and so cannot survive into the second process. Duplicate and unknown names are refused too.

- `seed: int = 42` Dataset seed, controlling the shuffle, the train/val/test split and the per-epoch batch order. Independent of the weight-init seed passed to `train`.
  There is no `dtype` argument. The npz caches on disk are the float64 the Zenodo release ships; the narrowing to `EVENT_DTYPE` happens once, here, on the way into the pipeline, so the standardization statistics are computed on the float32 columns.

Narrowing _after_ the observables are computed is deliberate, not incidental: `deconvolve.data.download._get_var` upcasts to float64 first, because the ε it uses to protect degenerate jets is below the smallest float32 denormal. Narrow before that and it rounds to zero, handing back `NaN` for exactly the jets the ε exists to protect.

**Returns**:

- `tuple[DatasetSplits, int, dict[str, tuple[T, T]]]` DatasetSplits, feature dimensionality, and standardization parameters {var_name: (mu, sigma)}.
