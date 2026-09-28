"""The `B x S` design: bootstrap datasets crossed with initialization seeds.

One invocation runs one cell, so a cluster can put every cell on its own GPU
and the whole design costs one training run of wall clock. `cell_of_index`
maps a flat array-task id onto `(dataset, seed)`, and each cell writes a small
`.npz` carrying its weights on the common evaluation set --- nothing else, and
in particular no model, because the design is about the spread of the outputs
rather than any one run's parameters.

Three decisions in here are the ones that make the numbers mean what the
report says they mean:

**The bootstrap resamples events, not the split.** Varying `data_seed`
reshuffles a fixed sample into different train/val/test splits and different
batch orders; every run still sees the same events. That is *method*
variance --- an artifact of the algorithm being order-dependent, removable by
ensembling --- and it is not the statistical uncertainty a measurement is
obliged to report. The nonparametric bootstrap, drawing `n` of `n` with
replacement, estimates the latter: how much the answer would move if
the experiment had collected a different sample of the same size.

**The split varies per cell, independently of both axes.** If it were a
function of the replicate, as it is with `data_seed` fixed, its variance would
be charged to the dataset component, which is the one reported as the
statistical uncertainty. Drawn per cell, it lands in the residual alongside
the interaction, with the rest of the method variance. `split_seed` derives it
from `(data_seed, dataset, seed)`, and it also sets the batch order and the
MMD selection subsamples, which are method variance of the same kind.

**Events are split before they are resampled.** A replicate is a multiplicity
per original event, drawn once per dataset index; each cell splits the
*original* events and then repeats each within its split. Splitting an
already-resampled sample instead puts copies of one event into different
splits --- about half of every validation set would also be training data ---
and the epoch selection that reads the validation set would no longer be the
procedure whose variance is being measured.

**MC and nature resample independently.** They are two separate samples in the
physics --- one generated, one measured --- and coupling their resampling
would impose a correlation that does not exist. The pairing *within* each is
preserved, because `(z_gen, x_sim)` and `(x_data, z_true)` are the same events
seen at two levels.

**Every cell is evaluated on one fixed common set of gen-level events**, held
out before any resampling and therefore absent from every replicate's
training. Bootstrap replicates contain different --- and duplicated --- events,
so their per-event weight vectors are not otherwise commensurable and cannot be
stacked into the matrix the covariance is computed from. Holding the set out
also means the finite size of the evaluation sample shifts every run together
and cancels out of the across-run contrast entirely.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from ..coretypes import (
    SUBSTRUCTURE_VARIABLES,
    DatasetName,
    Events,
    Populations,
    Resample,
    Split,
)
from ..data import DeconvolveDataset, load_jet_dataset, parse_gaussian_config

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from logging import Logger
    from pathlib import Path
    from typing import Any, Final, LiteralString

    from numpy.typing import NDArray

    from ..coretypes import DatasetSplits, EventArray, GaussianConfig
    from ..training import TrainResult

logger: Logger = logging.getLogger(name=__name__)

CELL_GLOB: str = "cell_*.npz"
FROZEN_NAME: Final[LiteralString] = "design.json"


class DesignSpec(NamedTuple):
    """The shape of the grid, and the seeds every cell derives its own from."""

    n_datasets: int
    n_seeds: int
    data_seed: int = 42
    init_seed: int = 0

    @property
    def n_cells(self) -> int:
        return self.n_datasets * self.n_seeds

    def cell_of_index(self, index: int, /) -> tuple[int, int]:
        """Flat task id -> `(dataset, seed)`, seed-major within a dataset.

        Seed-major so that a design cut short after `k * n_seeds` cells is a
        complete grid over the datasets that finished, rather than a ragged
        one no decomposition will accept.
        """
        if not 0 <= index < self.n_cells:
            raise IndexError(
                f"cell {index} is outside a {self.n_datasets}x{self.n_seeds} "
                f"design ({self.n_cells} cells)"
            )
        return divmod(index, self.n_seeds)


class EvaluationSet(NamedTuple):
    """What is left to train on, and the gen-level events every run is read on."""

    pool: Populations
    z: EventArray


def reserve_evaluation_set(
    pops: Populations, /, *, n_eval: int, seed: int | Sequence[int]
) -> EvaluationSet:
    """Hold out `n_eval` MC gen-level events, before any resampling touches them.

    Only the MC side is reserved: `g` is evaluated on `z_gen` and never on a
    nature event, so there is nothing to hold out on that side and no reason
    to spend the statistics.
    """
    n_mc: int = len(pops.mc)
    if not 0 < n_eval < n_mc:
        raise ValueError(
            f"n_eval must be between 1 and {n_mc - 1} (the MC events available), "
            f"got {n_eval}"
        )
    order: NDArray[np.intp] = np.random.default_rng(seed).permutation(x=n_mc)
    held: NDArray[np.intp] = order[:n_eval]
    kept: NDArray[np.intp] = order[n_eval:]
    return EvaluationSet(
        pool=Populations(
            mc=Events(z=pops.mc.z[kept], x=pops.mc.x[kept]),
            data=pops.data,
            truth=pops.truth,
        ),
        z=pops.mc.z[held],
    )


def bootstrap_multiplicities(
    pops: Populations,
    /,
    *,
    seed: int | Sequence[int],
    resample: Resample = Resample.both,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """How often each original event is drawn: `n` of `n` with replacement.

    Returned as `(mc, nature)` counts rather than as resampled arrays, so a
    cell can split the original events first and expand each split afterwards
    (`replicate_splits`). Both samples keep their original size, so a replicate
    is the same measurement repeated rather than a smaller one. Both sides are
    always drawn, so the stream a side sees does not depend on `resample`; a
    side that is not resampled is then replaced by ones.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    n_mc: int = len(pops.mc)
    n_nature: int = pops.data.shape[0]
    mc: NDArray[np.intp] = np.bincount(
        rng.integers(low=0, high=n_mc, size=n_mc), minlength=n_mc
    )
    nature: NDArray[np.intp] = np.bincount(
        rng.integers(low=0, high=n_nature, size=n_nature), minlength=n_nature
    )
    if resample is Resample.data:
        mc = np.ones(shape=n_mc, dtype=np.intp)
    elif resample is Resample.mc:
        nature = np.ones(shape=n_nature, dtype=np.intp)
    return mc, nature


def bootstrap(
    pops: Populations,
    /,
    *,
    seed: int | Sequence[int],
    resample: Resample = Resample.both,
) -> Populations:
    """One bootstrap replicate as arrays: each event repeated as often as drawn."""
    mc, nature = bootstrap_multiplicities(pops, seed=seed, resample=resample)
    return Populations(
        mc=Events(
            z=np.repeat(pops.mc.z, mc, axis=0), x=np.repeat(pops.mc.x, mc, axis=0)
        ),
        data=np.repeat(pops.data, nature, axis=0),
        truth=np.repeat(pops.truth, nature, axis=0),
    )


def replicate_splits(
    pool: Populations,
    multiplicities: tuple[NDArray[np.intp], NDArray[np.intp]],
    /,
    *,
    split_seed: int,
    batch_size: int,
) -> DatasetSplits:
    """Split the original events, then repeat each within its own split.

    Every copy of an event therefore lands in the same split, as in an
    ordinary run, where train, validation and test share no events.
    """
    mc, nature = multiplicities
    # `interleave` puts the nature rows first, then the MC rows.
    return DeconvolveDataset(batch_size=batch_size, seed=split_seed).splits_from_data(
        pool.interleave(), multiplicity=np.concatenate([nature, mc])
    )


def split_seed(data_seed: int, dataset_index: int, seed_index: int, /) -> int:
    """The split, batch order and selection subsamples of one cell.

    A separate stream from the replicate's `(data_seed, dataset_index)`, and
    different in every cell, so their variance cannot be charged to either
    axis of the design.
    """
    sequence = np.random.SeedSequence(entropy=(data_seed, dataset_index, seed_index))
    return int(sequence.generate_state(n_words=1)[0])


def base_populations(
    dataset: DatasetName,
    /,
    *,
    n_samples: int,
    batch_size: int,
    data_seed: int,
    variables: tuple[str, ...] = SUBSTRUCTURE_VARIABLES,
    params: GaussianConfig | None = None,
) -> tuple[Populations, int]:
    """The undisturbed sample the design bootstraps, plus its dimensionality.

    Built through the ordinary loaders and immediately flattened back out of
    the splits: the design does its own splitting per replicate, so the split
    boundaries here would only be re-drawn.

    The Gaussian branch takes already-parsed parameters rather than a YAML
    path, so `collect` can rebuild the identical sample months later from what
    the cells recorded, without the file still having to be on disk and
    unchanged.
    """
    if dataset == DatasetName.jets:
        splits: DatasetSplits = load_jet_dataset(
            n_samples=n_samples,
            batch_size=batch_size,
            variables=variables,
            seed=data_seed,
        )[0]
        return splits.select(Split.ALL).partition(), len(variables)
    if dataset == DatasetName.gaussian:
        if params is None:
            raise ValueError("Gaussian mode requires --config path/to/config.yaml")
        splits = DeconvolveDataset(batch_size, data_seed).generate_gaussian_dataset(
            params=params, n_samples=n_samples
        )
        return splits.select(Split.ALL).partition(), params.dim
    raise ValueError(f"Unknown dataset: {dataset!r}")


def cell_path(design_dir: Path, index: int, /) -> Path:
    return design_dir / f"cell_{index:04d}.npz"


def run_cell(
    index: int,
    design_dir: Path,
    spec: DesignSpec,
    /,
    *,
    dataset: DatasetName = DatasetName.jets,
    variables: tuple[str, ...] = SUBSTRUCTURE_VARIABLES,
    config: Path | None = None,
    n_samples: int = 500_000,
    n_eval: int = 100_000,
    batch_size: int = 1024,
    hidden_units: int = 64,
    n_layers: int = 2,
    n_epochs: int = 100,
    n_disc_steps: int = 5,
    lr_g: float = 3e-5,
    lr_d: float = 1e-4,
    lambda_dispersion: float = 0.015,
    resample: Resample = Resample.both,
) -> Path:
    """Train one `(dataset, seed)` cell and record its weights on the common set."""
    # Deferred so that `deconvolve uncertainty collect`, which only reads npz and
    # reports, does not pay for importing keras and jax.
    from ..evaluation.evaluate import _get_weights
    from ..training.engine import train

    b, s = spec.cell_of_index(index)
    params: GaussianConfig | None = (
        parse_gaussian_config(config_path=config)
        if dataset == DatasetName.gaussian and config is not None
        else None
    )
    pops, dim = base_populations(
        dataset,
        n_samples=n_samples,
        batch_size=batch_size,
        data_seed=spec.data_seed,
        variables=variables,
        params=params,
    )
    # Seeded off `spec.data_seed` alone, with no `b`: every cell has to reserve
    # the *same* events or the weight vectors are not stackable.
    evaluation: EvaluationSet = reserve_evaluation_set(
        pops, n_eval=n_eval, seed=spec.data_seed
    )
    # Two ints rather than an arithmetic combination, so no two replicates can
    # collide on one stream however the base seed is chosen. Drawn from `b`
    # alone, so every cell of a row trains on the same replicate.
    multiplicities: tuple[NDArray[np.intp], NDArray[np.intp]] = (
        bootstrap_multiplicities(
            evaluation.pool, seed=(spec.data_seed, b), resample=resample
        )
    )
    cell_split_seed: int = split_seed(spec.data_seed, b, s)
    splits: DatasetSplits = replicate_splits(
        evaluation.pool,
        multiplicities,
        split_seed=cell_split_seed,
        batch_size=batch_size,
    )
    result: TrainResult = train(
        splits,
        dim,
        hidden_units,
        n_layers,
        seed=spec.init_seed + s,
        n_epochs=n_epochs,
        n_disc_steps=n_disc_steps,
        lr_g=lr_g,
        lr_d=lr_d,
        lambda_dispersion=lambda_dispersion,
    )
    weights: EventArray = _get_weights(result.g, z_gen=evaluation.z)

    design_dir.mkdir(parents=True, exist_ok=True)
    out: Path = cell_path(design_dir, index)
    np.savez(
        file=out,
        weights=weights,
        meta=np.array(
            object=json.dumps(
                obj={
                    "index": index,
                    "dataset_index": b,
                    "seed_index": s,
                    "init_seed": result.seed,
                    "split_seed": cell_split_seed,
                    "data_seed": spec.data_seed,
                    "resample": resample.value,
                    "n_eval": n_eval,
                    "n_samples": n_samples,
                    "batch_size": batch_size,
                    "dim": dim,
                    "dataset": dataset.value,
                    "variables": list(variables),
                    "gaussian_params": params.model_dump() if params else None,
                    "hidden_units": hidden_units,
                    "n_layers": n_layers,
                    "n_epochs": n_epochs,
                    "n_disc_steps": n_disc_steps,
                    "lr_g": lr_g,
                    "lr_d": lr_d,
                    "lambda_dispersion": lambda_dispersion,
                    "mmd_test": result.mmd_test,
                }
            )
        ),
    )
    logger.info(
        "cell %d (dataset %d, seed %d): test MMD^2 %.3e -> %s",
        index,
        b,
        s,
        result.mmd_test,
        out,
    )
    return out


class Design(NamedTuple):
    """A loaded grid: `(B, S, n_eval)` weights plus what rebuilds its inputs.

    `meta` is cell zero's record with the per-cell fields dropped, because the
    rest of it --- dataset, variables, sample size, seeds --- is by
    construction identical across the grid, and `collect` uses it to
    regenerate the common evaluation set without storing a copy of it in
    every cell.
    """

    weights: NDArray[np.double]
    spec: DesignSpec
    meta: dict[str, Any]


_PER_CELL_KEYS: frozenset[str] = frozenset(
    ("index", "dataset_index", "seed_index", "init_seed", "split_seed", "mmd_test")
)

# The settings a sanctioned `--flag` override (`_resolve_cell_settings` in
# `cli.py`) can legitimately change on one cell without touching the rest.
# That override has to leave a trace: if a hand-rerun of one failed cell used
# different settings than the rest of the array, the grid is not one
# measurement, and `load_cells` must refuse it rather than average it in as
# if it were noise.
_SHARED_SETTINGS_KEYS: frozenset[str] = frozenset(
    (
        "n_epochs",
        "n_layers",
        "hidden_units",
        "n_disc_steps",
        "lr_g",
        "lr_d",
        "lambda_dispersion",
        "resample",
    )
)


def _check_settings_agree(metas: Sequence[Mapping[str, Any]], /) -> None:
    """Refuse a grid whose cells disagree on a setting that must be shared."""
    for key in sorted(_SHARED_SETTINGS_KEYS):
        values: list[Any] = [meta.get(key) for meta in metas]
        baseline: Any = values[0]
        disagreeing: list[int] = [i for i, v in enumerate(values) if v != baseline]
        if disagreeing:
            raise ValueError(
                f"cells disagree on `{key}`: cell 0 has {baseline!r}, but "
                f"cell(s) {disagreeing} do not -- a design's cells must all "
                f"train under the same settings"
            )


def _read_cell(path: Path, /) -> tuple[NDArray[np.double], dict[str, Any]]:
    with np.load(file=path) as cell:
        arrays: Mapping[str, NDArray[Any]] = cast(
            typ="Mapping[str, NDArray[Any]]", val=cell
        )
        return (
            np.asarray(a=arrays["weights"], dtype=np.double),
            json.loads(s=str(object=arrays["meta"].item())),
        )


def _missing_cells(design_dir: Path, spec: DesignSpec, /) -> list[int]:
    return [
        index
        for index in range(spec.n_cells)
        if not cell_path(design_dir, index).exists()
    ]


def load_cells(design_dir: Path, spec: DesignSpec, /) -> Design:
    """Stack the finished cells into a `(B, S, n_eval)` grid, or say what is missing.

    The decomposition needs a *balanced* grid --- the mean squares assume one
    run per cell --- so a partial design is refused with the list of gaps
    rather than silently decomposed over whatever landed, which would charge
    the imbalance to the dataset axis.
    """
    missing: list[int] = _missing_cells(design_dir, spec)
    if missing:
        raise FileNotFoundError(
            f"{design_dir} is missing cells {missing}; the decomposition needs "
            f"all {spec.n_cells} of a {spec.n_datasets}x{spec.n_seeds} grid"
        )

    first, meta = _read_cell(cell_path(design_dir, 0))
    grid: NDArray[np.double] = np.empty(
        shape=(spec.n_datasets, spec.n_seeds, first.size)
    )
    metas: list[dict[str, Any]] = [meta]
    b0, s0 = spec.cell_of_index(0)
    grid[b0, s0] = first
    for index in range(1, spec.n_cells):
        weights: NDArray[np.double]
        cell_meta: dict[str, Any]
        weights, cell_meta = _read_cell(cell_path(design_dir, index))
        if weights.size != first.size:
            raise ValueError(
                f"cell {index} holds {weights.size} weights but cell 0 holds "
                f"{first.size}; these cells are not from one design"
            )
        metas.append(cell_meta)
        b, s = spec.cell_of_index(index)
        grid[b, s] = weights
    _check_settings_agree(metas)
    return Design(
        weights=grid,
        spec=spec,
        meta={k: v for k, v in meta.items() if k not in _PER_CELL_KEYS},
    )


def freeze_design(
    design_dir: Path,
    values: dict[str, Any],
    origins: dict[str, str],
    *,
    force: bool = False,
) -> Path:
    """Write the settings every cell of this design will use.

    Refuses to overwrite: a design whose cells were trained under different
    settings is not a variance decomposition, and rewriting this file while an
    array is in flight is exactly how that happens. The no-`force` path opens
    the file with the `x` mode rather than checking `.exists()` first, so the
    refusal is atomic and not a check the array could race past.
    """
    design_dir.mkdir(parents=True, exist_ok=True)
    path: Path = design_dir / FROZEN_NAME
    payload: str = json.dumps(obj={"config": values, "_origin": origins}, indent=2)
    if not force:
        try:
            with path.open(mode="x", encoding="utf-8") as handle:
                _ = handle.write(payload)
        except FileExistsError as error:
            raise FileExistsError(
                f"{path} already exists; pass --force to overwrite it, but not "
                f"while an array is running"
            ) from error
        return path

    if any(design_dir.glob(CELL_GLOB)):
        raise FileExistsError(
            f"{design_dir} already has cell files matching {CELL_GLOB!r}; "
            f"--force would let a design overwrite its own settings while its "
            f"array is already running"
        )
    _ = path.write_text(data=payload)
    return path


def load_frozen(design_dir: Path) -> dict[str, Any]:
    """The frozen settings for this design."""
    path: Path = design_dir / FROZEN_NAME
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `deconvolve uncertainty freeze "
            f"{design_dir}` once before submitting the array"
        )
    frozen: dict[str, Any] = json.loads(s=path.read_text())
    config: dict[str, Any] = frozen["config"]
    return config
