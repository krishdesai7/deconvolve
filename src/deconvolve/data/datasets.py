from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..coretypes import (
    CACHE_DIR,
    EVENT_DTYPE,
    ZXY,
    DatasetSplits,
    Events,
    GaussianConfig,
    Populations,
)
from ..instrumentation import note
from .config import parse_gaussian_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from logging import Logger
    from pathlib import Path
    from typing import Any, Final, LiteralString, SupportsFloat

    from jax import Array
    from jaxtyping import Float
    from numpy.typing import NDArray

    from ..coretypes import Nested

logger: Logger = logging.getLogger(name=__name__)

_ONE_SOURCE_ONLY: Final[LiteralString] = (
    "Exactly one of config_path or params must be provided"
)

# Bumped whenever the generator changes, because the cache key is otherwise a
# pure function of the physics config -- an old .npz would be silently reused
# and hand back a sample drawn from a different stream. `jax-v2` pinned the
# matmul precision in `_draw_gaussian`, which moves every drawn value by ~5e-4
# relative on a GPU and by nothing at all on a CPU.
_RNG_VERSION: Final[LiteralString] = "jax-v2"


def _savez_atomic(path: Path, /, **arrays: NDArray[Any]) -> None:
    """Write an `.npz` that a concurrent reader either misses or sees whole.

    `np.savez` streams into the destination it is handed, so a reader that
    arrives mid-write gets a truncated zip. RAN's cache is shared by
    construction (`submit_uncertainty.zsh` packs a grid of cells against one
    `DECONVOLVE_CACHE_DIR`, and `pytest -n16` does the same on a smaller scale), so
    two workers can race to write the same cache key.

    Writing beside the target and renaming makes publishing one atomic
    `rename(2)`, which POSIX guarantees within a directory. Two writers racing
    is then harmless: each builds its own file, one rename wins, and both
    hold identical bytes because the key is a hash of what produced them.

    The temp name keeps a `.npz` suffix because `np.savez` appends one to any
    path lacking it, which would otherwise leave the real output beside a
    stray. `unlink` runs from a `finally` so a failed write leaves no partial
    file behind; `missing_ok` absorbs the already-renamed case.
    """
    tmp: Path = path.with_name(name=f"{path.name}.{uuid.uuid4().hex}.tmp.npz")
    try:
        #  Checkers read `**arrays` as a candidate for savez's
        # `allow_pickle: bool` keyword; every value here is an array.
        np.savez(file=tmp, **arrays)  # ty: ignore[invalid-argument-type]
        _ = tmp.replace(target=path)
    finally:
        tmp.unlink(missing_ok=True)


def _draw_gaussian(
    seed: int,
    /,
    *,
    mu_true: NDArray[np.double],
    mu_gen: NDArray[np.double],
    cov_true: NDArray[np.double],
    cov_gen: NDArray[np.double],
    cov_detector: NDArray[np.double],
    n_samples: int,
) -> tuple[
    Float[Array, "n d"], Float[Array, "n d"], Float[Array, "n d"], Float[Array, "n d"]
]:
    """Draw the four Gaussian populations: `(z_true, z_gen, x_data, x_sim)`.

    Runs on the default device. Sharing a node is handled by the launchers,
    which give each step exactly one visible GPU via `srun --gpus-per-task=1`,
    so a sibling run cannot have the card swallowed out from under it.

    The draw pins its matmul precision to `HIGHEST`. Two dots produce this
    sample -- the `@` against the Cholesky smear, and one inside
    `multivariate_normal(method="svd")` -- and XLA runs both at TF32 on an A100
    by default, which would make the sample a function of the hardware as well
    as of the config and the seed. The pin makes a `.npz` drawn on a login node
    and one drawn on a GPU node the same sample. The cache key is otherwise a
    pure function of the physics config, so `_RNG_VERSION` carries `jax-v2` to
    keep a pre-pin file from being silently reused.

    No `check_valid` equivalent is needed: `parse_gaussian_config` has already
    asserted positive-definiteness with a Cholesky factorization.
    """
    k_true, k_gen, k_data, k_sim = jax.random.split(jax.random.key(seed), num=4)

    with jax.default_matmul_precision("highest"):
        z_true: Float[Array, "n d"] = jax.random.multivariate_normal(
            k_true, mu_true, cov_true, shape=(n_samples,), method="svd"
        )
        z_gen: Float[Array, "n d"] = jax.random.multivariate_normal(
            k_gen, mu_gen, cov_gen, shape=(n_samples,), method="svd"
        )

        smear: Float[Array, "d d"] = jnp.linalg.cholesky(jnp.asarray(a=cov_detector)).T

        x_data: Float[Array, "n d"] = (
            z_true + jax.random.normal(key=k_data, shape=z_true.shape) @ smear
        )
        x_sim: Float[Array, "n d"] = (
            z_gen + jax.random.normal(key=k_sim, shape=z_gen.shape) @ smear
        )
    return z_true, z_gen, x_data, x_sim


class ArrayDataset:
    """An in-memory `ZXY` with deterministic minibatching.

    One host-resident split of (z, x, y), plus how it should be batched.

    This is a container, not an iterator. Batch order is drawn on device, per
    epoch, by `deconvolve.data.device.train_indices` -- so `batch_size` and
    `seed` are carried here as the split's own parameters and read by
    `DeviceSplits.from_splits`, but nothing iterates this object.

    Attributes:
        data: The split's events and labels. `data.z` and `data.x` reach
            through to the underlying `Events`.
        batch_size: Events per batch.
        seed: Seed for the reshuffling generator.
    """

    def __init__(
        self,
        data: ZXY,
        batch_size: int = 128,
        seed: int = 42,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.data: ZXY = data
        self.batch_size: int = batch_size
        self.seed: int = seed

    @property
    def size(self) -> int:
        return len(self.data)

    @property
    def dtype(self) -> np.dtype[np.single]:
        return self.data.dtype

    def __len__(self) -> int:
        """Number of batches per pass."""
        return (self.size + self.batch_size - 1) // self.batch_size

    def as_arrays(self) -> ZXY:
        """The whole split as flat labelled arrays, in stored order."""
        return self.data


class DeconvolveDataset:
    """Builds the train/val/test splits RAN trains on.

    Attributes:
        dataset: The events in shuffled order, once built.
        splits: The train/val/test splits, once built.
    """

    def __init__(
        self,
        batch_size: int = 128,
        seed: int = 42,
        cache_dir: Path = CACHE_DIR,
        val_fraction: float = 0.1,
        test_fraction: float = 0.2,
    ) -> None:
        self.batch_size: int = batch_size
        self.seed: int = seed
        self.cache_dir: Path = cache_dir
        self.dtype: np.dtype[np.single] = np.dtype(EVENT_DTYPE)

        if test_fraction < 0 or test_fraction > 1:
            raise ValueError("test_fraction must be between 0 and 1")
        if val_fraction < 0 or val_fraction > 1:
            raise ValueError("val_fraction must be between 0 and 1")
        if val_fraction + test_fraction >= 1:
            raise ValueError("val_fraction + test_fraction must be < 1")

        self.val_fraction: float = val_fraction
        self.test_fraction: float = test_fraction
        self.dataset: ZXY | None = None
        self.splits: DatasetSplits | None = None

    @staticmethod
    def _round_nested(
        obj: Nested[SupportsFloat], ndigits: int = 10, /
    ) -> Nested[float]:
        """Recursively round floats in a nested list/scalar for stable hashing."""
        if isinstance(obj, list):
            return [DeconvolveDataset._round_nested(v, ndigits) for v in obj]
        return float(np.round(a=float(obj), decimals=ndigits))

    def _cache_key(self, parsed: GaussianConfig, n_samples: int) -> str:
        """Hash the promoted covariance matrices for a canonical cache key."""
        key_data: dict[str, Nested[float] | str] = {
            "mu_gen": self._round_nested(parsed.mu_gen.tolist()),
            "mu_true": self._round_nested(parsed.mu_true.tolist()),
            "cov_gen": self._round_nested(parsed.cov_gen.tolist()),
            "cov_true": self._round_nested(parsed.cov_true.tolist()),
            "cov_detector": self._round_nested(parsed.cov_detector.tolist()),
            "n_samples": n_samples,
            "seed": self.seed,
            "rng": _RNG_VERSION,
            # Without this a float32 and a float64 run share one file
            "dtype": str(object=self.dtype),
        }
        # Positional, not `data=`: that keyword name only exists from 3.13,
        # and this is the one call in the package that would raise the floor.
        return hashlib.sha256(
            json.dumps(obj=key_data, sort_keys=True).encode(encoding="utf-8")
        ).hexdigest()[:16]

    def _cache_path(self, parsed: GaussianConfig, n_samples: int) -> Path:
        cache_key: str = self._cache_key(parsed, n_samples)
        return self.cache_dir / f"gaussian_{cache_key}.npz"

    def _order(self, data: ZXY) -> NDArray[np.intp]:
        """The shuffle that spreads both classes across every split.

        `interleave` stacks nature (y=1) on MC (y=0); the splits are contiguous
        slices, so they would otherwise be single-class. This shuffle happens
        once and is not repeated per epoch -- it defines the event ordering
        the splits cut into.
        """
        return np.random.default_rng(self.seed).permutation(x=len(data))

    def _split_dataset(
        self, dataset: ZXY, multiplicity: NDArray[np.intp] | None = None
    ) -> DatasetSplits:
        """Cut the shuffled arrays into contiguous train/val/test splits.

        Test is taken off the end, validation off the end of what remains, so
        train occupies the front. Only the training split reshuffles between
        epochs.
        """
        n: int = len(dataset)
        n_test: int = int(n * self.test_fraction)
        n_non_test: int = n - n_test
        val_of_non_test: float = self.val_fraction / (1.0 - self.test_fraction)
        n_val: int = int(n_non_test * val_of_non_test)
        n_train: int = n_non_test - n_val
        if n_train < 1 or n_val < 1 or n_test < 1:
            raise ValueError(
                f"{n} events split into train={n_train}, val={n_val}, test={n_test}; "
                "every split needs at least one event"
            )

        def _slice(lo: int, hi: int) -> ArrayDataset:
            rows: NDArray[np.intp] = np.arange(lo, hi)
            if multiplicity is not None:
                rows = np.repeat(a=rows, repeats=multiplicity[lo:hi])
                if rows.size == 0:
                    raise ValueError(
                        "a split holds no events after applying the multiplicities"
                    )
            return ArrayDataset(
                data=ZXY(Events(dataset.z[rows], dataset.x[rows]), dataset.y[rows]),
                batch_size=self.batch_size,
                seed=self.seed,
            )

        return DatasetSplits(
            train=_slice(lo=0, hi=n_train),
            val=_slice(lo=n_train, hi=n_non_test),
            test=_slice(lo=n_non_test, hi=n),
        )

    def generate_gaussian_dataset(
        self,
        config_path: Path | None = None,
        *,
        params: GaussianConfig | None = None,
        n_samples: int = 10**6,
    ) -> DatasetSplits:
        """Generate (or load from cache) a Gaussian dataset and split it.

        Exactly one of `config_path` or `params` must be provided.

        Args:
            config_path: A Gaussian YAML config file.
            params: An already-parsed config, in place of `config_path`.
            n_samples: Number of samples per class (data and MC).
        """
        # Written as a nested check rather than a single XOR so that each branch
        # narrows the argument it goes on to use.
        if params is not None:
            if config_path is not None:
                raise ValueError(_ONE_SOURCE_ONLY)
            parsed: GaussianConfig = params
        elif config_path is None:
            raise ValueError(_ONE_SOURCE_ONLY)
        else:
            parsed = parse_gaussian_config(config_path)

        # `_draw_gaussian` runs at the float64 the config was parsed in; the
        # sample narrows to EVENT_DTYPE once, on the way out, so the Cholesky
        # smear inside the draw never loses precision to an earlier cast.
        cache_path: Path = self._cache_path(parsed, n_samples)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if cache_path.exists():
            note("gaussian cache hit", to="data")
            logger.info("Loading dataset from cache: %s", cache_path)
            with np.load(file=cache_path) as cached:
                arrays: Mapping[str, NDArray[Any]] = cast(
                    typ="Mapping[str, NDArray[Any]]", val=cached
                )
                data: ZXY = ZXY(
                    Events(
                        z=arrays["z"].astype(dtype=self.dtype),
                        x=arrays["x"].astype(dtype=self.dtype),
                    ),
                    y=arrays["y"],
                )
        else:
            note("gaussian generated", to="data")
            z_true, z_gen, x_data, x_sim = _draw_gaussian(
                self.seed,
                mu_true=parsed.mu_true,
                mu_gen=parsed.mu_gen,
                cov_true=parsed.cov_true,
                cov_gen=parsed.cov_gen,
                cov_detector=parsed.cov_detector,
                n_samples=n_samples,
            )

            data: ZXY = Populations(
                mc=Events(
                    z=np.asarray(a=z_gen, dtype=self.dtype),
                    x=np.asarray(a=x_sim, dtype=self.dtype),
                ),
                data=np.asarray(a=x_data, dtype=self.dtype),
                truth=np.asarray(a=z_true, dtype=self.dtype),
            ).interleave()

            # Uncompressed, these are incompressible floats, so DEFLATE is a large read
            # tax for a few percent of disk.
            _savez_atomic(cache_path, z=data.z, x=data.x, y=data.y)
            logger.info("Generated and saved dataset to cache: %s", cache_path)

        return self.splits_from_data(data)

    def splits_from_data(
        self, data: ZXY, *, multiplicity: NDArray[np.intp] | None = None
    ) -> DatasetSplits:
        """Shuffle one labelled sample and cut it into train/val/test.

        `multiplicity`, one non-negative count per row of `data`, repeats each
        row that many times *after* the cut, so every copy of a row stays in the
        split the row was assigned to. This is how a bootstrap replicate is
        split: cutting an already-resampled sample would put copies of one
        event into different splits.

        `ZXY` has already checked that the particle- and detector-level arrays
        are row-aligned and that every label is zero or one, so this does no
        validation of its own.
        """
        if multiplicity is not None and multiplicity.shape != (len(data),):
            raise ValueError(
                f"multiplicity has shape {multiplicity.shape}; expected one count "
                f"per event, ({len(data)},)"
            )
        order: NDArray[np.intp] = self._order(data)
        self.dataset: ZXY = ZXY(Events(data.z[order], data.x[order]), data.y[order])
        self.splits: DatasetSplits = self._split_dataset(
            self.dataset, None if multiplicity is None else multiplicity[order]
        )
        return self.splits
