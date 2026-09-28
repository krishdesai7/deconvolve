"""Gaussian dataset configuration.

Configuration files for Gaussian datasets are YAML files with the following
structure:

```yaml
mu_gen: float | list[float]
mu_true: float | list[float]
sigma_gen: float | list[float] | list[list[float]]
sigma_true: float | list[float] | list[list[float]]
sigma_detector: float | list[float] | list[list[float]]
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml

from ..coretypes import REQUIRED_KEYS, GaussianConfig

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    from numpy.typing import ArrayLike, NDArray


def _check_dim(dim: int, /) -> None:
    """A covariance is square and non-degenerate, so there is no 0-dimensional one.

    numpy will not catch this on its own: `np.identity(-1)` raises, but
    `np.identity(0)` and `np.diag([])` both hand back a legal `(0, 0)` array.
    Negative is a numpy error; degenerate is only a physics one.
    """
    if dim < 1:
        raise ValueError(f"dim must be at least 1, got {dim}")


def _scalar_covariance(arr: NDArray[np.double], dim: int) -> NDArray[np.double]:
    """Cov = σ²·I  from a single sigma, however it is spelled.

    `0.5`, `[0.5]` and `[[0.5]]` are the same single sigma, so all three land
    here and all three are squared. Dispatching on `size` rather than `ndim`
    catches the last one too.
    """
    val: np.double = arr.ravel()[0]
    if val <= 0:
        raise ValueError(f"sigma scalar must be positive, got {val}")
    return val**2 * np.identity(n=dim, dtype=np.double)


def _diagonal_covariance(arr: NDArray[np.double], dim: int) -> NDArray[np.double]:
    """Cov = diag(σ²) from a per-dimension σ vector."""
    if arr.shape[0] != dim:
        raise ValueError(f"sigma vector has length {arr.shape[0]}, expected {dim = }")
    if np.any(a=arr <= 0):
        raise ValueError("sigma vector elements must be positive")
    return np.diag(v=arr**2).astype(np.double)


def _full_covariance(arr: NDArray[np.double], dim: int, /) -> NDArray[np.double]:
    """An already-formed covariance matrix, checked for shape, symmetry and PD.

    The other two branches build their matrix from strictly positive sigmas, so
    positive definiteness is theirs by construction; a full matrix is the only
    one that can arrive without it. The factorization is called for its side
    effect -- `LinAlgError` on a non-PD matrix.

    This is also the `cov_*` reader for `gaussian_config_from_run_config`, which
    calls it directly rather than through `sigma_to_covariance`. That matters:
    what arrives here is an already-squared covariance, so it is passed through
    untouched, where the same numbers reaching `sigma_to_covariance` would be
    sigmas and would be squared.
    """
    _check_dim(dim)
    if arr.shape != (dim, dim):
        raise ValueError(f"sigma matrix has shape {arr.shape}, expected {dim = }")
    if not np.allclose(a=arr, b=arr.T):
        raise ValueError("sigma matrix must be symmetric")
    _ = np.linalg.cholesky(arr)
    return arr


def sigma_to_covariance(
    sigma: ArrayLike,
    dim: int,
    /,
) -> NDArray[np.double]:
    """Promote a sigma in any of its three spellings to a covariance matrix.

    A single element is one sigma at any nesting depth (`σ²·I`), a 1-D array of
    `dim` of them is per-dimension (`diag(σ²)`), and a `dim x dim` array is an
    already-formed covariance. An empty sigma needs no branch of its own: it is
    caught by the length check on the vector branch, or the shape check on the
    matrix branch, once `dim` is known to be at least 1.
    """
    _check_dim(dim)
    arr: NDArray[np.double] = np.asarray(a=sigma, dtype=np.double)
    if arr.size == 1:
        return _scalar_covariance(arr, dim)
    if arr.ndim == 1:
        return _diagonal_covariance(arr, dim)
    if arr.ndim == 2:
        return _full_covariance(arr, dim)
    raise ValueError(
        f"sigma must be a single scalar, 1D vector, or 2D matrix, got {arr.ndim = }"
    )


def gaussian_config_from_run_config(
    params: Mapping[str, Any], dim: int
) -> GaussianConfig:
    """Rebuild a `GaussianConfig` from a run's `config.json` `gaussian_params`.

    Accepts every format `runs/` has ever held -- two deprecated formats and
    one current format. Two share their key names:

    - `cov_gen`: covariance matrices, written since the type refactor.
    - `sigma_gen` (2-D): covariance matrices under the old name -- master's
      `__main__` stored `cov_gen` as `sigma_gen`.
    - `sigma_gen` (scalar or vector): a raw sigma, from before that, needing
      promotion.

    A 2-D `sigma_*` is taken as an already-formed matrix (checked for shape,
    symmetry and positive-definiteness); anything else goes through
    `sigma_to_covariance`, so a raw scalar promotes to σ²I and a vector to
    diag(σ²). The `cov_*` spelling needs no promotion at all.
    """
    missing: set[str] = {"mu_gen", "mu_true"} - params.keys()
    if missing:
        raise ValueError(f"gaussian_params missing required keys: {missing}")

    def _covariance(name: str, /) -> NDArray[np.double]:
        if f"cov_{name}" in params:
            return _full_covariance(
                np.asarray(a=params[f"cov_{name}"], dtype=np.double), dim
            )
        if f"sigma_{name}" in params:
            raw: NDArray[np.double] = np.asarray(
                a=params[f"sigma_{name}"], dtype=np.double
            )
            # master's `__main__` wrote covariances under the `sigma_*` names,
            # and 2-D-ness is the only thing that tells the two apart. That rule
            # lives here rather than in `sigma_to_covariance`, where a single
            # element is now a sigma however deeply nested -- without this, a
            # reloaded 1D master-era run would have its `[[0.81]]` squared a
            # second time and quietly regenerate a different dataset.
            if raw.ndim == 2:
                return _full_covariance(raw, dim)
            return sigma_to_covariance(raw, dim)
        raise ValueError(f"gaussian_params has neither cov_{name} nor sigma_{name}")

    return GaussianConfig(
        dim,
        mu_gen=np.asarray(a=params["mu_gen"], dtype=np.double).ravel(),
        mu_true=np.asarray(a=params["mu_true"], dtype=np.double).ravel(),
        cov_gen=_covariance("gen"),
        cov_true=_covariance("true"),
        cov_detector=_covariance("detector"),
    )


def parse_gaussian_config(config_path: Path) -> GaussianConfig:
    """Parse a Gaussian YAML config file, promoting each sigma to a covariance."""
    with config_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(stream=f)

    missing: frozenset[str] = REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

    mu_gen: NDArray[np.double] = np.asarray(a=raw["mu_gen"], dtype=np.double).ravel()
    mu_true: NDArray[np.double] = np.asarray(a=raw["mu_true"], dtype=np.double).ravel()

    dim: int = mu_gen.shape[0]
    if mu_true.shape[0] != dim:
        raise ValueError(f"mu_true has dim {mu_true.shape[0]}, mu_gen has {dim=}")

    cov_gen: NDArray[np.double] = sigma_to_covariance(raw["sigma_gen"], dim)
    cov_true: NDArray[np.double] = sigma_to_covariance(raw["sigma_true"], dim)
    cov_detector: NDArray[np.double] = sigma_to_covariance(raw["sigma_detector"], dim)

    return GaussianConfig(
        dim,
        mu_gen,
        mu_true,
        cov_gen,
        cov_true,
        cov_detector,
    )
