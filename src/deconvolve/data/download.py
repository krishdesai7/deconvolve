r"""One-time download of jet substructure data from Zenodo (record 3548091).

Downloads Pythia26 and Herwig Z+jets Delphes datasets (17 `.npz` files each),
extracts the substructure variables, saves per-variable `.npz` files to
`CACHE_DIR` (`.cache/`, or wherever `DECONVOLVE_CACHE_DIR` points), and deletes
the raw downloads.

## Degenerate jets

Two of the observables are undefined for a small number of jets. For
$\beta = 1$ the jet width is $\tau_1$, so $\tau_{21} = \frac{\tau_2}{\tau_1}$;
a jet of one constituent has neither, both vanish and the ratio is
$\frac00$. This is the case for around 100-300 jets per array. A jet that soft
drop grooms down to a single prong has $m_{\text{sd}} = 0$, so
$\ln \rho = \ln \left( \frac{m_{\text{sd}}^2}{p_T^2} \right) = -\infty$. This is
the case for a few hundred more jets per array.

`_get_var` computes each observable only where it is defined and fills the
rest with a stated value: `_ONE_PRONG_TAU21` and `LOG_RHO_FLOOR`.

The usual alternative is to nudge the denominator or the log argument by an
epsilon, and it is worse in three ways.

1. It hides the convention inside a number that reads like a rounding
   allowance.
2. It depends on the dtype the raw arrays happen to arrive in. For example
   $10^{-50}$ (used in OmniFold) is below the smallest `float32` denormal, so
   if arrays were stored as `float32`, it would round away and hand back `NaN`
   for exactly the jets it was meant to protect.
3. An epsilon scaled to the data, such as $10^{-12} \times \text{mean}(p_T^2)$
   (used in OmniFold), is a *different* epsilon for each of the four arrays,
   which puts the floor of $\ln\rho$ in a different place for nature than for
   MC. Several hundred jets per array sit on that floor and thousands more are
   compressed against it, so the discriminator gets handed a spike whose
   position differs between the classes for reasons that have nothing to do
   with physics. The four arrays are two samples that get compared to each
   other; an observable that means something slightly different in each is
   not a comparison.

The one-prong $\tau_{21}$ is assigned zero, matching OmniFold's published
convention, though it is a convention rather than a measurement -- and not
obviously the right limit, since zero is what a cleanly two-pronged jet
approaches, which a one-constituent jet is not.
"""

from __future__ import annotations

import logging
import urllib.request
from typing import TYPE_CHECKING, cast

import numpy as np
from rich.progress import Progress, TaskID

from ..coretypes import (
    CACHE_DIR,
    CACHE_FILENAMES,
    GENERATORS,
    LOG_RHO_FLOOR,
    N_FILES,
    SUBSTRUCTURE_VARIABLES,
    ZENODO_RECORD,
)

if TYPE_CHECKING:
    from logging import Logger
    from pathlib import Path
    from typing import Final

    from numpy.typing import NDArray

logger: Logger = logging.getLogger(name=__name__)

PID_CHARGE: Final[np.uintc] = np.uintc(0x5228849)


def _download_url(generator: str, file_idx: int) -> str:
    return (
        f"https://zenodo.org/record/{ZENODO_RECORD}/files/"
        f"{generator}_Zjet_pTZ-200GeV_{file_idx}.npz?download=1"
    )


def _download_file(url: str, dest: Path, progress: Progress, task_id: TaskID) -> None:
    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            progress.update(
                task_id,
                total=total_size,
                completed=min(block_num * block_size, total_size),
            )

    # urlretrieve honours file:// and any other scheme urllib knows, so pin it to
    # https before opening. Callers only ever pass _download_url output, which is
    # built from a hardcoded https://zenodo.org prefix; this keeps that true if
    # the helper ever grows another caller.
    if not url.startswith("https://"):
        raise ValueError(f"refusing to fetch non-https URL: {url!r}")
    # ruff: ignore[suspicious-url-open-usage]
    _ = urllib.request.urlretrieve(url, filename=dest, reporthook=_progress)
    logger.info("Downloaded %s", dest)


_ONE_PRONG_TAU21: Final[float] = 0.0


# `PID_CHARGE` packs one 2-bit charge field per particle-type index into a
# 32-bit word, so only indices 0..15 are addressable. A larger index shifts by
# 32 or more, which is undefined rather than merely wrong, and would hand back
# a plausible-looking charge for a particle type this table does not cover.
_MAX_PID_INDEX: Final[int] = 15


def _constituents(
    data: dict[str, NDArray[np.double]], ptype: str, /
) -> tuple[NDArray[np.double], NDArray[np.int8]]:
    """Per-constituent `(pt, charge)` for every jet.

    `particles` is `(jets, constituents, 4)` with columns `(pt, y, phi, pid/10)`
    and the constituent axis zero-padded. Indexing it as `[:, 0]` selects one
    *constituent's* four features rather than every constituent's pt, which is
    a silent axis error whenever a jet happens to have four constituents and a
    broadcast failure otherwise.
    """
    particles: NDArray[np.double] = data[f"{ptype}_particles"]
    if particles.ndim != 3 or particles.shape[2] < 4:
        raise ValueError(
            f"{ptype}_particles has shape {particles.shape}; "
            "expected (jets, constituents, >=4)"
        )
    pt: NDArray[np.double] = particles[:, :, 0].astype(dtype=np.double)
    # Widened before the shift: `ubyte * 2` wraps at 128, and the index feeds a
    # shift distance rather than a value, so a wrap is silent corruption.
    ids: NDArray[np.intp] = np.round(a=particles[:, :, 3] * 10).astype(dtype=np.intp)
    if ids.min() < 0 or ids.max() > _MAX_PID_INDEX:
        raise ValueError(
            f"{ptype}_particles carries pid indices in "
            f"[{ids.min()}, {ids.max()}]; PID_CHARGE only encodes "
            f"[0, {_MAX_PID_INDEX}]"
        )
    charge: NDArray[np.int8] = (((PID_CHARGE >> (ids * 2)) & 3) - 1).astype(
        dtype=np.int8
    )
    return pt, charge


def _safe_ratio(
    numerator: NDArray[np.double], denominator: NDArray[np.double], /
) -> NDArray[np.double]:
    """`numerator / denominator`, zero where the jet carries no pt at all.

    Same `where=` discipline as the tau21 and sdm branches: an empty jet is a
    degenerate input to guard explicitly, not a nan to propagate into the
    standardization statistics of every other event.
    """
    return np.divide(
        numerator,
        denominator,
        out=np.zeros(shape=denominator.shape, dtype=np.double),
        where=denominator > 0.0,
    )


# Derived from the per-constituent `particles` array rather than read from a
# stored per-jet array. Kept as a set so `_get_var` dispatches once instead of
# falling through one branch per observable.
_CONSTITUENT_VARS: Final[frozenset[str]] = frozenset({"q", "f_ch", "n_ch", "ptd"})


def _stored_var(
    data: dict[str, NDArray[np.double]], var: str, ptype: str, /
) -> NDArray[np.double]:
    """Observables the Zenodo release already carries, one value per jet."""
    match var:
        case "m":
            return data[f"{ptype}_jets"][:, 3].astype(dtype=np.double)
        case "M":
            return data[f"{ptype}_mults"].astype(dtype=np.double)
        case "w":
            return data[f"{ptype}_widths"].astype(dtype=np.double)
        case "tau21":
            tau2: NDArray[np.double] = data[f"{ptype}_tau2s"].astype(dtype=np.double)
            width: NDArray[np.double] = data[f"{ptype}_widths"].astype(dtype=np.double)
            return np.divide(
                tau2,
                width,
                out=np.full(shape=width.shape, fill_value=_ONE_PRONG_TAU21),
                where=width > 0,
            )
        case "zg":
            return data[f"{ptype}_zgs"].astype(dtype=np.double)
        case "sdm":
            sdm: NDArray[np.double] = data[f"{ptype}_sdms"].astype(dtype=np.double)
            jet_pt: NDArray[np.double] = data[f"{ptype}_jets"][:, 0].astype(
                dtype=np.double
            )
            rho_sq: NDArray[np.double] = (sdm / jet_pt) ** 2
            return np.log(
                rho_sq,
                out=np.full(shape=rho_sq.shape, fill_value=LOG_RHO_FLOOR),
                where=rho_sq > 0,
            )
        case "lha":
            return data[f"{ptype}_lhas"].astype(dtype=np.double)
        case "ang2":
            return data[f"{ptype}_ang2s"].astype(dtype=np.double)
        case _:
            raise ValueError(f"Unknown variable '{var}'")


def _constituent_var(
    data: dict[str, NDArray[np.double]], var: str, ptype: str, /
) -> NDArray[np.double]:
    """Observables computed from the jet's constituents."""
    pt, charge = _constituents(data, ptype)
    match var:
        case "q":
            # Jet charge, pT^kappa-weighted at kappa = 1/2.
            root_pt: NDArray[np.double] = np.sqrt(pt)
            return _safe_ratio((charge * root_pt).sum(axis=1), root_pt.sum(axis=1))
        case "f_ch":
            return _safe_ratio((np.abs(charge) * pt).sum(axis=1), pt.sum(axis=1))
        case "n_ch":
            # Masked on pt, not on charge alone. The constituent axis is
            # zero-padded to the longest jet in the file, and a count is the
            # one observable here that a padded row could enter -- every other
            # one weights by pt, which is zero on padding.
            populated: NDArray[np.bool] = cast(
                "NDArray[np.bool]", (charge != 0) & (pt > 0.0)
            )
            return np.asarray(a=np.count_nonzero(populated, axis=1), dtype=np.double)
        case "ptd":
            # sqrt(sum pt^2) / sum pt: 1 for a one-particle jet, 1/sqrt(n) for
            # n equal ones. Padding contributes zero to both sums.
            return _safe_ratio(np.sqrt(np.square(pt).sum(axis=1)), pt.sum(axis=1))
        case _:
            raise ValueError(f"Unknown variable '{var}'")


def _get_var(
    data: dict[str, NDArray[np.double]], var: str, ptype: str, /
) -> NDArray[np.double]:
    r"""Extract a substructure variable from raw arrays.

    Two of them are undefined for a jet the detector or the groomer has left
    with nothing to measure: $\tau_{21}$ is $\frac00$ when the jet has one
    constituent, and $\ln\rho = -\infty$ when soft drop leaves no groomed mass.
    Both are handled by computing the observable only where it exists and
    filling the rest with a declared value, `_ONE_PRONG_TAU21` and
    `LOG_RHO_FLOOR`.
    """
    if var in _CONSTITUENT_VARS:
        return _constituent_var(data, var, ptype)
    return _stored_var(data, var, ptype)


def _ensure_shard(dest: Path, gen: str, idx: int, progress: Progress, /) -> None:
    """Download one shard unless it is already cached."""
    if dest.exists():
        logger.info("%s: already downloaded", dest.name)
        return
    task_id: TaskID = progress.add_task(description=dest.name, total=None)
    _download_file(_download_url(generator=gen, file_idx=idx), dest, progress, task_id)
    progress.remove_task(task_id)


_COLUMN_KEYS: Final[tuple[str, ...]] = tuple(
    f"{ptype}_{var}" for var in SUBSTRUCTURE_VARIABLES for ptype in ("gen", "sim")
)


def _shard_observables(path: Path, /) -> dict[str, NDArray[np.double]]:
    """Every observable, both levels, for one downloaded shard."""
    # Materialized once: an `NpzFile` decompresses on every member access, and
    # four observables read `particles`.
    with np.load(file=path) as handle:
        shard: dict[str, NDArray[np.double]] = {
            key: handle[key] for key in handle.files
        }
    return {
        f"{ptype}_{var}": _get_var(shard, var, ptype)
        for var in SUBSTRUCTURE_VARIABLES
        for ptype in ("gen", "sim")
    }


def _fetch_generator(
    gen: str,
    cache_dir: Path,
    progress: Progress,
    all_raw_paths: list[Path],
) -> dict[str, NDArray[np.double]]:
    """One generator's observables, reduced shard by shard.

    Returns `{"<ptype>_<var>": values}` over every event, never the raw arrays.
    Appends each shard path to `all_raw_paths` so the caller can delete the raw
    downloads once the per-variable caches have been written.

    Reducing inside the loop rather than concatenating first makes `particles`
    affordable and correct. Affordable: the constituent arrays are
    the bulk of the release, and holding both generators' at float64 would be
    ~12 GB against ~100 MB of derived observables. Correct: the constituent
    axis is padded to the longest jet *in that array*, which differs between
    shards and even between `gen` and `sim` in one shard (116 against 94 in
    file 0), so a buffer preallocated from the first shard cannot hold the
    rest -- it raises on the first wider one.

    Concatenating the reductions also drops the assumption that every shard
    carries the same number of events.
    """
    logger.info("Downloading %s (%d files)", gen, N_FILES)

    files: list[Path] = [
        cache_dir / f"{gen}_Zjet_pTZ-200GeV_{i}.npz" for i in range(N_FILES)
    ]
    for i, path in enumerate(iterable=files):
        all_raw_paths.append(path)
        _ensure_shard(path, gen, i, progress)

    columns: dict[str, list[NDArray[np.double]]] = {key: [] for key in _COLUMN_KEYS}
    for path in files:
        for key, values in _shard_observables(path).items():
            columns[key].append(values)

    return {key: np.concatenate(parts) for key, parts in columns.items()}


def download_jet_data(cache_dir: Path = CACHE_DIR) -> None:
    """Download Pythia26/Herwig data from Zenodo, extract variables, save to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_data: dict[str, dict[str, NDArray[np.double]]] = {}
    all_raw_paths: list[Path] = []

    with Progress() as progress:
        for gen in GENERATORS:
            raw_data[gen] = _fetch_generator(gen, cache_dir, progress, all_raw_paths)
            n_events: int = len(next(iter(raw_data[gen].values())))
            logger.info("%s: %d events loaded", gen, n_events)

    # Herwig = data (nature), Pythia26 = MC (synthetic)
    nature: dict[str, NDArray[np.double]] = raw_data["Herwig"]
    synthetic: dict[str, NDArray[np.double]] = raw_data["Pythia26"]

    logger.info(msg="Extracting substructure variables...")
    for var in SUBSTRUCTURE_VARIABLES:
        out_path: Path = cache_dir / f"{CACHE_FILENAMES[var]}.npz"
        # `savez`, not `savez_compressed`. These observables are float64 and
        # very nearly incompressible: measured on representative data, DEFLATE
        # lands at ~0.94 of raw for a continuous variable while costing ~20x
        # on read. Integer-valued variables (e.g. `mult`) compress to ~0.18,
        # but the read cost is paid on every run while the write happens once,
        # so uncompressed wins on balance. `np.load` reads either format
        # identically, so this is a write-side choice only.
        np.savez(
            file=out_path,
            z_true=nature[f"gen_{var}"],
            x_data=nature[f"sim_{var}"],
            z_gen=synthetic[f"gen_{var}"],
            x_sim=synthetic[f"sim_{var}"],
        )
        logger.info("Saved %s", out_path)

    # Clean up raw files
    for path in all_raw_paths:
        if path.exists():
            path.unlink()
    logger.info("Cleaned up %d raw files.", len(all_raw_paths))
    logger.info(msg="Jet data cached.")


if __name__ == "__main__":
    download_jet_data()
