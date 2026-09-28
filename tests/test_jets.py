"""Tests for jet loading, and specifically for the column order.

`load_jet_dataset` fills column `i` from the `i`-th requested variable, so the
container carrying those names is an ordering. It used to be a `frozenset`,
whose iteration order depends on per-process randomized string hashes: `deconvolve
train` built columns in one order and recorded it in `config.json`, then `deconvolve
baseline ibu` and `deconvolve evaluate` rebuilt the same dataset in a different order
in their own processes and labelled it with the recorded one. Every jet metric
came back under the wrong variable name, and a generator trained on one column
order was fed another.

Nothing tested this module at all, which is how it shipped.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from deconvolve.coretypes import (
    CACHE_FILENAMES,
    JET_DISPLAY_ORDER,
    JET_VARIABLE_GROUPS,
    SUBSTRUCTURE_VARIABLES,
    Split,
    display_order,
)
from deconvolve.data.jets import load_jet_dataset

if TYPE_CHECKING:
    from pathlib import Path

_N: int = 400

# One recognisable offset per observable. It has to survive standardisation,
# which subtracts the mean of `z_gen` and divides by its standard deviation --- so
# `z_gen` is made identical across observables and the signature is carried by
# `x_data`, where the shared affine transform leaves it distinguishable.
_SIGNATURE: dict[str, float] = {
    name: 1000.0 * (i + 1) for i, name in enumerate(SUBSTRUCTURE_VARIABLES)
}


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """A synthetic stand-in for the Zenodo cache, one npz per observable."""
    ramp = np.arange(_N, dtype=np.double)
    for name, offset in _SIGNATURE.items():
        np.savez(
            tmp_path / f"{CACHE_FILENAMES[name]}.npz",
            z_gen=ramp,
            x_sim=ramp,
            z_true=ramp + offset,
            x_data=ramp + offset,
        )
    return tmp_path


def _column_order(cache: Path, variables: tuple[str, ...]) -> list[str]:
    """Which observable's file actually landed in each column.

    Read off the arrays themselves, not off `std_params` --- that dict is keyed
    by variable name and built in the same loop as the columns, so it agrees
    with the request by construction and could never have caught this.
    """
    splits, dim, _ = load_jet_dataset(
        n_samples=_N, batch_size=32, cache_dir=cache, variables=variables
    )
    assert dim == len(variables)

    pops = splits.select(Split.ALL).partition()
    sigma = float(np.arange(_N, dtype=np.double).std())
    by_offset = {round(v / sigma, 2): k for k, v in _SIGNATURE.items()}

    means = [round(float(pops.data[:, i].mean()), 2) for i in range(len(variables))]
    return [by_offset[m] for m in means]


class TestColumnOrderFollowsTheRequest:
    def test_the_requested_order_is_the_column_order(self, cache: Path) -> None:
        wanted = ("zg", "m", "tau21")

        assert _column_order(cache, wanted) == list(wanted)

    def test_a_different_request_permutes_the_columns(self, cache: Path) -> None:
        """The property the old code could not provide across two processes."""
        forward = ("m", "M", "w")
        reverse = ("w", "M", "m")

        assert _column_order(cache, forward) == list(forward)
        assert _column_order(cache, reverse) == list(reverse)

    def test_a_set_is_refused_rather_than_silently_ordered(self, cache: Path) -> None:
        """The actual bug, now unrepresentable.

        A `frozenset` has an iteration order, so this call used to succeed and
        produce a column order no caller could predict or reproduce.
        """
        with pytest.raises(TypeError, match="ordered sequence, not a set"):
            _ = load_jet_dataset(
                n_samples=_N,
                batch_size=32,
                cache_dir=cache,
                # ty: ignore[invalid-argument-type]
                variables=frozenset(("m", "w")),  # pyrefly: ignore[bad-argument-type]
            )

    def test_duplicates_are_refused(self, cache: Path) -> None:
        """Two columns of identical data under one name would look plausible."""
        with pytest.raises(ValueError, match="duplicates"):
            _ = load_jet_dataset(
                n_samples=_N, batch_size=32, cache_dir=cache, variables=("m", "m")
            )

    def test_an_unknown_name_is_refused_with_the_accepted_list(
        self, cache: Path
    ) -> None:
        with pytest.raises(ValueError, match="unknown jet variables"):
            _ = load_jet_dataset(
                n_samples=_N, batch_size=32, cache_dir=cache, variables=("m", "pt")
            )


def test_the_canonical_order_is_an_ordering_not_a_set() -> None:
    """Pins the container type, because that is what the bug was.

    A set would pass every other test in this file in a single process and fail
    only across two, which is exactly how this went unnoticed.
    """
    assert isinstance(SUBSTRUCTURE_VARIABLES, tuple)
    assert not isinstance(SUBSTRUCTURE_VARIABLES, (set, frozenset))


def test_display_order_permutes_the_twelve_observables() -> None:
    """Physics reading order: mass, angularities, splitting, hadronization."""
    order: tuple[int, ...] = display_order(SUBSTRUCTURE_VARIABLES)
    named: tuple[str, ...] = tuple(SUBSTRUCTURE_VARIABLES[i] for i in order)

    assert named == (
        "m",
        "sdm",
        "lha",
        "w",
        "ang2",
        "zg",
        "tau21",
        "M",
        "n_ch",
        "f_ch",
        "ptd",
        "q",
    )


def test_display_order_filters_a_subset() -> None:
    """`--var w --var m` still presents mass before angularity."""
    assert tuple("wm"[i] for i in display_order(("w", "m"))) == ("m", "w")


def test_display_order_is_the_identity_for_a_gaussian_run() -> None:
    """A Gaussian run has `dim_0 ... dim_n` and no physics ordering."""
    assert display_order(("dim_0", "dim_1", "dim_2")) == (0, 1, 2)


def test_the_display_order_and_the_column_order_hold_the_same_names() -> None:
    """Two tuples that must never drift apart, guarded rather than documented."""
    assert set(JET_DISPLAY_ORDER) == set(SUBSTRUCTURE_VARIABLES)
    assert len(JET_DISPLAY_ORDER) == len(SUBSTRUCTURE_VARIABLES)


def test_the_groups_partition_the_display_order() -> None:
    grouped: list[str] = [v for _, members in JET_VARIABLE_GROUPS for v in members]
    assert tuple(grouped) == JET_DISPLAY_ORDER


def test_the_column_order_is_unchanged() -> None:
    """The cache key. See the Jet Column Order section of CLAUDE.md."""
    assert SUBSTRUCTURE_VARIABLES == (
        "m",
        "M",
        "w",
        "tau21",
        "zg",
        "sdm",
        "q",
        "f_ch",
        "lha",
        "ang2",
        "ptd",
        "n_ch",
    )
