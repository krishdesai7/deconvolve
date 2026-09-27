from __future__ import annotations

from dataclasses import dataclass
from enum import Flag, auto
from typing import TYPE_CHECKING, NamedTuple, cast

import numpy as np

from .constants import EVENT_DTYPE, TRUTH_SENTINEL

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from numpy.typing import NDArray

    from ..data import ArrayDataset
    from .types import EventArray


class Split(Flag):
    """Which of the train/val/test splits to draw events from."""

    TRAIN = auto()
    VAL = auto()
    TEST = auto()
    FIT = TRAIN | VAL
    ALL = FIT | TEST


@dataclass(frozen=True, eq=False, slots=True)
class Events:
    z: EventArray
    x: EventArray

    def __post_init__(self) -> None:
        if self.z.shape[0] != self.x.shape[0]:
            raise ValueError(
                f"z has {self.z.shape[0]} rows and x has {self.x.shape[0]}; "
                "particle and detector level arrays must be row-aligned"
            )

    @property
    def dtype(self) -> np.dtype[np.single]:
        return self.z.dtype

    def __len__(self) -> int:
        return int(self.z.shape[0])

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of events")
        return cls(
            z=np.concatenate([part.z for part in parts], axis=0),
            x=np.concatenate([part.x for part in parts], axis=0),
        )


@dataclass(frozen=True, eq=False, slots=True)
class Populations:
    mc: Events
    data: EventArray
    truth: EventArray

    def __post_init__(self) -> None:
        # `EventArray` pins the dtype for the type checkers only, so a source
        # that forgot to narrow would otherwise travel as far as the first
        # jitted function, which fails far from the cause or silently casts.
        # Every source builds one of these, which makes it the one boundary
        # worth checking at run time.
        arrays: dict[str, EventArray] = {
            "mc.z": self.mc.z,
            "mc.x": self.mc.x,
            "data": self.data,
            "truth": self.truth,
        }
        for name, array in arrays.items():
            if array.dtype != EVENT_DTYPE:
                raise ValueError(
                    f"{name} is {array.dtype}; events must be "
                    f"{np.dtype(EVENT_DTYPE)}, narrowed where the source builds them"
                )
            if array.ndim != 2:
                raise ValueError(
                    f"{name} has shape {array.shape}; events must be 2-D, "
                    "(events, features), even with a single feature"
                )
        for name, array, reference in (
            ("truth", self.truth, "mc.z"),
            ("data", self.data, "mc.x"),
        ):
            if array.shape[1] != arrays[reference].shape[1]:
                raise ValueError(
                    f"{name} has {array.shape[1]} features and {reference} has "
                    f"{arrays[reference].shape[1]}; they describe the same "
                    "observables"
                )
        if self.data.shape[0] != self.truth.shape[0]:
            raise ValueError(
                f"data has {self.data.shape[0]} rows and truth has "
                f"{self.truth.shape[0]}; both describe the same nature events"
            )
        if len(self.mc) == 0 or self.data.shape[0] == 0:
            raise ValueError(
                f"populations must be nonempty, got {self.data.shape[0]} nature "
                f"and {len(self.mc)} MC events"
            )

    @classmethod
    def create(
        cls,
        mc: Events,
        data: EventArray,
        truth: EventArray | None = None,
    ) -> Self:
        resolved_truth: EventArray = (
            truth
            if truth is not None
            else np.full(
                (data.shape[0], *mc.z.shape[1:]), TRUTH_SENTINEL, dtype=mc.z.dtype
            )
        )
        return cls(mc=mc, data=data, truth=resolved_truth)

    @property
    def has_truth(self) -> bool:
        return not np.all(a=self.truth == TRUTH_SENTINEL)

    def require_truth(self) -> EventArray:
        if not self.has_truth:
            raise ValueError(
                "this sample has no particle-level truth to score against: it "
                "was built without one, so `truth` is the sentinel stand-in"
            )
        return self.truth

    def interleave(self) -> ZXY:
        return ZXY(
            Events(
                z=np.concatenate([self.truth, self.mc.z], axis=0),
                x=np.concatenate([self.data, self.mc.x], axis=0),
            ),
            y=np.concatenate(
                [
                    np.ones(self.data.shape[0], dtype=np.ubyte),
                    np.zeros(shape=len(self.mc), dtype=np.ubyte),
                ]
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class ZXY:
    events: Events
    y: NDArray[np.ubyte]

    def __post_init__(self) -> None:
        if self.y.ndim != 1 or self.y.shape[0] != len(self.events):
            raise ValueError(
                f"y has shape {self.y.shape}; expected one label per event in a "
                f"one-dimensional array of length {len(self.events)}"
            )
        bad_labels: NDArray[np.bool_] = cast(
            typ="NDArray[np.bool_]", val=(self.y != 0) & (self.y != 1)
        )
        if np.any(a=bad_labels):
            raise ValueError("labels must be zero (MC) or one (nature)")

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def z(self) -> EventArray:
        return self.events.z

    @property
    def x(self) -> EventArray:
        return self.events.x

    @property
    def dtype(self) -> np.dtype[np.single]:
        return self.events.dtype

    @classmethod
    def concatenate(cls, parts: Sequence[Self]) -> Self:
        if not parts:
            raise ValueError("cannot concatenate an empty sequence of labelled events")
        return cls(
            events=Events.concatenate(parts=[part.events for part in parts]),
            y=np.concatenate([part.y for part in parts], axis=0),
        )

    def partition(self) -> Populations:
        """Separate the labelled events into the four physics populations."""
        mc: NDArray[np.bool] = self.y == 0
        nature: NDArray[np.bool] = ~mc
        return Populations(
            mc=Events(self.events.z[mc], self.events.x[mc]),
            data=self.events.x[nature],
            truth=self.events.z[nature],
        )


class DatasetSplits(NamedTuple):
    train: ArrayDataset
    val: ArrayDataset
    test: ArrayDataset

    def select(self, which: Split = Split.ALL, /) -> ZXY:
        chosen: list[ZXY] = [
            split.as_arrays()
            for flag, split in (
                (Split.TRAIN, self.train),
                (Split.VAL, self.val),
                (Split.TEST, self.test),
            )
            if flag in which
        ]
        if not chosen:
            raise ValueError("select needs at least one split")
        return ZXY.concatenate(parts=chosen)
