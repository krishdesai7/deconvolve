from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict

import numpy as np

# `Variables` is used as a runtime annotation by the `@jaxtyped(beartype)`
# functions in `deconvolve.training.engine`, and beartype has to evaluate the
# alias to check it -- so `JaxArray` cannot hide under TYPE_CHECKING.
from jax import Array as JaxArray

if TYPE_CHECKING:
    from os import PathLike

    from numpy.typing import ArrayLike, NDArray

    from ..training import TrainState

# ---------------------------------
# Plotting
# ---------------------------------


class VarInfo(TypedDict):
    xlim: tuple[float, float]
    xlabel: str
    symbol: str
    mu: float
    sigma: float


# ---------------------------------
# Training
# ---------------------------------
type Variables = list[JaxArray]


class KerasVariable(Protocol):
    @property
    def value(self) -> JaxArray: ...

    def assign(self, value: JaxArray) -> None: ...


class DeconvolveModel(Protocol):
    @property
    def trainable_variables(self) -> list[KerasVariable]: ...

    @property
    def non_trainable_variables(self) -> list[KerasVariable]: ...

    def __call__(self, inputs: ArrayLike, /) -> JaxArray: ...

    def stateless_call(
        self,
        trainable_variables: Variables,
        non_trainable_variables: Variables,
        inputs: ArrayLike,
        *,
        training: bool,
    ) -> tuple[JaxArray, Variables]: ...

    def save(self, filepath: str | PathLike[str]) -> None: ...


class StatelessOptimizer(Protocol):
    @property
    def variables(self) -> list[KerasVariable]: ...

    def build(self, variables: list[KerasVariable]) -> None: ...

    def stateless_apply(
        self,
        optimizer_variables: Variables,
        grads: Variables,
        trainable_variables: Variables,
    ) -> tuple[Variables, Variables]: ...


# `jax.value_and_grad(..., has_aux=True)` reveals as `(...) -> tuple[Any, Any]`,
# so naming the real shape here is a legal narrowing that types the
# destructuring at the call sites instead of leaving it as `Any`.
type GradsAndAux = tuple[tuple[JaxArray, Variables], Variables]
#: `g`'s aux carries a second array: the loss it is *scored* on, which is not
#: the loss it is differentiated through once a dispersion penalty is on. See
#: `engine.weight_dispersion`.
type GenGradsAndAux = tuple[tuple[JaxArray, tuple[Variables, JaxArray]], Variables]


class DiscGradFn(Protocol):
    def __call__(
        self,
        d_trainable: Variables,
        d_non_trainable: Variables,
        x: JaxArray,
        y: JaxArray,
        w: JaxArray,
        mask: JaxArray,
        /,
    ) -> GradsAndAux: ...


class GenGradFn(Protocol):
    def __call__(
        self,
        g_trainable: Variables,
        g_non_trainable: Variables,
        d_trainable: Variables,
        d_non_trainable: Variables,
        z: JaxArray,
        x: JaxArray,
        y: JaxArray,
        mask: JaxArray,
        /,
    ) -> GenGradsAndAux: ...


class TrainStep(Protocol):
    """A traced ``(state, z, x, y, mask) -> (state, loss)`` update."""

    def __call__(
        self,
        state: TrainState,
        z: JaxArray,
        x: JaxArray,
        y: JaxArray,
        mask: JaxArray,
        /,
    ) -> tuple[TrainState, JaxArray]: ...


class EvalStep(Protocol):
    """A traced forward pass with no updates.

    Returns the *unnormalized* masked loss and the mask total, so a scan over
    batches can accumulate both and divide once --- a true mean over the split
    rather than a mean of per-batch means.
    """

    def __call__(
        self,
        state: TrainState,
        z: JaxArray,
        x: JaxArray,
        y: JaxArray,
        mask: JaxArray,
        /,
    ) -> tuple[JaxArray, JaxArray]: ...


# ---------------------------------
# Baselines
# ---------------------------------


class MetricRecord(TypedDict):
    """Before/after scores for one variable at one level."""

    wasserstein_before: float
    wasserstein_after: float
    wasserstein_improvement_pct: float
    jensenshannon_before: float
    jensenshannon_after: float
    jensenshannon_improvement_pct: float
    triangular_before: float
    triangular_after: float
    triangular_improvement_pct: float


# ---------------------------------
# Data
# ---------------------------------
type Nested[T] = T | list[Nested[T]]

# Every event array in the pipeline. The dtype is pinned in
# `deconvolve.coretypes.constants.EVENT_DTYPE`; this is its annotation-space twin, so
# changing precision is two lines rather than a sweep through the package.
type EventArray = NDArray[np.single]
