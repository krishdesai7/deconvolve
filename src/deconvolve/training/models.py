from __future__ import annotations

from typing import TYPE_CHECKING, cast

import keras

if TYPE_CHECKING:
    from typing import Protocol

    from keras.src.backend.common.keras_tensor import KerasTensor

    from ..coretypes import DeconvolveModel

    class _InputFactory(Protocol):
        def __call__(
            self, *, shape: tuple[int | None, ...], dtype: str
        ) -> keras.KerasTensor: ...

    class _DenseLayer(Protocol):
        def __call__(self, inputs: keras.KerasTensor) -> keras.KerasTensor: ...

    class _DenseFactory(Protocol):
        def __call__(
            self, units: int, *, activation: str, dtype: str
        ) -> _DenseLayer: ...

    class _ModelFactory(Protocol):
        def __call__(
            self,
            inputs: keras.KerasTensor,
            outputs: keras.KerasTensor,
            *,
            name: str,
        ) -> DeconvolveModel: ...


_keras_input: _InputFactory = cast(typ="_InputFactory", val=keras.Input)
_keras_dense: _DenseFactory = cast(typ="_DenseFactory", val=keras.layers.Dense)
_keras_model: _ModelFactory = cast(typ="_ModelFactory", val=keras.Model)


def build_generator(
    dim: int = 1, hidden_units: int = 64, n_layers: int = 2
) -> DeconvolveModel:
    """g(z): nominal-level events -> per-event weights."""
    inputs: KerasTensor = _keras_input(shape=(dim,), dtype="float32")
    x: KerasTensor = inputs
    for _ in range(n_layers):
        x = _keras_dense(hidden_units, activation="relu", dtype="float32")(inputs=x)
    x = _keras_dense(units=1, activation="softplus", dtype="float32")(inputs=x)
    return _keras_model(inputs, outputs=x, name="generator")


def build_discriminator(
    dim: int = 1, hidden_units: int = 64, n_layers: int = 2
) -> DeconvolveModel:
    """d(x): reco-level events -> data vs MC probability."""
    inputs: KerasTensor = _keras_input(shape=(dim,), dtype="float32")
    x: KerasTensor = inputs
    for _ in range(n_layers):
        x = _keras_dense(hidden_units, activation="relu", dtype="float32")(inputs=x)
    x = _keras_dense(units=1, activation="sigmoid", dtype="float32")(inputs=x)
    return _keras_model(inputs, outputs=x, name="discriminator")
