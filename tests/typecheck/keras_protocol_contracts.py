from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, assert_type

from deconvolve.coretypes import (
    DeconvolveModel,
    KerasVariable,
    Variables,
)
from deconvolve.evaluation.evaluate import _get_weights as evaluate_weights

# Specifically what this test is checking
from deconvolve.evaluation.plotting import _get_weights as plotting_weights  # pyrefly: ignore[implicit-reexport]
from deconvolve.training.engine import TrainResult, _make_steps
from deconvolve.training.models import build_discriminator, build_generator
from deconvolve.workflows.train import _load_artifacts
from jax._src.basearray import Array as JaxArray

if TYPE_CHECKING:
    from typing import Any

    from deconvolve.coretypes import EvalStep, StatelessOptimizer, TrainStep
    from numpy.typing import ArrayLike, NDArray


def check_protocol_surfaces(
    model: DeconvolveModel,
    optimizer: StatelessOptimizer,
    variable: KerasVariable,
    inputs: ArrayLike,
    state: Variables,
) -> None:
    assert_type(model.trainable_variables, list[KerasVariable])
    assert_type(model.non_trainable_variables, list[KerasVariable])
    assert_type(model(inputs), JaxArray)
    output, next_state = model.stateless_call(state, state, inputs, training=True)
    assert_type(output, JaxArray)
    assert_type(next_state, Variables)
    model.save(filepath="model.keras")

    assert_type(variable.value, JaxArray)
    variable.assign(value=state[0])

    optimizer.build(model.trainable_variables)
    assert_type(optimizer.variables, list[KerasVariable])
    trainable, optimizer_state = optimizer.stateless_apply(state, state, state)
    assert_type(trainable, Variables)
    assert_type(optimizer_state, Variables)


def check_training_boundary(
    model: DeconvolveModel, optimizer: StatelessOptimizer
) -> None:
    assert_type(build_generator(), DeconvolveModel)
    assert_type(build_discriminator(), DeconvolveModel)
    _: tuple[TrainStep, TrainStep, EvalStep] = _make_steps(
        g=model, d=model, opt_g=optimizer, opt_d=optimizer, lambda_dispersion=0.0
    )


def check_model_consumers(
    model: DeconvolveModel,
    inputs: NDArray[Any],
    result: TrainResult,
) -> None:
    _ = evaluate_weights(g=model, z_gen=inputs)
    _ = plotting_weights(g=model, z_gen=inputs)
    assert_type(result.g, DeconvolveModel)
    loaded, _ = _load_artifacts(run_dir=Path("run"))
    assert_type(loaded, DeconvolveModel)
