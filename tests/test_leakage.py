"""The z_true leakage check's argument guard.

Nothing here trains. `train` is stubbed out, and which side of it the call stops
on is the assertion: the bad argument must be refused before any model is built.
"""

import pytest
from deconvolve.coretypes import POISON_SENTINEL, TRUTH_SENTINEL
from deconvolve.workflows import leakage


class ReachedTrainingError(Exception):
    """Raised by the stubbed `train`, to mark that the guard let the call past."""


@pytest.fixture
def no_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub(*_args: object, **_kwargs: object) -> None:
        raise ReachedTrainingError

    monkeypatch.setattr(leakage, "train", _stub)


@pytest.mark.usefixtures("no_training")
def test_poisoning_with_the_truth_sentinel_is_refused() -> None:
    """Overwriting z_true with TRUTH_SENTINEL makes the sample indistinguishable
    from one that never had truth, so `require_truth()` would refuse the
    particle-level comparison the check exists to make -- and only after a full
    training run had already been paid for."""
    with pytest.raises(ValueError, match="TRUTH_SENTINEL"):
        leakage.run_leakage_check(
            poison=True, sentinel=float(TRUTH_SENTINEL), seed=0, init_seed=0
        )


@pytest.mark.usefixtures("no_training")
def test_the_clean_arm_ignores_the_sentinel() -> None:
    """The clean arm never writes the sentinel, so its value cannot matter."""
    with pytest.raises(ReachedTrainingError):
        leakage.run_leakage_check(
            poison=False, sentinel=float(TRUTH_SENTINEL), seed=0, init_seed=0
        )


@pytest.mark.usefixtures("no_training")
def test_the_default_poison_value_is_usable() -> None:
    assert POISON_SENTINEL != TRUTH_SENTINEL
    with pytest.raises(ReachedTrainingError):
        leakage.run_leakage_check(
            poison=True, sentinel=float(POISON_SENTINEL), seed=0, init_seed=0
        )
