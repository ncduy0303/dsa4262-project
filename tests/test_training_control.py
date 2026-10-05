"""Test stopping/checkpoint selection without installing legacy PyTorch in uv."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "training_control",
    Path(__file__).resolve().parents[1] / "environments/m6anet/training_control.py",
)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def test_stopping_keeps_absolute_best_but_ignores_tiny_improvements_for_patience():
    stopper = control.EarlyStopping(patience=2, min_delta=0.01, min_epochs=1)
    assert stopper.step(1, 0.5) == (True, False)
    assert stopper.step(2, 0.499) == (True, False)
    assert stopper.step(3, 0.498) == (True, True)
    assert stopper.best_epoch == 3
    assert stopper.best_loss == 0.498


def test_minimum_epochs_and_reset_on_meaningful_improvement():
    stopper = control.EarlyStopping(patience=2, min_delta=0.01, min_epochs=5)
    for epoch, loss in enumerate([0.5, 0.6, 0.6, 0.4, 0.41], 1):
        _, stop = stopper.step(epoch, loss)
        assert not stop
    assert stopper.step(6, 0.42) == (False, True)
    assert stopper.best_epoch == 4


def test_invalid_loss_cannot_replace_checkpoint():
    stopper = control.EarlyStopping()
    stopper.step(1, 0.4)
    with pytest.raises(ValueError, match="Non-finite"):
        stopper.step(2, float("nan"))
    assert stopper.best_epoch == 1
