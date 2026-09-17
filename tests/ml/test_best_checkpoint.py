from pathlib import Path

import pytest
import torch
from torch import nn

from nlp_platform.ml.best_checkpoint import (
    BestCheckpointTracker,
)
from nlp_platform.ml.checkpoint import CheckpointManager


def create_model() -> nn.Module:
    return nn.Linear(4, 2)


def create_tracker(
    tmp_path: Path,
) -> BestCheckpointTracker:
    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    return BestCheckpointTracker(manager)


def test_tracker_starts_without_best_model(
    tmp_path: Path,
) -> None:
    tracker = create_tracker(tmp_path)

    assert tracker.best_loss is None
    assert tracker.best_epoch is None
    assert tracker.best_path is None


def test_first_update_becomes_best(
    tmp_path: Path,
) -> None:
    tracker = create_tracker(tmp_path)
    model = create_model()

    result = tracker.update(
        model=model,
        validation_loss=0.8,
        epoch=1,
    )

    assert result is True
    assert tracker.best_loss == 0.8
    assert tracker.best_epoch == 1
    assert tracker.best_path is not None
    assert tracker.best_path.exists()


def test_improved_loss_updates_best(
    tmp_path: Path,
) -> None:
    tracker = create_tracker(tmp_path)
    model = create_model()

    tracker.update(
        model=model,
        validation_loss=0.8,
        epoch=1,
    )

    result = tracker.update(
        model=model,
        validation_loss=0.5,
        epoch=2,
    )

    assert result is True
    assert tracker.best_loss == 0.5
    assert tracker.best_epoch == 2


def test_worse_loss_does_not_update_best(
    tmp_path: Path,
) -> None:
    tracker = create_tracker(tmp_path)
    model = create_model()

    tracker.update(
        model=model,
        validation_loss=0.5,
        epoch=1,
    )

    result = tracker.update(
        model=model,
        validation_loss=0.8,
        epoch=2,
    )

    assert result is False
    assert tracker.best_loss == 0.5
    assert tracker.best_epoch == 1


def test_equal_loss_does_not_update_best(
    tmp_path: Path,
) -> None:
    tracker = create_tracker(tmp_path)
    model = create_model()

    tracker.update(
        model=model,
        validation_loss=0.5,
        epoch=1,
    )

    result = tracker.update(
        model=model,
        validation_loss=0.5,
        epoch=2,
    )

    assert result is False
    assert tracker.best_epoch == 1


def test_best_checkpoint_can_be_loaded(
    tmp_path: Path,
) -> None:
    tracker = create_tracker(tmp_path)
    model = create_model()

    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    tracker.update(
        model=model,
        validation_loss=0.4,
        epoch=3,
    )

    restored = create_model()

    assert tracker.best_path is not None

    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    manager.load(
        restored,
        tracker.best_path,
        device=torch.device("cpu"),
    )

    for name, parameter in restored.named_parameters():
        assert torch.equal(
            original[name],
            parameter,
        )


@pytest.mark.parametrize(
    "validation_loss, epoch",
    [
        (-0.1, 1),
        (0.5, 0),
    ],
)
def test_invalid_update_values_raise(
    tmp_path: Path,
    validation_loss: float,
    epoch: int,
) -> None:
    tracker = create_tracker(tmp_path)

    with pytest.raises(ValueError):
        tracker.update(
            model=create_model(),
            validation_loss=validation_loss,
            epoch=epoch,
        )


def test_empty_filename_raises(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    with pytest.raises(
        ValueError,
        match="filename must not be empty",
    ):
        BestCheckpointTracker(
            manager,
            filename="   ",
        )