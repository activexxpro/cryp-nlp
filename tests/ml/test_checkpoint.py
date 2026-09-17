from pathlib import Path

import pytest
import torch
from torch import nn

from nlp_platform.ml.checkpoint import CheckpointManager


def create_model() -> nn.Module:
    return nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 4),
    )


def test_checkpoint_manager_creates_directory(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    CheckpointManager(checkpoint_dir)

    assert checkpoint_dir.exists()


def test_save_creates_checkpoint(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    model = create_model()

    path = manager.save(
        model,
        "model.pt",
    )

    assert path.exists()
    assert path.name == "model.pt"


def test_saved_checkpoint_preserves_parameters(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    model = create_model()

    original = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    path = manager.save(
        model,
        "model.pt",
    )

    restored = create_model()

    manager.load(
        restored,
        path,
        device=torch.device("cpu"),
    )

    for name, parameter in restored.named_parameters():
        assert torch.equal(
            original[name],
            parameter,
        )


def test_load_returns_model(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    model = create_model()

    path = manager.save(
        model,
        "model.pt",
    )

    restored = create_model()

    result = manager.load(
        restored,
        path,
        device=torch.device("cpu"),
    )

    assert result is restored


def test_relative_checkpoint_path_uses_directory(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    manager = CheckpointManager(checkpoint_dir)

    model = create_model()

    path = manager.save(
        model,
        "nested/model.pt",
    )

    assert path == (
        checkpoint_dir / "nested" / "model.pt"
    )
    assert path.exists()


def test_missing_checkpoint_raises(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(
        tmp_path / "checkpoints"
    )

    model = create_model()

    with pytest.raises(
        FileNotFoundError,
        match="checkpoint does not exist",
    ):
        manager.load(
            model,
            "missing.pt",
        )