from pathlib import Path

import pytest
import torch
from torch import nn

from nlp_platform.ml.transformer_checkpoint import (
    TransformerCheckpointManager,
)
from nlp_platform.ml.transformer_classifier import TransformerClassifier


class FakeTransformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 2)


def create_classifier() -> TransformerClassifier:
    return TransformerClassifier(
        model=FakeTransformerModel(),
    )


def test_save_creates_checkpoint(
    tmp_path: Path,
) -> None:
    manager = TransformerCheckpointManager(tmp_path)
    classifier = create_classifier()

    path = manager.save(classifier)

    assert path.exists()
    assert path.name == "transformer_model.pt"
    assert manager.exists()


def test_save_supports_custom_filename(
    tmp_path: Path,
) -> None:
    manager = TransformerCheckpointManager(tmp_path)
    classifier = create_classifier()

    path = manager.save(
        classifier,
        filename="custom.pt",
    )

    assert path == tmp_path / "custom.pt"
    assert path.exists()


def test_load_restores_model_weights(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    source = create_classifier()
    manager = TransformerCheckpointManager(tmp_path)

    original_state = {
        name: parameter.detach().clone()
        for name, parameter in source.model.named_parameters()
    }

    manager.save(source)

    restored = create_classifier()

    for parameter in restored.model.parameters():
        parameter.data.zero_()

    manager.load(restored)

    for name, parameter in restored.model.named_parameters():
        assert torch.equal(
            parameter.detach(),
            original_state[name],
        )


def test_load_returns_checkpoint_path(
    tmp_path: Path,
) -> None:
    manager = TransformerCheckpointManager(tmp_path)
    classifier = create_classifier()

    saved_path = manager.save(classifier)
    loaded_path = manager.load(classifier)

    assert loaded_path == saved_path


def test_missing_checkpoint_raises_error(
    tmp_path: Path,
) -> None:
    manager = TransformerCheckpointManager(tmp_path)
    classifier = create_classifier()

    with pytest.raises(
        FileNotFoundError,
        match="Transformer checkpoint not found",
    ):
        manager.load(classifier)


def test_exists_returns_false_for_missing_checkpoint(
    tmp_path: Path,
) -> None:
    manager = TransformerCheckpointManager(tmp_path)

    assert not manager.exists()


def test_directory_property(
    tmp_path: Path,
) -> None:
    manager = TransformerCheckpointManager(tmp_path)

    assert manager.directory == tmp_path