import pytest
import torch

from nlp_platform.ml.model_config import ModelConfig
from nlp_platform.ml.text_classifier import TextClassifier


def create_model() -> TextClassifier:
    config = ModelConfig(
        vocab_size=10,
        embedding_dim=8,
        hidden_dim=16,
        num_classes=4,
    )

    return TextClassifier(config)


def test_model_can_be_created() -> None:
    model = create_model()

    assert isinstance(model, TextClassifier)


def test_model_has_embedding_layer() -> None:
    model = create_model()

    assert model.embedding.num_embeddings == 10
    assert model.embedding.embedding_dim == 8


def test_forward_returns_class_logits() -> None:
    model = create_model()

    input_ids = torch.tensor(
        [
            [2, 4, 0, 0],
            [2, 3, 5, 0],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape == (2, 4)


def test_forward_preserves_batch_size() -> None:
    model = create_model()

    input_ids = torch.tensor(
        [
            [2, 4, 0, 0],
            [2, 3, 5, 0],
            [6, 7, 0, 0],
        ],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.shape[0] == 3


def test_forward_returns_float_tensor() -> None:
    model = create_model()

    input_ids = torch.tensor(
        [[2, 4, 0, 0]],
        dtype=torch.long,
    )

    logits = model(input_ids)

    assert logits.dtype == torch.float32


def test_padding_tokens_do_not_change_pooled_representation() -> None:
    model = create_model()
    model.eval()

    input_ids_without_extra_padding = torch.tensor(
        [[2, 4]],
        dtype=torch.long,
    )

    input_ids_with_padding = torch.tensor(
        [[2, 4, 0, 0]],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits_without_padding = model(
            input_ids_without_extra_padding
        )
        logits_with_padding = model(
            input_ids_with_padding
        )

    assert torch.allclose(
        logits_without_padding,
        logits_with_padding,
    )


def test_invalid_input_dimensions_raise() -> None:
    model = create_model()

    input_ids = torch.tensor(
        [2, 4, 0, 0],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="two-dimensional tensor",
    ):
        model(input_ids)