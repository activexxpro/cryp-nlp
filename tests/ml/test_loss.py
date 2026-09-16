import pytest
import torch

from nlp_platform.ml.loss import (
    calculate_classification_loss,
    create_classification_loss,
)


def test_create_classification_loss() -> None:
    loss = create_classification_loss()

    assert isinstance(
        loss,
        torch.nn.CrossEntropyLoss,
    )


def test_calculate_classification_loss() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.5],
            [0.2, 1.5],
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    loss = calculate_classification_loss(
        logits,
        labels,
    )

    assert loss.ndim == 0
    assert loss.item() > 0


def test_loss_returns_scalar_tensor() -> None:
    logits = torch.randn(4, 3)
    labels = torch.tensor(
        [0, 1, 2, 1],
        dtype=torch.long,
    )

    loss = calculate_classification_loss(
        logits,
        labels,
    )

    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()


@pytest.mark.parametrize(
    "logits, labels",
    [
        (
            torch.randn(4),
            torch.tensor([0]),
        ),
        (
            torch.randn(2, 3),
            torch.tensor([[0, 1]]),
        ),
    ],
)
def test_invalid_dimensions_raise(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        calculate_classification_loss(
            logits,
            labels,
        )


def test_mismatched_batch_sizes_raise() -> None:
    logits = torch.randn(3, 4)
    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    with pytest.raises(
        ValueError,
        match="number of logits",
    ):
        calculate_classification_loss(
            logits,
            labels,
        )