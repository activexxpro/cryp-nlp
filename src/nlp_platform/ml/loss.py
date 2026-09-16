import torch
from torch import nn


def create_classification_loss() -> nn.Module:
    """Create the cross-entropy loss used for classification."""

    return nn.CrossEntropyLoss()


def calculate_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Calculate cross-entropy classification loss."""

    if logits.ndim != 2:
        raise ValueError(
            "logits must be a two-dimensional tensor"
        )

    if labels.ndim != 1:
        raise ValueError(
            "labels must be a one-dimensional tensor"
        )

    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "number of logits must match number of labels"
        )

    return nn.functional.cross_entropy(
        logits,
        labels,
    )