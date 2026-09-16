import torch
from torch import nn

from nlp_platform.ml.training_config import TrainingConfig


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig | None = None,
) -> torch.optim.Optimizer:
    """Create an Adam optimizer for the model."""

    training_config = config or TrainingConfig()

    return torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )