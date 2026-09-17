import torch
from torch import nn

from nlp_platform.ml.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
)
from nlp_platform.ml.optimizer import create_optimizer
from nlp_platform.ml.training_config import TrainingConfig
from nlp_platform.ml.training_run_config import TrainingRunConfig


class TrainingComponents:
    """Components required to execute a training run."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        early_stopping: EarlyStopping | None,
    ) -> None:
        self.optimizer = optimizer
        self.early_stopping = early_stopping


def create_training_components(
    model: nn.Module,
    config: TrainingRunConfig,
) -> TrainingComponents:
    """Create optimizer and optional early stopping."""

    optimizer = create_optimizer(
        model,
        TrainingConfig(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            epochs=config.epochs,
        ),
    )

    early_stopping = None

    if config.early_stopping_enabled:
        early_stopping = EarlyStopping(
            EarlyStoppingConfig(
                patience=config.early_stopping_patience,
                min_delta=config.early_stopping_min_delta,
            )
        )

    return TrainingComponents(
        optimizer=optimizer,
        early_stopping=early_stopping,
    )