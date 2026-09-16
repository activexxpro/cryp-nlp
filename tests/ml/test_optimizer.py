import torch
from torch import nn

from nlp_platform.ml.optimizer import create_optimizer
from nlp_platform.ml.training_config import TrainingConfig


def test_create_optimizer() -> None:
    model = nn.Linear(8, 4)

    optimizer = create_optimizer(model)

    assert isinstance(
        optimizer,
        torch.optim.Adam,
    )


def test_optimizer_uses_learning_rate() -> None:
    model = nn.Linear(8, 4)

    config = TrainingConfig(
        learning_rate=0.005,
    )

    optimizer = create_optimizer(
        model,
        config,
    )

    assert optimizer.param_groups[0]["lr"] == 0.005


def test_optimizer_uses_weight_decay() -> None:
    model = nn.Linear(8, 4)

    config = TrainingConfig(
        weight_decay=0.01,
    )

    optimizer = create_optimizer(
        model,
        config,
    )

    assert (
        optimizer.param_groups[0]["weight_decay"]
        == 0.01
    )