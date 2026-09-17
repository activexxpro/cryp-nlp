import torch
from torch import nn

from nlp_platform.ml.early_stopping import EarlyStopping
from nlp_platform.ml.training_factory import (
    TrainingComponents,
    create_training_components,
)
from nlp_platform.ml.training_run_config import TrainingRunConfig


def create_model() -> nn.Module:
    return nn.Linear(4, 2)


def test_create_training_components() -> None:
    model = create_model()

    components = create_training_components(
        model,
        TrainingRunConfig(),
    )

    assert isinstance(
        components,
        TrainingComponents,
    )
    assert isinstance(
        components.optimizer,
        torch.optim.Adam,
    )
    assert components.early_stopping is None


def test_create_training_components_with_early_stopping() -> None:
    model = create_model()

    components = create_training_components(
        model,
        TrainingRunConfig(
            early_stopping_patience=3,
            early_stopping_min_delta=0.01,
        ),
    )

    assert isinstance(
        components.optimizer,
        torch.optim.Adam,
    )
    assert isinstance(
        components.early_stopping,
        EarlyStopping,
    )


def test_optimizer_uses_configured_learning_rate() -> None:
    model = create_model()

    components = create_training_components(
        model,
        TrainingRunConfig(
            learning_rate=0.0005,
        ),
    )

    learning_rate = components.optimizer.param_groups[0]["lr"]

    assert learning_rate == 0.0005


def test_optimizer_uses_configured_weight_decay() -> None:
    model = create_model()

    components = create_training_components(
        model,
        TrainingRunConfig(
            weight_decay=0.01,
        ),
    )

    weight_decay = (
        components.optimizer.param_groups[0]["weight_decay"]
    )

    assert weight_decay == 0.01