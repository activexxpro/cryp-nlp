import pytest

from nlp_platform.ml.training_config import TrainingConfig


def test_default_training_config() -> None:
    config = TrainingConfig()

    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.0
    assert config.epochs == 5


def test_custom_training_config() -> None:
    config = TrainingConfig(
        learning_rate=0.01,
        weight_decay=0.001,
        epochs=10,
    )

    assert config.learning_rate == 0.01
    assert config.weight_decay == 0.001
    assert config.epochs == 10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0},
        {"learning_rate": -0.1},
        {"weight_decay": -0.1},
        {"epochs": 0},
        {"epochs": -1},
    ],
)
def test_invalid_training_config_raises(
    kwargs: dict[str, float | int],
) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)