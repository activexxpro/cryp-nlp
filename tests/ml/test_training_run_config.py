import pytest

from nlp_platform.ml.training_run_config import TrainingRunConfig


def test_default_config() -> None:
    config = TrainingRunConfig()

    assert config.epochs == 5
    assert config.learning_rate == 1e-3
    assert config.weight_decay == 0.0
    assert config.early_stopping_patience is None
    assert config.early_stopping_min_delta == 0.0
    assert not config.early_stopping_enabled


def test_custom_config() -> None:
    config = TrainingRunConfig(
        epochs=10,
        learning_rate=0.0005,
        weight_decay=0.01,
        early_stopping_patience=3,
        early_stopping_min_delta=0.001,
    )

    assert config.epochs == 10
    assert config.learning_rate == 0.0005
    assert config.weight_decay == 0.01
    assert config.early_stopping_patience == 3
    assert config.early_stopping_min_delta == 0.001
    assert config.early_stopping_enabled


@pytest.mark.parametrize(
    "epochs",
    [0, -1],
)
def test_invalid_epochs(epochs: int) -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig(epochs=epochs)


def test_invalid_learning_rate() -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig(
            learning_rate=0.0
        )


def test_invalid_weight_decay() -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig(
            weight_decay=-0.1
        )


@pytest.mark.parametrize(
    "patience",
    [0, -1],
)
def test_invalid_early_stopping_patience(
    patience: int,
) -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig(
            early_stopping_patience=patience
        )


def test_invalid_early_stopping_min_delta() -> None:
    with pytest.raises(ValueError):
        TrainingRunConfig(
            early_stopping_min_delta=-0.1
        )