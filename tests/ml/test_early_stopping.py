import pytest

from nlp_platform.ml.early_stopping import (
    EarlyStopping,
    EarlyStoppingConfig,
)


def test_default_config() -> None:
    config = EarlyStoppingConfig()

    assert config.patience == 3
    assert config.min_delta == 0.0


def test_custom_config() -> None:
    config = EarlyStoppingConfig(
        patience=5,
        min_delta=0.01,
    )

    assert config.patience == 5
    assert config.min_delta == 0.01


@pytest.mark.parametrize(
    "patience",
    [0, -1],
)
def test_invalid_patience(patience: int) -> None:
    with pytest.raises(ValueError):
        EarlyStoppingConfig(
            patience=patience
        )


def test_invalid_min_delta() -> None:
    with pytest.raises(ValueError):
        EarlyStoppingConfig(
            min_delta=-0.1
        )


def test_starts_without_best_loss() -> None:
    stopping = EarlyStopping()

    assert stopping.best_loss is None
    assert stopping.bad_epochs == 0
    assert not stopping.should_stop


def test_first_loss_is_improvement() -> None:
    stopping = EarlyStopping()

    improved = stopping.update(0.8)

    assert improved
    assert stopping.best_loss == 0.8
    assert stopping.bad_epochs == 0
    assert not stopping.should_stop


def test_improved_loss_resets_bad_epochs() -> None:
    stopping = EarlyStopping()

    stopping.update(0.8)
    stopping.update(0.9)

    assert stopping.bad_epochs == 1

    improved = stopping.update(0.7)

    assert improved
    assert stopping.best_loss == 0.7
    assert stopping.bad_epochs == 0


def test_non_improving_loss_increments_bad_epochs() -> None:
    stopping = EarlyStopping(
        EarlyStoppingConfig(patience=3)
    )

    stopping.update(0.8)

    assert not stopping.update(0.9)
    assert stopping.bad_epochs == 1

    assert not stopping.update(0.9)
    assert stopping.bad_epochs == 2


def test_should_stop_after_patience() -> None:
    stopping = EarlyStopping(
        EarlyStoppingConfig(patience=2)
    )

    stopping.update(0.8)
    stopping.update(0.9)

    assert not stopping.should_stop

    stopping.update(0.9)

    assert stopping.should_stop


def test_min_delta_controls_improvement() -> None:
    stopping = EarlyStopping(
        EarlyStoppingConfig(
            patience=2,
            min_delta=0.1,
        )
    )

    stopping.update(0.8)

    assert not stopping.update(0.75)
    assert stopping.bad_epochs == 1

    assert stopping.update(0.6)
    assert stopping.best_loss == 0.6


def test_negative_loss_is_rejected() -> None:
    stopping = EarlyStopping()

    with pytest.raises(ValueError):
        stopping.update(-0.1)