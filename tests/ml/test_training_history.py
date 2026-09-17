import pytest

from nlp_platform.ml.training_history import (
    EpochMetrics,
    TrainingHistory,
)


def create_metrics(epoch: int = 1) -> EpochMetrics:
    return EpochMetrics(
        epoch=epoch,
        train_loss=0.8,
        validation_loss=0.7,
        validation_accuracy=0.75,
    )


def test_epoch_metrics_stores_values() -> None:
    metrics = create_metrics()

    assert metrics.epoch == 1
    assert metrics.train_loss == 0.8
    assert metrics.validation_loss == 0.7
    assert metrics.validation_accuracy == 0.75


def test_training_history_starts_empty() -> None:
    history = TrainingHistory(epochs=[])

    assert history.epoch_count == 0
    assert history.epochs == []


def test_add_epoch_metrics() -> None:
    history = TrainingHistory(epochs=[])

    history.add(create_metrics())

    assert history.epoch_count == 1
    assert history.epochs[0].epoch == 1


def test_multiple_epochs_preserve_order() -> None:
    history = TrainingHistory(epochs=[])

    history.add(create_metrics(1))
    history.add(create_metrics(2))
    history.add(create_metrics(3))

    assert [item.epoch for item in history.epochs] == [
        1,
        2,
        3,
    ]


def test_train_losses() -> None:
    history = TrainingHistory(epochs=[])

    history.add(
        EpochMetrics(
            epoch=1,
            train_loss=0.9,
            validation_loss=0.8,
            validation_accuracy=0.6,
        )
    )
    history.add(
        EpochMetrics(
            epoch=2,
            train_loss=0.5,
            validation_loss=0.4,
            validation_accuracy=0.8,
        )
    )

    assert history.train_losses == [0.9, 0.5]


def test_validation_losses() -> None:
    history = TrainingHistory(epochs=[])

    history.add(
        EpochMetrics(
            epoch=1,
            train_loss=0.9,
            validation_loss=0.8,
            validation_accuracy=0.6,
        )
    )
    history.add(
        EpochMetrics(
            epoch=2,
            train_loss=0.5,
            validation_loss=0.4,
            validation_accuracy=0.8,
        )
    )

    assert history.validation_losses == [0.8, 0.4]


def test_validation_accuracies() -> None:
    history = TrainingHistory(epochs=[])

    history.add(
        EpochMetrics(
            epoch=1,
            train_loss=0.9,
            validation_loss=0.8,
            validation_accuracy=0.6,
        )
    )
    history.add(
        EpochMetrics(
            epoch=2,
            train_loss=0.5,
            validation_loss=0.4,
            validation_accuracy=0.8,
        )
    )

    assert history.validation_accuracies == [0.6, 0.8]


def test_invalid_epoch_raises() -> None:
    history = TrainingHistory(epochs=[])

    with pytest.raises(
        ValueError,
        match="epoch must be at least 1",
    ):
        history.add(create_metrics(0))