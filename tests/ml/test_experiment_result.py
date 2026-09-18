from nlp_platform.ml.experiment_result import ExperimentResult
from nlp_platform.ml.training_history import (
    EpochMetrics,
    TrainingHistory,
)


def create_history() -> TrainingHistory:
    return TrainingHistory(
        epochs=[
            EpochMetrics(
                epoch=1,
                train_loss=0.8,
                validation_loss=0.7,
                validation_accuracy=0.75,
            ),
            EpochMetrics(
                epoch=2,
                train_loss=0.5,
                validation_loss=0.4,
                validation_accuracy=0.9,
            ),
        ]
    )


def test_experiment_result() -> None:
    result = ExperimentResult(
        experiment_name="baseline",
        seed=42,
        history=create_history(),
        epochs_completed=2,
        stopped_early=False,
        best_epoch=2,
        best_validation_loss=0.4,
        best_checkpoint="checkpoints/best_model.pt",
    )

    assert result.experiment_name == "baseline"
    assert result.seed == 42
    assert result.epochs_completed == 2
    assert not result.stopped_early
    assert result.best_epoch == 2
    assert result.best_validation_loss == 0.4
    assert result.best_checkpoint == (
        "checkpoints/best_model.pt"
    )


def test_final_validation_loss() -> None:
    result = ExperimentResult(
        experiment_name="baseline",
        seed=42,
        history=create_history(),
        epochs_completed=2,
        stopped_early=False,
        best_epoch=2,
        best_validation_loss=0.4,
        best_checkpoint=None,
    )

    assert result.final_validation_loss == 0.4


def test_final_validation_accuracy() -> None:
    result = ExperimentResult(
        experiment_name="baseline",
        seed=42,
        history=create_history(),
        epochs_completed=2,
        stopped_early=False,
        best_epoch=2,
        best_validation_loss=0.4,
        best_checkpoint=None,
    )

    assert result.final_validation_accuracy == 0.9


def test_empty_history_returns_no_final_metrics() -> None:
    result = ExperimentResult(
        experiment_name="empty",
        seed=42,
        history=TrainingHistory(epochs=[]),
        epochs_completed=0,
        stopped_early=False,
        best_epoch=None,
        best_validation_loss=None,
        best_checkpoint=None,
    )

    assert result.final_validation_loss is None
    assert result.final_validation_accuracy is None