from nlp_platform.ml.experiment_comparison import (
    ExperimentSummary,
    summarize_experiment,
    summarize_experiments,
)
from nlp_platform.ml.experiment_result import ExperimentResult
from nlp_platform.ml.training_history import EpochMetrics, TrainingHistory


def create_result(
    name: str,
    accuracy: float,
    loss: float,
) -> ExperimentResult:
    history = TrainingHistory(
        epochs=[
            EpochMetrics(
                epoch=1,
                train_loss=0.8,
                validation_loss=loss,
                validation_accuracy=accuracy,
            ),
        ]
    )

    return ExperimentResult(
        experiment_name=name,
        seed=42,
        history=history,
        epochs_completed=1,
        stopped_early=False,
        best_epoch=1,
        best_validation_loss=loss,
        best_checkpoint=None,
    )


def test_summarize_experiment() -> None:
    result = create_result(
        name="baseline",
        accuracy=0.85,
        loss=0.42,
    )

    summary = summarize_experiment(result)

    assert isinstance(summary, ExperimentSummary)
    assert summary.experiment_name == "baseline"
    assert summary.seed == 42
    assert summary.epochs_completed == 1
    assert summary.final_validation_loss == 0.42
    assert summary.final_validation_accuracy == 0.85
    assert summary.best_validation_loss == 0.42
    assert summary.best_epoch == 1
    assert summary.stopped_early is False


def test_summarize_empty_history() -> None:
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

    summary = summarize_experiment(result)

    assert summary.final_validation_loss is None
    assert summary.final_validation_accuracy is None
    assert summary.best_validation_loss is None
    assert summary.best_epoch is None


def test_summarize_multiple_experiments() -> None:
    results = [
        create_result("baseline", 0.80, 0.50),
        create_result("larger-model", 0.90, 0.30),
    ]

    summaries = summarize_experiments(results)

    assert len(summaries) == 2
    assert summaries[0].experiment_name == "baseline"
    assert summaries[1].experiment_name == "larger-model"
    assert summaries[0].final_validation_accuracy == 0.80
    assert summaries[1].final_validation_accuracy == 0.90