from nlp_platform.ml.transformer_evaluator import (
    TransformerEvaluationResult,
)
from nlp_platform.ml.transformer_result import TransformerRunResult
from nlp_platform.ml.transformer_trainer import TransformerTrainingResult


def create_training_result() -> TransformerTrainingResult:
    return TransformerTrainingResult(
        average_loss=0.42,
        samples=100,
        steps=10,
    )


def create_evaluation_result() -> TransformerEvaluationResult:
    return TransformerEvaluationResult(
        loss=0.35,
        accuracy=0.91,
        samples=20,
        batches=2,
    )


def test_transformer_run_result() -> None:
    result = TransformerRunResult(
        training=create_training_result(),
        evaluation=create_evaluation_result(),
        epochs_completed=3,
        best_validation_loss=0.30,
        best_epoch=2,
    )

    assert result.training.average_loss == 0.42
    assert result.evaluation.accuracy == 0.91
    assert result.epochs_completed == 3
    assert result.best_validation_loss == 0.30
    assert result.best_epoch == 2


def test_final_loss() -> None:
    result = TransformerRunResult(
        training=create_training_result(),
        evaluation=create_evaluation_result(),
        epochs_completed=3,
    )

    assert result.final_loss == 0.35


def test_final_accuracy() -> None:
    result = TransformerRunResult(
        training=create_training_result(),
        evaluation=create_evaluation_result(),
        epochs_completed=3,
    )

    assert result.final_accuracy == 0.91


def test_samples_evaluated() -> None:
    result = TransformerRunResult(
        training=create_training_result(),
        evaluation=create_evaluation_result(),
        epochs_completed=3,
    )

    assert result.samples_evaluated == 20


def test_optional_best_metrics_default_to_none() -> None:
    result = TransformerRunResult(
        training=create_training_result(),
        evaluation=create_evaluation_result(),
        epochs_completed=1,
    )

    assert result.best_validation_loss is None
    assert result.best_epoch is None