from dataclasses import dataclass

from nlp_platform.ml.experiment_result import ExperimentResult


@dataclass(frozen=True)
class ExperimentSummary:
    """Comparable summary of one experiment."""

    experiment_name: str
    seed: int
    epochs_completed: int
    final_validation_loss: float | None
    final_validation_accuracy: float | None
    best_validation_loss: float | None
    best_epoch: int | None
    stopped_early: bool


def summarize_experiment(
    result: ExperimentResult,
) -> ExperimentSummary:
    """Create a comparison summary for one experiment."""
    return ExperimentSummary(
        experiment_name=result.experiment_name,
        seed=result.seed,
        epochs_completed=result.epochs_completed,
        final_validation_loss=result.final_validation_loss,
        final_validation_accuracy=result.final_validation_accuracy,
        best_validation_loss=result.best_validation_loss,
        best_epoch=result.best_epoch,
        stopped_early=result.stopped_early,
    )


def summarize_experiments(
    results: list[ExperimentResult],
) -> list[ExperimentSummary]:
    """Create summaries for multiple experiments."""
    return [summarize_experiment(result) for result in results]