from dataclasses import dataclass

from nlp_platform.ml.training_history import TrainingHistory


@dataclass(frozen=True)
class ExperimentResult:
    """Summarize the outcome of a training experiment."""

    experiment_name: str
    seed: int
    history: TrainingHistory
    epochs_completed: int
    stopped_early: bool
    best_epoch: int | None
    best_validation_loss: float | None
    best_checkpoint: str | None

    @property
    def final_validation_loss(self) -> float | None:
        """Return the final recorded validation loss."""

        if not self.history.epochs:
            return None

        return self.history.epochs[-1].validation_loss

    @property
    def final_validation_accuracy(self) -> float | None:
        """Return the final recorded validation accuracy."""

        if not self.history.epochs:
            return None

        return self.history.epochs[-1].validation_accuracy