from dataclasses import dataclass

from nlp_platform.ml.transformer_evaluator import (
    TransformerEvaluationResult,
)
from nlp_platform.ml.transformer_trainer import TransformerTrainingResult


@dataclass(frozen=True)
class TransformerRunResult:
    """Result of a transformer training and evaluation run."""

    training: TransformerTrainingResult
    evaluation: TransformerEvaluationResult
    epochs_completed: int
    best_validation_loss: float | None = None
    best_epoch: int | None = None

    @property
    def final_loss(self) -> float:
        """Return the final evaluation loss."""
        return self.evaluation.loss

    @property
    def final_accuracy(self) -> float:
        """Return the final evaluation accuracy."""
        return self.evaluation.accuracy

    @property
    def samples_evaluated(self) -> int:
        """Return the number of evaluation samples."""
        return self.evaluation.samples