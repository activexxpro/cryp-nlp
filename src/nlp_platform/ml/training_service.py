from dataclasses import dataclass

from torch import nn
from torch.utils.data import DataLoader

from nlp_platform.ml.evaluator import EvaluationResult, Evaluator
from nlp_platform.ml.trainer import Trainer
from nlp_platform.ml.training_history import (
    EpochMetrics,
    TrainingHistory,
)


@dataclass(frozen=True)
class TrainingRunResult:
    """Result of a complete model training run."""

    history: TrainingHistory
    epochs_completed: int


class TrainingService:
    """Orchestrate multi-epoch model training."""

    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        epochs: int,
    ) -> None:
        if epochs < 1:
            raise ValueError(
                "epochs must be at least 1"
            )

        self._trainer = trainer
        self._evaluator = evaluator
        self._epochs = epochs

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> TrainingRunResult:
        """Train the model for the configured number of epochs."""

        history = TrainingHistory(epochs=[])

        for epoch in range(1, self._epochs + 1):
            train_result = self._trainer.train_epoch(
                train_dataloader
            )

            validation_result: EvaluationResult = (
                self._evaluator.evaluate(
                    validation_dataloader
                )
            )

            history.add(
                EpochMetrics(
                    epoch=epoch,
                    train_loss=train_result.loss,
                    validation_loss=validation_result.loss,
                    validation_accuracy=(
                        validation_result.accuracy
                    ),
                )
            )

        return TrainingRunResult(
            history=history,
            epochs_completed=history.epoch_count,
        )