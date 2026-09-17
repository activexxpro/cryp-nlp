from dataclasses import dataclass

from torch.utils.data import DataLoader

from nlp_platform.ml.best_checkpoint import BestCheckpointTracker
from nlp_platform.ml.evaluator import EvaluationResult, Evaluator
from nlp_platform.ml.trainer import Trainer
from nlp_platform.ml.training_history import (
    EpochMetrics,
    TrainingHistory,
)

from nlp_platform.ml.early_stopping import EarlyStopping

@dataclass(frozen=True)
class TrainingRunResult:
    """Result of a complete model training run."""

    history: TrainingHistory
    epochs_completed: int
    best_epoch: int | None
    best_validation_loss: float | None
    stopped_early: bool


class TrainingService:
    """Orchestrate multi-epoch model training."""

    def __init__(
        self,
        trainer: Trainer,
        evaluator: Evaluator,
        epochs: int,
        checkpoint_tracker: BestCheckpointTracker | None = None,
        early_stopping: EarlyStopping | None = None,
    ) -> None:
        if epochs < 1:
            raise ValueError(
                "epochs must be at least 1"
            )

        self._trainer = trainer
        self._evaluator = evaluator
        self._epochs = epochs
        self._checkpoint_tracker = checkpoint_tracker
        self._early_stopping = early_stopping

    def train(
        self,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
    ) -> TrainingRunResult:
        """Train the model for the configured number of epochs."""

        history = TrainingHistory(epochs=[])
        stopped_early = False

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

            if self._checkpoint_tracker is not None:
                self._checkpoint_tracker.update(
                    model=self._trainer.model,
                    validation_loss=validation_result.loss,
                    epoch=epoch,
                )

            if self._early_stopping is not None:
                self._early_stopping.update(
                    validation_result.loss
                )

                if self._early_stopping.should_stop:
                    stopped_early = True
                    break

        return TrainingRunResult(
            history=history,
            epochs_completed=history.epoch_count,
            best_epoch=(
                self._checkpoint_tracker.best_epoch
                if self._checkpoint_tracker is not None
                else None
            ),
            best_validation_loss=(
                self._checkpoint_tracker.best_loss
                if self._checkpoint_tracker is not None
                else None
            ),
            stopped_early=stopped_early,
        )