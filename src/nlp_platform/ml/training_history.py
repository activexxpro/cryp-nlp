from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics recorded for one training epoch."""

    epoch: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float


@dataclass
class TrainingHistory:
    """Collection of metrics recorded during training."""

    epochs: list[EpochMetrics]

    def add(self, metrics: EpochMetrics) -> None:
        """Add metrics for one epoch."""

        if metrics.epoch < 1:
            raise ValueError(
                "epoch must be at least 1"
            )

        self.epochs.append(metrics)

    @property
    def epoch_count(self) -> int:
        """Return the number of recorded epochs."""

        return len(self.epochs)

    @property
    def train_losses(self) -> list[float]:
        """Return training losses."""

        return [
            metrics.train_loss
            for metrics in self.epochs
        ]

    @property
    def validation_losses(self) -> list[float]:
        """Return validation losses."""

        return [
            metrics.validation_loss
            for metrics in self.epochs
        ]

    @property
    def validation_accuracies(self) -> list[float]:
        """Return validation accuracies."""

        return [
            metrics.validation_accuracy
            for metrics in self.epochs
        ]