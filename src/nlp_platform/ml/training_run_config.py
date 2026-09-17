from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingRunConfig:
    """Configuration for a complete model training run."""

    epochs: int = 5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError(
                "epochs must be at least 1"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must not be negative"
            )

        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 1
        ):
            raise ValueError(
                "early_stopping_patience must be at least 1"
            )

        if self.early_stopping_min_delta < 0:
            raise ValueError(
                "early_stopping_min_delta must not be negative"
            )

    @property
    def early_stopping_enabled(self) -> bool:
        """Return whether early stopping is enabled."""

        return self.early_stopping_patience is not None