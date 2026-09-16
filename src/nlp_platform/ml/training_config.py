from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 5

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must not be negative"
            )

        if self.epochs < 1:
            raise ValueError(
                "epochs must be at least 1"
            )