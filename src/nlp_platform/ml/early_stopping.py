from dataclasses import dataclass


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Configuration for early stopping."""

    patience: int = 3
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        if self.patience < 1:
            raise ValueError(
                "patience must be at least 1"
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta must not be negative"
            )


class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(
        self,
        config: EarlyStoppingConfig | None = None,
    ) -> None:
        self._config = config or EarlyStoppingConfig()
        self._best_loss: float | None = None
        self._bad_epochs = 0

    @property
    def best_loss(self) -> float | None:
        """Return the best validation loss."""

        return self._best_loss

    @property
    def bad_epochs(self) -> int:
        """Return the number of consecutive non-improving epochs."""

        return self._bad_epochs

    @property
    def should_stop(self) -> bool:
        """Return whether training should stop."""

        return self._bad_epochs >= self._config.patience

    def update(
        self,
        validation_loss: float,
    ) -> bool:
        """Update early stopping state.

        Return True when validation loss improves.
        """

        if validation_loss < 0:
            raise ValueError(
                "validation_loss must not be negative"
            )

        if (
            self._best_loss is None
            or validation_loss
            < self._best_loss - self._config.min_delta
        ):
            self._best_loss = validation_loss
            self._bad_epochs = 0
            return True

        self._bad_epochs += 1
        return False