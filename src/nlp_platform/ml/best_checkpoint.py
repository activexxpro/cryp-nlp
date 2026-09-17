from pathlib import Path

import torch
from torch import nn

from nlp_platform.ml.checkpoint import CheckpointManager


class BestCheckpointTracker:
    """Track and save the best model checkpoint."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        filename: str = "best_model.pt",
    ) -> None:
        if not filename.strip():
            raise ValueError(
                "filename must not be empty"
            )

        self._checkpoint_manager = checkpoint_manager
        self._filename = filename
        self._best_loss: float | None = None
        self._best_epoch: int | None = None
        self._best_path: Path | None = None

    @property
    def best_loss(self) -> float | None:
        """Return the best validation loss."""

        return self._best_loss

    @property
    def best_epoch(self) -> int | None:
        """Return the epoch with the best validation loss."""

        return self._best_epoch

    @property
    def best_path(self) -> Path | None:
        """Return the best checkpoint path."""

        return self._best_path

    def update(
        self,
        model: nn.Module,
        validation_loss: float,
        epoch: int,
    ) -> bool:
        """Save the model when validation loss improves."""

        if epoch < 1:
            raise ValueError(
                "epoch must be at least 1"
            )

        if validation_loss < 0:
            raise ValueError(
                "validation_loss must not be negative"
            )

        is_best = (
            self._best_loss is None
            or validation_loss < self._best_loss
        )

        if not is_best:
            return False

        path = self._checkpoint_manager.save(
            model,
            self._filename,
        )

        self._best_loss = validation_loss
        self._best_epoch = epoch
        self._best_path = path

        return True