from pathlib import Path

import torch
from torch import nn


class CheckpointManager:
    """Save and load PyTorch model checkpoints."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    def save(
        self,
        model: nn.Module,
        path: str | Path,
    ) -> Path:
        """Save model parameters to a checkpoint file."""

        checkpoint_path = Path(path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = self._directory / checkpoint_path

        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            model.state_dict(),
            checkpoint_path,
        )

        return checkpoint_path

    def load(
        self,
        model: nn.Module,
        path: str | Path,
        device: torch.device | None = None,
    ) -> nn.Module:
        """Load model parameters from a checkpoint."""

        checkpoint_path = Path(path)

        if not checkpoint_path.is_absolute():
            checkpoint_path = self._directory / checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint does not exist: {checkpoint_path}"
            )

        load_kwargs: dict[str, object] = {}

        if device is not None:
            load_kwargs["map_location"] = device

        state_dict = torch.load(
            checkpoint_path,
            **load_kwargs,
        )

        model.load_state_dict(state_dict)

        return model