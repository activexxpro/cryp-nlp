from pathlib import Path

import torch

from nlp_platform.ml.transformer_classifier import TransformerClassifier


class TransformerCheckpointManager:
    """Save and load transformer classifier checkpoints."""

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        classifier: TransformerClassifier,
        filename: str = "transformer_model.pt",
    ) -> Path:
        """Save transformer model weights."""
        path = self._directory / filename
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            classifier.model.state_dict(),
            path,
        )

        return path

    def load(
        self,
        classifier: TransformerClassifier,
        filename: str = "transformer_model.pt",
    ) -> Path:
        """Load transformer model weights."""
        path = self._directory / filename

        if not path.exists():
            raise FileNotFoundError(
                f"Transformer checkpoint not found: {path}",
            )

        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        classifier.model.load_state_dict(state_dict)

        return path

    def exists(
        self,
        filename: str = "transformer_model.pt",
    ) -> bool:
        """Return whether a checkpoint exists."""
        return (self._directory / filename).is_file()

    @property
    def directory(self) -> Path:
        """Return the checkpoint directory."""
        return self._directory