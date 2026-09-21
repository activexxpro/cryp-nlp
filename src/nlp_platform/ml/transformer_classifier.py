from typing import Any

import torch
from transformers import AutoModelForSequenceClassification

from nlp_platform.ml.transformer_config import TransformerConfig


class TransformerClassifier:
    """Adapter around a Hugging Face sequence classification model."""

    def __init__(
        self,
        model: Any,
        config: TransformerConfig | None = None,
    ) -> None:
        self._model = model
        self._config = config or TransformerConfig()

    @classmethod
    def from_pretrained(
        cls,
        config: TransformerConfig | None = None,
    ) -> "TransformerClassifier":
        """Create a classifier from a pretrained Hugging Face model."""
        resolved_config = config or TransformerConfig()

        model = AutoModelForSequenceClassification.from_pretrained(
            resolved_config.model_name,
            num_labels=resolved_config.num_labels,
        )

        return cls(
            model=model,
            config=resolved_config,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> Any:
        """Run the transformer model."""
        return self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def parameters(self) -> Any:
        """Return trainable model parameters."""
        return self._model.parameters()

    def train(self) -> "TransformerClassifier":
        """Set the underlying model to training mode."""
        self._model.train()
        return self

    def eval(self) -> "TransformerClassifier":
        """Set the underlying model to evaluation mode."""
        self._model.eval()
        return self

    def to(self, device: torch.device) -> "TransformerClassifier":
        """Move the underlying model to a device."""
        self._model.to(device)
        return self

    @property
    def model(self) -> Any:
        """Return the underlying Hugging Face model."""
        return self._model

    @property
    def config(self) -> TransformerConfig:
        """Return the transformer configuration."""
        return self._config