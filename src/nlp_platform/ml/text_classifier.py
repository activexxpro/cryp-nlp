import torch
from torch import nn

from nlp_platform.ml.model_config import ModelConfig


class TextClassifier(nn.Module):
    """Embedding-based neural network for text classification."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.embedding_dim,
            padding_idx=config.padding_idx,
        )

        self.classifier = nn.Sequential(
            nn.Linear(
                config.embedding_dim,
                config.hidden_dim,
            ),
            nn.ReLU(),
            nn.Linear(
                config.hidden_dim,
                config.num_classes,
            ),
        )

        self.padding_idx = config.padding_idx

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass and return class logits."""

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be a two-dimensional tensor"
            )

        embeddings = self.embedding(input_ids)

        mask = (
            input_ids != self.padding_idx
        ).unsqueeze(-1)

        masked_embeddings = embeddings * mask

        token_counts = mask.sum(
            dim=1
        ).clamp(min=1)

        pooled = (
            masked_embeddings.sum(dim=1)
            / token_counts
        )

        return self.classifier(pooled)