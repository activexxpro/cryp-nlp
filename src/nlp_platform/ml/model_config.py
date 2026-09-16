from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a text classification model."""

    vocab_size: int
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 4
    padding_idx: int = 0

    def __post_init__(self) -> None:
        if self.vocab_size < 2:
            raise ValueError(
                "vocab_size must be at least 2"
            )

        if self.embedding_dim < 1:
            raise ValueError(
                "embedding_dim must be at least 1"
            )

        if self.hidden_dim < 1:
            raise ValueError(
                "hidden_dim must be at least 1"
            )

        if self.num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2"
            )

        if self.padding_idx < 0:
            raise ValueError(
                "padding_idx must not be negative"
            )

        if self.padding_idx >= self.vocab_size:
            raise ValueError(
                "padding_idx must be smaller than vocab_size"
            )