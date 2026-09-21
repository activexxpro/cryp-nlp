from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for a transformer-based NLP model."""

    model_name: str = "distilbert-base-uncased"
    max_length: int = 128
    num_labels: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty")

        if self.max_length < 1:
            raise ValueError("max_length must be at least 1")

        if self.num_labels < 2:
            raise ValueError("num_labels must be at least 2")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")

        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")