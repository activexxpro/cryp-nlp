from dataclasses import dataclass


@dataclass(frozen=True)
class TokenizationConfig:
    """Configuration for text tokenization."""

    lowercase: bool = True
    min_token_length: int = 1

    def __post_init__(self) -> None:
        if self.min_token_length < 1:
            raise ValueError(
                "min_token_length must be at least 1"
            )