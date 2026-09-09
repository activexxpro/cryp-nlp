from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingConfig:
    """Configuration for model input encoding."""

    max_length: int = 128
    padding: bool = True
    truncation: bool = True

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError(
                "max_length must be at least 1"
            )