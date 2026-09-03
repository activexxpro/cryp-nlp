from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    lowercase: bool = True
    strip_whitespace: bool = True
    normalize_whitespace: bool = True