from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    """Configuration for dataset splitting."""

    validation_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError(
                "validation_size must be between 0 and 1"
            )

        if not 0.0 < self.test_size < 1.0:
            raise ValueError(
                "test_size must be between 0 and 1"
            )

        if self.validation_size + self.test_size >= 1.0:
            raise ValueError(
                "validation_size + test_size must be less than 1"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must not be negative"
            )