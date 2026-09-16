from dataclasses import dataclass


@dataclass(frozen=True)
class DataLoaderConfig:
    """Configuration for PyTorch data loading."""

    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0
    drop_last: bool = False

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1"
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers must not be negative"
            )