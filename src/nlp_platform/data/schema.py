from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetStats:
    train_samples: int
    test_samples: int
    num_classes: int