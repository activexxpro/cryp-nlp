from dataclasses import dataclass

from nlp_platform.nlp.encoded_dataset import EncodedDataset


@dataclass(frozen=True)
class TrainingDataset:
    """Container for train, validation, and test datasets."""

    train: EncodedDataset
    validation: EncodedDataset
    test: EncodedDataset

    @property
    def train_size(self) -> int:
        """Return the number of training samples."""

        return self.train.size

    @property
    def validation_size(self) -> int:
        """Return the number of validation samples."""

        return self.validation.size

    @property
    def test_size(self) -> int:
        """Return the number of test samples."""

        return self.test.size

    @property
    def total_size(self) -> int:
        """Return the total number of samples."""

        return (
            self.train_size
            + self.validation_size
            + self.test_size
        )