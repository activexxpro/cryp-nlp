from collections import Counter
from dataclasses import dataclass

from nlp_platform.nlp.encoded_dataset import EncodedDataset


@dataclass(frozen=True)
class EncodingStatistics:
    """Statistics describing an encoded dataset."""

    total_samples: int
    sequence_length: int
    num_classes: int
    label_counts: dict[int, int]

    @property
    def majority_class(self) -> int | None:
        """Return the most frequent class."""

        if not self.label_counts:
            return None

        return max(
            self.label_counts,
            key=self.label_counts.get,
        )

    @property
    def minority_class(self) -> int | None:
        """Return the least frequent class."""

        if not self.label_counts:
            return None

        return min(
            self.label_counts,
            key=self.label_counts.get,
        )


def calculate_encoding_statistics(
    dataset: EncodedDataset,
) -> EncodingStatistics:
    """Calculate statistics for an encoded dataset."""

    if dataset.size == 0:
        return EncodingStatistics(
            total_samples=0,
            sequence_length=0,
            num_classes=0,
            label_counts={},
        )

    sequence_lengths = {
        len(item.input_ids)
        for item in dataset.inputs
    }

    if len(sequence_lengths) != 1:
        raise ValueError(
            "all encoded sequences must have the same length"
        )

    label_counts = Counter(dataset.labels)

    return EncodingStatistics(
        total_samples=dataset.size,
        sequence_length=next(iter(sequence_lengths)),
        num_classes=len(label_counts),
        label_counts=dict(
            sorted(label_counts.items())
        ),
    )