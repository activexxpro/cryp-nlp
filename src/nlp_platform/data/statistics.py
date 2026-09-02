from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean

from nlp_platform.data.records import TextRecord


@dataclass(frozen=True)
class RecordStatistics:
    """Summary statistics for a collection of text records."""

    total_records: int
    num_classes: int
    label_counts: dict[int, int]
    average_text_length: float
    min_text_length: int
    max_text_length: int


def calculate_statistics(
    records: Sequence[TextRecord],
) -> RecordStatistics:
    """Calculate descriptive statistics for text records."""

    if not records:
        return RecordStatistics(
            total_records=0,
            num_classes=0,
            label_counts={},
            average_text_length=0.0,
            min_text_length=0,
            max_text_length=0,
        )

    label_counts = Counter(record.label for record in records)
    text_lengths = [len(record.text) for record in records]

    return RecordStatistics(
        total_records=len(records),
        num_classes=len(label_counts),
        label_counts=dict(sorted(label_counts.items())),
        average_text_length=mean(text_lengths),
        min_text_length=min(text_lengths),
        max_text_length=max(text_lengths),
    )