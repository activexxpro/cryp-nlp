from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from nlp_platform.data.records import TextRecord


@dataclass(frozen=True)
class ClassDistribution:
    """Distribution statistics for dataset labels."""

    counts: dict[int, int]
    total_records: int

    @property
    def num_classes(self) -> int:
        """Return the number of observed classes."""
        return len(self.counts)

    @property
    def majority_class(self) -> int | None:
        """Return the most frequent class."""

        if not self.counts:
            return None

        return max(
            self.counts,
            key=self.counts.get,
        )

    @property
    def minority_class(self) -> int | None:
        """Return the least frequent class."""

        if not self.counts:
            return None

        return min(
            self.counts,
            key=self.counts.get,
        )

    @property
    def imbalance_ratio(self) -> float:
        """Return majority-to-minority class ratio."""

        if not self.counts:
            return 0.0

        minimum = min(self.counts.values())

        if minimum == 0:
            return float("inf")

        return max(self.counts.values()) / minimum


def calculate_class_distribution(
    records: Sequence[TextRecord],
) -> ClassDistribution:
    """Calculate label distribution for records."""

    counts = Counter(
        record.label
        for record in records
    )

    return ClassDistribution(
        counts=dict(sorted(counts.items())),
        total_records=len(records),
    )