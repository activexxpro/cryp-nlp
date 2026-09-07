from dataclasses import dataclass
from collections.abc import Sequence

from nlp_platform.data.distribution import (
    ClassDistribution,
    calculate_class_distribution,
)
from nlp_platform.data.metadata import DatasetMetadata
from nlp_platform.data.records import TextRecord


@dataclass(frozen=True)
class PreparedDataset:
    """Prepared dataset with metadata and distribution information."""

    records: tuple[TextRecord, ...]
    metadata: DatasetMetadata
    distribution: ClassDistribution

    @classmethod
    def create(
        cls,
        records: Sequence[TextRecord],
        metadata: DatasetMetadata,
    ) -> "PreparedDataset":
        """Create a prepared dataset from text records."""

        record_tuple = tuple(records)

        if metadata.num_records != len(record_tuple):
            raise ValueError(
                "metadata.num_records does not match "
                "the number of records"
            )

        distribution = calculate_class_distribution(
            record_tuple
        )

        if metadata.num_classes != distribution.num_classes:
            raise ValueError(
                "metadata.num_classes does not match "
                "the observed number of classes"
            )

        return cls(
            records=record_tuple,
            metadata=metadata,
            distribution=distribution,
        )

    @property
    def size(self) -> int:
        """Return the number of records."""

        return len(self.records)

    @property
    def labels(self) -> tuple[int, ...]:
        """Return labels in dataset order."""

        return tuple(
            record.label
            for record in self.records
        )

    @property
    def texts(self) -> tuple[str, ...]:
        """Return texts in dataset order."""

        return tuple(
            record.text
            for record in self.records
        )