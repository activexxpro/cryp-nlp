from dataclasses import dataclass

from nlp_platform.data.records import TextRecord
from nlp_platform.data.statistics import (
    RecordStatistics,
    calculate_statistics,
)


@dataclass(frozen=True)
class DataQualityReport:
    """Quality summary for a collection of text records."""

    statistics: RecordStatistics
    empty_text_count: int
    negative_label_count: int

    @property
    def is_valid(self) -> bool:
        """Return whether the dataset passes basic quality checks."""

        return (
            self.empty_text_count == 0
            and self.negative_label_count == 0
        )


def generate_quality_report(
    records: list[TextRecord],
) -> DataQualityReport:
    """Generate a quality report for text records."""

    empty_text_count = sum(
        not record.text.strip()
        for record in records
    )

    negative_label_count = sum(
        record.label < 0
        for record in records
    )

    statistics = calculate_statistics(records)

    return DataQualityReport(
        statistics=statistics,
        empty_text_count=empty_text_count,
        negative_label_count=negative_label_count,
    )