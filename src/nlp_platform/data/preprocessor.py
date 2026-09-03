from collections.abc import Sequence

from nlp_platform.data.cleaning import TextCleaningPipeline
from nlp_platform.data.records import TextRecord


class TextPreprocessor:
    """Apply text cleaning to a collection of text records."""

    def __init__(self, pipeline: TextCleaningPipeline) -> None:
        self._pipeline = pipeline

    def process(self, record: TextRecord) -> TextRecord:
        """Process a single text record."""

        return TextRecord(
            text=self._pipeline.transform(record.text),
            label=record.label,
        )

    def process_many(
        self,
        records: Sequence[TextRecord],
    ) -> list[TextRecord]:
        """Process multiple text records."""

        return [
            self.process(record)
            for record in records
        ]