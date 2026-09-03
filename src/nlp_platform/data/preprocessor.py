from collections.abc import Sequence

from nlp_platform.data.cleaning import (
    TextCleaningPipeline,
    lowercase_text,
)
from nlp_platform.data.normalizer import normalize_text
from nlp_platform.data.preprocessing_config import PreprocessingConfig
from nlp_platform.data.records import TextRecord


class TextPreprocessor:
    """Apply configurable text preprocessing."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
    ) -> None:
        self._config = config or PreprocessingConfig()
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> TextCleaningPipeline:
        transforms = []

        if self._config.strip_whitespace:
            transforms.append(str.strip)

        if self._config.normalize_whitespace:
            transforms.append(normalize_text)

        if self._config.lowercase:
            transforms.append(lowercase_text)

        return TextCleaningPipeline(transforms=transforms)

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