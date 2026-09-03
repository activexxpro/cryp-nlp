from collections.abc import Sequence

from nlp_platform.data.duplicates import remove_duplicates
from nlp_platform.data.length_filter import TextLengthFilter
from nlp_platform.data.preprocessing_config import PreprocessingConfig
from nlp_platform.data.preprocessor import TextPreprocessor
from nlp_platform.data.records import TextRecord


class DatasetPreprocessingPipeline:
    """Run preprocessing and dataset-quality filtering."""

    def __init__(
        self,
        config: PreprocessingConfig | None = None,
        min_length: int = 1,
        max_length: int | None = None,
        remove_duplicate_records: bool = True,
    ) -> None:
        self._preprocessor = TextPreprocessor(config)
        self._length_filter = TextLengthFilter(
            min_length=min_length,
            max_length=max_length,
        )
        self._remove_duplicate_records = remove_duplicate_records

    def process(
        self,
        records: Sequence[TextRecord],
    ) -> list[TextRecord]:
        """Preprocess and filter records."""

        processed = self._preprocessor.process_many(records)

        processed = self._length_filter.filter(processed)

        if self._remove_duplicate_records:
            processed = remove_duplicates(processed)

        return processed