from collections.abc import Sequence

from nlp_platform.data.preprocessing import (
    DatasetPreprocessingPipeline,
)
from nlp_platform.data.records import TextRecord
from nlp_platform.data.validation import validate_records


class DatasetPreparationService:
    """Prepare text records for downstream NLP workflows."""

    def __init__(
        self,
        preprocessing_pipeline: DatasetPreprocessingPipeline | None = None,
    ) -> None:
        self._pipeline = (
            preprocessing_pipeline
            or DatasetPreprocessingPipeline()
        )

    def prepare(
        self,
        records: Sequence[TextRecord],
    ) -> list[TextRecord]:
        """Preprocess and validate a dataset."""

        validate_records(records)

        prepared_records = self._pipeline.process(records)

        validate_records(prepared_records)

        return prepared_records