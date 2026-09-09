from collections.abc import Sequence

from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.model_input import ModelInput
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline


class DatasetEncoder:
    """Convert text records into model-ready inputs."""

    def __init__(
        self,
        pipeline: NLPPreprocessingPipeline,
    ) -> None:
        self._pipeline = pipeline

    def encode(
        self,
        record: TextRecord,
    ) -> ModelInput:
        """Encode a single text record."""

        input_ids = self._pipeline.process(
            record.text
        )

        return ModelInput(
            input_ids=tuple(input_ids),
            label=record.label,
        )

    def encode_many(
        self,
        records: Sequence[TextRecord],
    ) -> list[ModelInput]:
        """Encode multiple text records."""

        return [
            self.encode(record)
            for record in records
        ]