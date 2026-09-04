from collections.abc import Sequence

from nlp_platform.data.dataset_splitter import (
    DatasetSplits,
    split_dataset,
)
from nlp_platform.data.metadata import DatasetMetadata
from nlp_platform.data.prepared_dataset import PreparedDataset
from nlp_platform.data.preprocessing import DatasetPreprocessingPipeline
from nlp_platform.data.records import TextRecord
from nlp_platform.data.split_config import SplitConfig
from nlp_platform.data.validation import validate_records


class DatasetPreparationPipeline:
    """Prepare a dataset for model training and evaluation."""

    def __init__(
        self,
        preprocessing_pipeline: DatasetPreprocessingPipeline | None = None,
        split_config: SplitConfig | None = None,
    ) -> None:
        self._preprocessing_pipeline = (
            preprocessing_pipeline
            or DatasetPreprocessingPipeline()
        )
        self._split_config = split_config or SplitConfig()

    def prepare(
        self,
        records: Sequence[TextRecord],
        name: str = "dataset",
        version: str = "1.0.0",
        source: str = "unknown",
    ) -> DatasetSplits:
        """Preprocess and split records."""

        validate_records(records)

        processed_records = self._preprocessing_pipeline.process(
            records
        )

        validate_records(processed_records)

        return split_dataset(
            processed_records,
            config=self._split_config,
        )

    def create_metadata(
        self,
        records: Sequence[TextRecord],
        name: str = "dataset",
        version: str = "1.0.0",
        source: str = "unknown",
    ) -> DatasetMetadata:
        """Create metadata for prepared records."""

        labels = {
            record.label
            for record in records
        }

        return DatasetMetadata.create(
            name=name,
            version=version,
            source=source,
            num_records=len(records),
            num_classes=len(labels),
        )

    def build(
        self,
        records: Sequence[TextRecord],
        name: str = "dataset",
        version: str = "1.0.0",
        source: str = "unknown",
    ) -> PreparedDataset:
        """Build a prepared dataset from records."""

        validate_records(records)

        processed_records = self._preprocessing_pipeline.process(
            records
        )

        validate_records(processed_records)

        metadata = self.create_metadata(
            processed_records,
            name=name,
            version=version,
            source=source,
        )

        return PreparedDataset.create(
            processed_records,
            metadata,
        )