from nlp_platform.data.dataset_splitter import DatasetSplits
from nlp_platform.nlp.encoded_dataset import DatasetEncodingService
from nlp_platform.nlp.training_dataset import TrainingDataset


class TrainingDatasetEncoder:
    """Encode all dataset splits for model training."""

    def __init__(
        self,
        encoding_service: DatasetEncodingService,
    ) -> None:
        self._encoding_service = encoding_service

    def encode(
        self,
        splits: DatasetSplits,
    ) -> TrainingDataset:
        """Encode train, validation, and test splits."""

        return TrainingDataset(
            train=self._encoding_service.encode(
                splits.train
            ),
            validation=self._encoding_service.encode(
                splits.validation
            ),
            test=self._encoding_service.encode(
                splits.test
            ),
        )