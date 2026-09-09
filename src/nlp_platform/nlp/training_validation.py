from nlp_platform.nlp.encoding_statistics import (
    calculate_encoding_statistics,
)
from nlp_platform.nlp.encoded_dataset import EncodedDataset
from nlp_platform.nlp.training_dataset import TrainingDataset


class TrainingDatasetValidationError(ValueError):
    """Raised when training dataset validation fails."""


def _validate_split(
    dataset: EncodedDataset,
    split_name: str,
    expected_sequence_length: int | None,
) -> None:
    """Validate an individual dataset split."""

    if dataset.size == 0:
        raise TrainingDatasetValidationError(
            f"{split_name} split must not be empty"
        )

    try:
        calculate_encoding_statistics(dataset)
    except ValueError as exc:
        raise TrainingDatasetValidationError(
            f"{split_name} split is invalid: {exc}"
        ) from exc

    if expected_sequence_length is not None:
        for item in dataset.inputs:
            if len(item.input_ids) != expected_sequence_length:
                raise TrainingDatasetValidationError(
                    f"{split_name} split contains an invalid "
                    "sequence length"
                )


def validate_training_dataset(
    dataset: TrainingDataset,
    expected_sequence_length: int | None = None,
) -> None:
    """Validate all training, validation, and test splits."""

    if expected_sequence_length is not None:
        if expected_sequence_length < 1:
            raise ValueError(
                "expected_sequence_length must be at least 1"
            )

    _validate_split(
        dataset.train,
        "train",
        expected_sequence_length,
    )

    _validate_split(
        dataset.validation,
        "validation",
        expected_sequence_length,
    )

    _validate_split(
        dataset.test,
        "test",
        expected_sequence_length,
    )

    train_classes = set(dataset.train.labels)

    if set(dataset.validation.labels) - train_classes:
        raise TrainingDatasetValidationError(
            "validation split contains labels not present "
            "in the training split"
        )

    if set(dataset.test.labels) - train_classes:
        raise TrainingDatasetValidationError(
            "test split contains labels not present "
            "in the training split"
        )