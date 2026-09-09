import pytest

from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import DatasetEncodingService
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.training_dataset import TrainingDataset
from nlp_platform.nlp.training_validation import (
    TrainingDatasetValidationError,
    validate_training_dataset,
)
from nlp_platform.nlp.vocabulary import Vocabulary


def create_dataset(
    records: list[TextRecord],
):
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
            ["machine", "learning"],
        ]
    )

    tokenizer = BasicTokenizer()
    encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )

    batch_encoder = BatchEncoder(
        tokenizer=tokenizer,
        encoder=encoder,
        padder=padder,
    )

    pipeline = NLPPreprocessingPipeline(
        batch_encoder=batch_encoder,
    )

    dataset_encoder = DatasetEncoder(
        pipeline=pipeline,
    )

    service = DatasetEncodingService(
        encoder=dataset_encoder,
    )

    return service.encode(records)


def create_valid_training_dataset() -> TrainingDataset:
    return TrainingDataset(
        train=create_dataset(
            [
                TextRecord("hello world", 0),
                TextRecord("hello nlp", 1),
                TextRecord("machine learning", 1),
                TextRecord("hello", 0),
            ]
        ),
        validation=create_dataset(
            [
                TextRecord("hello world", 0),
                TextRecord("machine learning", 1),
            ]
        ),
        test=create_dataset(
            [
                TextRecord("hello nlp", 1),
                TextRecord("hello", 0),
            ]
        ),
    )


def test_valid_training_dataset() -> None:
    dataset = create_valid_training_dataset()

    validate_training_dataset(
        dataset,
        expected_sequence_length=4,
    )


def test_empty_training_split_is_rejected() -> None:
    dataset = TrainingDataset(
        train=create_dataset(
            [
                TextRecord("hello world", 0),
            ]
        ),
        validation=create_dataset([]),
        test=create_dataset(
            [
                TextRecord("hello", 0),
            ]
        ),
    )

    with pytest.raises(
        TrainingDatasetValidationError,
        match="validation split must not be empty",
    ):
        validate_training_dataset(dataset)


def test_empty_test_split_is_rejected() -> None:
    dataset = TrainingDataset(
        train=create_dataset(
            [
                TextRecord("hello world", 0),
            ]
        ),
        validation=create_dataset(
            [
                TextRecord("hello", 0),
            ]
        ),
        test=create_dataset([]),
    )

    with pytest.raises(
        TrainingDatasetValidationError,
        match="test split must not be empty",
    ):
        validate_training_dataset(dataset)


def test_invalid_expected_sequence_length() -> None:
    dataset = create_valid_training_dataset()

    with pytest.raises(
        ValueError,
        match="expected_sequence_length must be at least 1",
    ):
        validate_training_dataset(
            dataset,
            expected_sequence_length=0,
        )


def test_wrong_sequence_length_is_rejected() -> None:
    dataset = create_valid_training_dataset()

    with pytest.raises(
        TrainingDatasetValidationError,
        match="train split contains an invalid sequence length",
    ):
        validate_training_dataset(
            dataset,
            expected_sequence_length=8,
        )


def test_validation_labels_must_exist_in_training() -> None:
    dataset = TrainingDataset(
        train=create_dataset(
            [
                TextRecord("hello world", 0),
            ]
        ),
        validation=create_dataset(
            [
                TextRecord("hello nlp", 1),
            ]
        ),
        test=create_dataset(
            [
                TextRecord("hello", 0),
            ]
        ),
    )

    with pytest.raises(
        TrainingDatasetValidationError,
        match="validation split contains labels",
    ):
        validate_training_dataset(dataset)


def test_test_labels_must_exist_in_training() -> None:
    dataset = TrainingDataset(
        train=create_dataset(
            [
                TextRecord("hello world", 0),
            ]
        ),
        validation=create_dataset(
            [
                TextRecord("hello", 0),
            ]
        ),
        test=create_dataset(
            [
                TextRecord("hello nlp", 1),
            ]
        ),
    )

    with pytest.raises(
        TrainingDatasetValidationError,
        match="test split contains labels",
    ):
        validate_training_dataset(dataset)


def test_all_splits_are_validated() -> None:
    dataset = create_valid_training_dataset()

    validate_training_dataset(dataset)

    assert dataset.train_size > 0
    assert dataset.validation_size > 0
    assert dataset.test_size > 0