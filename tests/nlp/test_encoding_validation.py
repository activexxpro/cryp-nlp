import pytest

from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import (
    EncodedDataset,
    DatasetEncodingService,
)
from nlp_platform.nlp.encoding_validation import (
    EncodingValidationError,
    validate_batch_dimensions,
    validate_encoded_dataset,
)
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_dataset() -> EncodedDataset:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
        ]
    )

    tokenizer = BasicTokenizer()
    token_encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )

    batch_encoder = BatchEncoder(
        tokenizer=tokenizer,
        encoder=token_encoder,
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

    return service.encode(
        [
            TextRecord("hello world", 0),
            TextRecord("hello nlp", 1),
        ]
    )


def test_valid_encoded_dataset() -> None:
    dataset = create_dataset()

    validate_encoded_dataset(
        dataset,
        expected_sequence_length=4,
    )


def test_empty_encoded_dataset_is_valid() -> None:
    dataset = EncodedDataset(inputs=())

    validate_encoded_dataset(dataset)


def test_invalid_expected_sequence_length() -> None:
    dataset = create_dataset()

    with pytest.raises(
        EncodingValidationError,
        match="expected length",
    ):
        validate_encoded_dataset(
            dataset,
            expected_sequence_length=8,
        )


def test_invalid_expected_sequence_length_value() -> None:
    dataset = create_dataset()

    with pytest.raises(
        ValueError,
        match="expected_sequence_length must be at least 1",
    ):
        validate_encoded_dataset(
            dataset,
            expected_sequence_length=0,
        )


def test_validate_batch_dimensions() -> None:
    validate_batch_dimensions(
        input_ids=[
            [1, 2, 3],
            [4, 5, 6],
        ],
        labels=[0, 1],
    )


def test_mismatched_batch_size() -> None:
    with pytest.raises(
        EncodingValidationError,
        match="number of inputs must match",
    ):
        validate_batch_dimensions(
            input_ids=[
                [1, 2, 3],
                [4, 5, 6],
            ],
            labels=[0],
        )


def test_inconsistent_sequence_lengths() -> None:
    with pytest.raises(
        EncodingValidationError,
        match="same length",
    ):
        validate_batch_dimensions(
            input_ids=[
                [1, 2],
                [3, 4, 5],
            ],
            labels=[0, 1],
        )


def test_empty_batch_is_valid() -> None:
    validate_batch_dimensions(
        input_ids=[],
        labels=[],
    )