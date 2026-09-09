from nlp_platform.data.dataset_splitter import DatasetSplits
from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import DatasetEncodingService
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.training_dataset_encoder import (
    TrainingDatasetEncoder,
)
from nlp_platform.nlp.vocabulary import Vocabulary


def create_encoder() -> TrainingDatasetEncoder:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
            ["machine", "learning"],
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

    return TrainingDatasetEncoder(
        encoding_service=service,
    )


def test_encode_all_dataset_splits() -> None:
    encoder = create_encoder()

    splits = DatasetSplits(
        train=[
            TextRecord("hello world", 0),
            TextRecord("hello nlp", 1),
        ],
        validation=[
            TextRecord("hello", 0),
        ],
        test=[
            TextRecord("machine learning", 1),
        ],
    )

    result = encoder.encode(splits)

    assert result.train_size == 2
    assert result.validation_size == 1
    assert result.test_size == 1
    assert result.total_size == 4


def test_train_split_is_encoded() -> None:
    encoder = create_encoder()

    splits = DatasetSplits(
        train=[
            TextRecord("hello world", 0),
        ],
        validation=[],
        test=[],
    )

    result = encoder.encode(splits)

    assert result.train.input_ids == (
        (2, 6, 0, 0),
    )
    assert result.train.labels == (0,)


def test_validation_split_is_encoded() -> None:
    encoder = create_encoder()

    splits = DatasetSplits(
        train=[],
        validation=[
            TextRecord("hello nlp", 1),
        ],
        test=[],
    )

    result = encoder.encode(splits)

    assert result.validation.input_ids == (
        (2, 5, 0, 0),
    )
    assert result.validation.labels == (1,)


def test_test_split_is_encoded() -> None:
    encoder = create_encoder()

    splits = DatasetSplits(
        train=[],
        validation=[],
        test=[
            TextRecord("machine learning", 1),
        ],
    )

    result = encoder.encode(splits)

    assert result.test.input_ids == (
        (4, 3, 0, 0),
    )
    assert result.test.labels == (1,)


def test_empty_splits_are_supported() -> None:
    encoder = create_encoder()

    splits = DatasetSplits(
        train=[],
        validation=[],
        test=[],
    )

    result = encoder.encode(splits)

    assert result.train_size == 0
    assert result.validation_size == 0
    assert result.test_size == 0
    assert result.total_size == 0


def test_split_order_is_preserved() -> None:
    encoder = create_encoder()

    splits = DatasetSplits(
        train=[
            TextRecord("hello nlp", 1),
            TextRecord("hello world", 0),
        ],
        validation=[],
        test=[],
    )

    result = encoder.encode(splits)

    assert result.train.labels == (1, 0)

    assert result.train.input_ids == (
        (2, 5, 0, 0),
        (2, 6, 0, 0),
    )