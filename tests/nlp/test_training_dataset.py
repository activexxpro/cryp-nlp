from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import (
    DatasetEncodingService,
    EncodedDataset,
)
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.training_dataset import TrainingDataset
from nlp_platform.nlp.vocabulary import Vocabulary


def create_encoded_dataset(
    records: list[TextRecord],
) -> EncodedDataset:
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


def test_training_dataset_sizes() -> None:
    dataset = TrainingDataset(
        train=create_encoded_dataset(
            [
                TextRecord("hello world", 0),
                TextRecord("hello nlp", 1),
            ]
        ),
        validation=create_encoded_dataset(
            [
                TextRecord("hello", 0),
            ]
        ),
        test=create_encoded_dataset(
            [
                TextRecord("machine learning", 1),
            ]
        ),
    )

    assert dataset.train_size == 2
    assert dataset.validation_size == 1
    assert dataset.test_size == 1
    assert dataset.total_size == 4


def test_training_dataset_preserves_datasets() -> None:
    train = create_encoded_dataset(
        [TextRecord("hello world", 0)]
    )
    validation = create_encoded_dataset(
        [TextRecord("hello nlp", 1)]
    )
    test = create_encoded_dataset(
        [TextRecord("hello", 0)]
    )

    dataset = TrainingDataset(
        train=train,
        validation=validation,
        test=test,
    )

    assert dataset.train is train
    assert dataset.validation is validation
    assert dataset.test is test


def test_empty_training_dataset() -> None:
    empty = EncodedDataset(inputs=())

    dataset = TrainingDataset(
        train=empty,
        validation=empty,
        test=empty,
    )

    assert dataset.train_size == 0
    assert dataset.validation_size == 0
    assert dataset.test_size == 0
    assert dataset.total_size == 0


def test_total_size_is_sum_of_splits() -> None:
    train = create_encoded_dataset(
        [
            TextRecord("hello", 0),
            TextRecord("world", 1),
            TextRecord("hello nlp", 1),
        ]
    )

    validation = create_encoded_dataset(
        [
            TextRecord("hello world", 0),
            TextRecord("machine learning", 1),
        ]
    )

    test = create_encoded_dataset(
        [
            TextRecord("hello", 0),
        ]
    )

    dataset = TrainingDataset(
        train=train,
        validation=validation,
        test=test,
    )

    assert dataset.total_size == (
        dataset.train_size
        + dataset.validation_size
        + dataset.test_size
    )