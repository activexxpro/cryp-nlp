from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import (
    DatasetEncodingService,
)
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_service() -> DatasetEncodingService:
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

    return DatasetEncodingService(
        encoder=dataset_encoder,
    )


def test_encode_dataset() -> None:
    service = create_service()

    records = [
        TextRecord("hello world", 0),
        TextRecord("hello nlp", 1),
    ]

    result = service.encode(records)

    assert result.size == 2
    assert result.input_ids == (
        (2, 6, 0, 0),
        (2, 5, 0, 0),
    )
    assert result.labels == (0, 1)


def test_encoded_dataset_preserves_order() -> None:
    service = create_service()

    records = [
        TextRecord("hello nlp", 1),
        TextRecord("hello world", 0),
    ]

    result = service.encode(records)

    assert result.labels == (1, 0)

    assert result.input_ids == (
        (2, 5, 0, 0),
        (2, 6, 0, 0),
    )


def test_empty_dataset() -> None:
    service = create_service()

    result = service.encode([])

    assert result.size == 0
    assert result.input_ids == ()
    assert result.labels == ()


def test_encoded_dataset_stores_immutable_inputs() -> None:
    service = create_service()

    records = [
        TextRecord("hello world", 0),
    ]

    result = service.encode(records)

    assert isinstance(result.inputs, tuple)
    assert isinstance(result.inputs[0].input_ids, tuple)


def test_dataset_size_matches_inputs() -> None:
    service = create_service()

    records = [
        TextRecord("hello", 0),
        TextRecord("world", 1),
        TextRecord("hello nlp", 1),
    ]

    result = service.encode(records)

    assert result.size == len(result.inputs)
    assert result.size == 3