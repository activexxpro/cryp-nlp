from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_encoder() -> DatasetEncoder:
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

    return DatasetEncoder(pipeline)


def test_encode_record() -> None:
    encoder = create_encoder()

    record = TextRecord(
        text="Hello world",
        label=1,
    )

    result = encoder.encode(record)

    assert result.input_ids == (2, 4, 0, 0)
    assert result.label == 1
    assert result.sequence_length == 4


def test_encode_record_pads_input() -> None:
    encoder = create_encoder()

    record = TextRecord(
        text="hello",
        label=0,
    )

    result = encoder.encode(record)

    assert result.input_ids == (2, 0, 0, 0)


def test_encode_unknown_token() -> None:
    encoder = create_encoder()

    record = TextRecord(
        text="hello unknown",
        label=1,
    )

    result = encoder.encode(record)

    assert result.input_ids == (2, 1, 0, 0)


def test_encode_many() -> None:
    encoder = create_encoder()

    records = [
        TextRecord(
            text="hello world",
            label=0,
        ),
        TextRecord(
            text="hello nlp",
            label=1,
        ),
    ]

    result = encoder.encode_many(records)

    assert result == [
        result[0],
        result[1],
    ]

    assert result[0].input_ids == (2, 4, 0, 0)
    assert result[0].label == 0

    assert result[1].input_ids == (2, 3, 0, 0)
    assert result[1].label == 1


def test_encode_many_preserves_order() -> None:
    encoder = create_encoder()

    records = [
        TextRecord("hello nlp", 1),
        TextRecord("hello world", 0),
    ]

    result = encoder.encode_many(records)

    assert result[0].label == 1
    assert result[1].label == 0

    assert result[0].input_ids == (2, 3, 0, 0)
    assert result[1].input_ids == (2, 4, 0, 0)


def test_encode_empty_dataset() -> None:
    encoder = create_encoder()

    assert encoder.encode_many([]) == []