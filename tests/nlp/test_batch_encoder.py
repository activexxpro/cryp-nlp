from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenization_config import TokenizationConfig
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.vocabulary import Vocabulary


def create_batch_encoder() -> BatchEncoder:
    token_sequences = [
        ["hello", "world"],
        ["hello", "nlp"],
        ["machine", "learning"],
    ]

    vocabulary = Vocabulary.build(token_sequences)

    tokenizer = BasicTokenizer(
        config=TokenizationConfig(
            lowercase=True,
        )
    )

    encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )

    return BatchEncoder(
        tokenizer=tokenizer,
        encoder=encoder,
        padder=padder,
    )


def test_encode_text() -> None:
    batch_encoder = create_batch_encoder()

    result = batch_encoder.encode("Hello world")

    assert result == [2, 6, 0, 0]


def test_encode_text_with_unknown_token() -> None:
    batch_encoder = create_batch_encoder()

    result = batch_encoder.encode("hello unknown")

    assert result == [2, 1, 0, 0]


def test_encode_long_text_is_truncated() -> None:
    batch_encoder = create_batch_encoder()

    result = batch_encoder.encode(
        "hello world machine learning"
    )

    assert len(result) == 4


def test_encode_empty_text() -> None:
    batch_encoder = create_batch_encoder()

    result = batch_encoder.encode("")

    assert result == [0, 0, 0, 0]


def test_encode_many() -> None:
    batch_encoder = create_batch_encoder()

    result = batch_encoder.encode_many(
        [
            "hello world",
            "hello nlp",
        ]
    )

    assert result == [
        [2, 6, 0, 0],
        [2, 5, 0, 0],
    ]


def test_encode_many_empty_input() -> None:
    batch_encoder = create_batch_encoder()

    assert batch_encoder.encode_many([]) == []


def test_output_sequences_have_equal_length() -> None:
    batch_encoder = create_batch_encoder()

    result = batch_encoder.encode_many(
        [
            "hello",
            "hello world",
            "hello world machine learning",
        ]
    )

    assert all(
        len(sequence) == 4
        for sequence in result
    )