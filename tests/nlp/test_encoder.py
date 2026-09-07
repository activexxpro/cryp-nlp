import pytest

from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.vocabulary import Vocabulary


@pytest.fixture
def encoder() -> TokenEncoder:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
        ]
    )

    return TokenEncoder(vocabulary)


def test_encode_tokens(encoder: TokenEncoder) -> None:
    encoded = encoder.encode(["hello", "world"])

    assert encoded == [2, 4]


def test_encode_unknown_token(encoder: TokenEncoder) -> None:
    encoded = encoder.encode(["unknown"])

    assert encoded == [1]


def test_encode_empty_sequence(encoder: TokenEncoder) -> None:
    assert encoder.encode([]) == []


def test_encode_many(encoder: TokenEncoder) -> None:
    encoded = encoder.encode_many(
        [
            ["hello", "world"],
            ["hello", "nlp"],
        ]
    )

    assert encoded == [
        [2, 4],
        [2, 3],
    ]


def test_decode_ids(encoder: TokenEncoder) -> None:
    decoded = encoder.decode([2, 4])

    assert decoded == ["hello", "world"]


def test_decode_many(encoder: TokenEncoder) -> None:
    decoded = encoder.decode_many(
        [
            [2, 4],
            [2, 3],
        ]
    )

    assert decoded == [
        ["hello", "world"],
        ["hello", "nlp"],
    ]


def test_decode_unknown_id_raises_error(
    encoder: TokenEncoder,
) -> None:
    with pytest.raises(ValueError, match="unknown token ID"):
        encoder.decode([999])


def test_encode_decode_round_trip(
    encoder: TokenEncoder,
) -> None:
    tokens = ["hello", "world", "nlp"]

    encoded = encoder.encode(tokens)
    decoded = encoder.decode(encoded)

    assert decoded == tokens