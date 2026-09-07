import pytest

from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.vocabulary import Vocabulary


@pytest.fixture
def padder() -> SequencePadder:
    vocabulary = Vocabulary.build(
        [
            ["hello", "world"],
            ["hello", "nlp"],
        ]
    )

    return SequencePadder(
        vocabulary=vocabulary,
        max_length=4,
    )


def test_short_sequence_is_padded(
    padder: SequencePadder,
) -> None:
    result = padder.pad_or_truncate([2, 4])

    assert result == [2, 4, 0, 0]


def test_exact_length_sequence_is_unchanged(
    padder: SequencePadder,
) -> None:
    result = padder.pad_or_truncate([2, 4, 3, 2])

    assert result == [2, 4, 3, 2]


def test_long_sequence_is_truncated(
    padder: SequencePadder,
) -> None:
    result = padder.pad_or_truncate(
        [2, 4, 3, 2, 99, 100]
    )

    assert result == [2, 4, 3, 2]


def test_empty_sequence_is_fully_padded(
    padder: SequencePadder,
) -> None:
    result = padder.pad_or_truncate([])

    assert result == [0, 0, 0, 0]


def test_padding_uses_pad_token_id(
    padder: SequencePadder,
) -> None:
    result = padder.pad_or_truncate([2])

    assert result[-1] == padder._vocabulary.token_to_id(
        Vocabulary.PAD_TOKEN
    )


def test_pad_or_truncate_many(
    padder: SequencePadder,
) -> None:
    result = padder.pad_or_truncate_many(
        [
            [2],
            [2, 4],
            [2, 4, 3, 2, 99],
        ]
    )

    assert result == [
        [2, 0, 0, 0],
        [2, 4, 0, 0],
        [2, 4, 3, 2],
    ]


def test_invalid_max_length() -> None:
    vocabulary = Vocabulary.build([["hello"]])

    with pytest.raises(
        ValueError,
        match="max_length must be at least 1",
    ):
        SequencePadder(
            vocabulary=vocabulary,
            max_length=0,
        )


def test_input_sequence_is_not_modified(
    padder: SequencePadder,
) -> None:
    token_ids = [2, 4]

    padder.pad_or_truncate(token_ids)

    assert token_ids == [2, 4]