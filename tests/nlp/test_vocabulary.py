import pytest

from nlp_platform.nlp.vocabulary import Vocabulary


def test_build_vocabulary():
    token_sequences = [
        ["hello", "world"],
        ["hello", "nlp"],
    ]

    vocabulary = Vocabulary.build(
        token_sequences
    )

    assert vocabulary.size == 5


def test_special_tokens_have_reserved_ids():
    vocabulary = Vocabulary.build(
        [["hello"]]
    )

    assert vocabulary.token_to_id(
        Vocabulary.PAD_TOKEN
    ) == 0

    assert vocabulary.token_to_id(
        Vocabulary.UNK_TOKEN
    ) == 1


def test_token_to_id():
    vocabulary = Vocabulary.build(
        [["hello", "world"]]
    )

    assert vocabulary.token_to_id("hello") == 2
    assert vocabulary.token_to_id("world") == 3


def test_unknown_token_returns_unk_id():
    vocabulary = Vocabulary.build(
        [["hello"]]
    )

    assert vocabulary.token_to_id("unknown") == 1


def test_id_to_token():
    vocabulary = Vocabulary.build(
        [["hello", "world"]]
    )

    assert vocabulary.id_to_token(2) == "hello"
    assert vocabulary.id_to_token(3) == "world"


def test_unknown_id_raises_error():
    vocabulary = Vocabulary.build(
        [["hello"]]
    )

    with pytest.raises(
        ValueError,
        match="unknown token ID",
    ):
        vocabulary.id_to_token(999)


def test_contains():
    vocabulary = Vocabulary.build(
        [["hello", "world"]]
    )

    assert vocabulary.contains("hello")
    assert vocabulary.contains("world")
    assert not vocabulary.contains("unknown")


def test_tokens_are_returned_in_id_order():
    vocabulary = Vocabulary.build(
        [["hello", "world", "nlp"]]
    )

    assert vocabulary.tokens() == [
        "<PAD>",
        "<UNK>",
        "hello",
        "nlp",
        "world",
    ]


def test_min_frequency():
    token_sequences = [
        ["hello", "world"],
        ["hello", "nlp"],
        ["hello"],
    ]

    vocabulary = Vocabulary.build(
        token_sequences,
        min_frequency=2,
    )

    assert vocabulary.contains("hello")
    assert not vocabulary.contains("world")
    assert not vocabulary.contains("nlp")


def test_invalid_min_frequency():
    with pytest.raises(
        ValueError,
        match="min_frequency must be at least 1",
    ):
        Vocabulary.build(
            [["hello"]],
            min_frequency=0,
        )


def test_empty_token_sequences():
    vocabulary = Vocabulary.build([])

    assert vocabulary.size == 2
    assert vocabulary.tokens() == [
        "<PAD>",
        "<UNK>",
    ]