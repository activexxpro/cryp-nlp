import pytest

from nlp_platform.nlp.tokenizer import (
    BasicTokenizer,
    Tokenizer,
)


def test_basic_tokenizer():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "Hello world from NLP."
    )

    assert tokens == [
        "hello",
        "world",
        "from",
        "nlp",
    ]


def test_basic_tokenizer_lowercases_text():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "Machine LEARNING"
    )

    assert tokens == [
        "machine",
        "learning",
    ]


def test_basic_tokenizer_handles_punctuation():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "Hello, world! NLP?"
    )

    assert tokens == [
        "hello",
        "world",
        "nlp",
    ]


def test_basic_tokenizer_handles_numbers():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "NLP 2026 version 3"
    )

    assert tokens == [
        "nlp",
        "2026",
        "version",
        "3",
    ]


def test_basic_tokenizer_handles_empty_text():
    tokenizer = BasicTokenizer()

    assert tokenizer.tokenize("") == []


def test_basic_tokenizer_rejects_non_string():
    tokenizer = BasicTokenizer()

    with pytest.raises(
        TypeError,
        match="text must be a string",
    ):
        tokenizer.tokenize(123)


def test_tokenize_many():
    tokenizer = BasicTokenizer()

    result = tokenizer.tokenize_many(
        [
            "Hello world",
            "Machine learning",
            "Natural language processing",
        ]
    )

    assert result == [
        ["hello", "world"],
        ["machine", "learning"],
        ["natural", "language", "processing"],
    ]


def test_tokenizer_is_abstract():
    assert Tokenizer.__abstractmethods__ == {
        "tokenize",
    }