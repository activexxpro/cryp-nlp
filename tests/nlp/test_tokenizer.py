import pytest

from nlp_platform.nlp.tokenizer import (
    BasicTokenizer,
    Tokenizer,
)

def test_basic_tokenizer_can_preserve_case():
    tokenizer = BasicTokenizer(
        lowercase=False,
    )

    tokens = tokenizer.tokenize(
        "Machine LEARNING"
    )

    assert tokens == [
        "Machine",
        "LEARNING",
    ]


def test_basic_tokenizer_defaults_to_lowercase():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "Machine LEARNING"
    )

    assert tokens == [
        "machine",
        "learning",
    ]


def test_basic_tokenizer_handles_repeated_whitespace():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "hello     world"
    )

    assert tokens == [
        "hello",
        "world",
    ]


def test_basic_tokenizer_handles_newlines():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "hello\nworld\nnlp"
    )

    assert tokens == [
        "hello",
        "world",
        "nlp",
    ]


def test_basic_tokenizer_handles_mixed_content():
    tokenizer = BasicTokenizer()

    tokens = tokenizer.tokenize(
        "NLP 2026: version 3.0"
    )

    assert tokens == [
        "nlp",
        "2026",
        "version",
        "3",
        "0",
    ]


def test_tokenize_many_preserves_order():
    tokenizer = BasicTokenizer()

    texts = [
        "First document",
        "Second document",
        "Third document",
    ]

    result = tokenizer.tokenize_many(texts)

    assert result == [
        ["first", "document"],
        ["second", "document"],
        ["third", "document"],
    ]