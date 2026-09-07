import pytest

from nlp_platform.nlp.tokenization_config import (
    TokenizationConfig,
)
from nlp_platform.nlp.tokenizer import BasicTokenizer


def test_default_configuration():
    config = TokenizationConfig()

    assert config.lowercase is True
    assert config.min_token_length == 1


def test_custom_configuration():
    config = TokenizationConfig(
        lowercase=False,
        min_token_length=3,
    )

    assert config.lowercase is False
    assert config.min_token_length == 3


def test_min_token_length_must_be_positive():
    with pytest.raises(
        ValueError,
        match="min_token_length must be at least 1",
    ):
        TokenizationConfig(
            min_token_length=0,
        )


def test_negative_min_token_length_is_rejected():
    with pytest.raises(
        ValueError,
        match="min_token_length must be at least 1",
    ):
        TokenizationConfig(
            min_token_length=-1,
        )


def test_configuration_is_immutable():
    config = TokenizationConfig()

    with pytest.raises(
        AttributeError,
    ):
        config.lowercase = False

def test_tokenizer_accepts_configuration():
    config = TokenizationConfig(
        lowercase=False,
    )

    tokenizer = BasicTokenizer(
        config=config,
    )

    assert tokenizer.tokenize(
        "Hello WORLD"
    ) == [
        "Hello",
        "WORLD",
    ]


def test_tokenizer_filters_short_tokens():
    config = TokenizationConfig(
        min_token_length=3,
    )

    tokenizer = BasicTokenizer(
        config=config,
    )

    assert tokenizer.tokenize(
        "I am an NLP engineer"
    ) == [
        "nlp",
        "engineer",
    ]


def test_tokenizer_combines_configuration_options():
    config = TokenizationConfig(
        lowercase=True,
        min_token_length=4,
    )

    tokenizer = BasicTokenizer(
        config=config,
    )

    assert tokenizer.tokenize(
        "This is NLP and AI engineering"
    ) == [
        "this",
        "engineering",
    ]