import pytest

from nlp_platform.nlp.encoding_config import EncodingConfig


def test_default_config() -> None:
    config = EncodingConfig()

    assert config.max_length == 128
    assert config.padding is True
    assert config.truncation is True


def test_custom_max_length() -> None:
    config = EncodingConfig(
        max_length=64,
    )

    assert config.max_length == 64


def test_padding_can_be_disabled() -> None:
    config = EncodingConfig(
        max_length=64,
        padding=False,
    )

    assert config.padding is False


def test_truncation_can_be_disabled() -> None:
    config = EncodingConfig(
        max_length=64,
        truncation=False,
    )

    assert config.truncation is False


def test_invalid_max_length() -> None:
    with pytest.raises(
        ValueError,
        match="max_length must be at least 1",
    ):
        EncodingConfig(max_length=0)