import pytest

from nlp_platform.data.split_config import SplitConfig


def test_default_split_config():
    config = SplitConfig()

    assert config.validation_size == 0.1
    assert config.test_size == 0.1
    assert config.seed == 42


def test_custom_split_config():
    config = SplitConfig(
        validation_size=0.2,
        test_size=0.15,
        seed=123,
    )

    assert config.validation_size == 0.2
    assert config.test_size == 0.15
    assert config.seed == 123


def test_validation_size_must_be_positive():
    with pytest.raises(
        ValueError,
        match="validation_size must be between",
    ):
        SplitConfig(validation_size=0.0)


def test_validation_size_must_be_less_than_one():
    with pytest.raises(
        ValueError,
        match="validation_size must be between",
    ):
        SplitConfig(validation_size=1.0)


def test_test_size_must_be_positive():
    with pytest.raises(
        ValueError,
        match="test_size must be between",
    ):
        SplitConfig(test_size=0.0)


def test_test_size_must_be_less_than_one():
    with pytest.raises(
        ValueError,
        match="test_size must be between",
    ):
        SplitConfig(test_size=1.0)


def test_split_sizes_cannot_consume_entire_dataset():
    with pytest.raises(
        ValueError,
        match="must be less than 1",
    ):
        SplitConfig(
            validation_size=0.5,
            test_size=0.5,
        )


def test_negative_seed_is_rejected():
    with pytest.raises(
        ValueError,
        match="seed must not be negative",
    ):
        SplitConfig(seed=-1)