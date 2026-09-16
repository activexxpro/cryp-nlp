import pytest

from nlp_platform.ml.dataloader_config import DataLoaderConfig


def test_default_configuration() -> None:
    config = DataLoaderConfig()

    assert config.batch_size == 32
    assert config.shuffle is True
    assert config.num_workers == 0
    assert config.drop_last is False


def test_custom_configuration() -> None:
    config = DataLoaderConfig(
        batch_size=16,
        shuffle=False,
        num_workers=2,
        drop_last=True,
    )

    assert config.batch_size == 16
    assert config.shuffle is False
    assert config.num_workers == 2
    assert config.drop_last is True


def test_invalid_batch_size() -> None:
    with pytest.raises(
        ValueError,
        match="batch_size must be at least 1",
    ):
        DataLoaderConfig(batch_size=0)


def test_negative_worker_count() -> None:
    with pytest.raises(
        ValueError,
        match="num_workers must not be negative",
    ):
        DataLoaderConfig(num_workers=-1)