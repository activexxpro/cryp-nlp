import pytest

from nlp_platform.ml.model_config import ModelConfig


def test_default_model_config() -> None:
    config = ModelConfig(vocab_size=100)

    assert config.vocab_size == 100
    assert config.embedding_dim == 128
    assert config.hidden_dim == 128
    assert config.num_classes == 4
    assert config.padding_idx == 0


def test_custom_model_config() -> None:
    config = ModelConfig(
        vocab_size=500,
        embedding_dim=64,
        hidden_dim=256,
        num_classes=3,
        padding_idx=1,
    )

    assert config.vocab_size == 500
    assert config.embedding_dim == 64
    assert config.hidden_dim == 256
    assert config.num_classes == 3
    assert config.padding_idx == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vocab_size": 1},
        {"vocab_size": 100, "embedding_dim": 0},
        {"vocab_size": 100, "hidden_dim": 0},
        {"vocab_size": 100, "num_classes": 1},
        {"vocab_size": 100, "padding_idx": -1},
        {"vocab_size": 100, "padding_idx": 100},
    ],
)
def test_invalid_model_config_raises(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)