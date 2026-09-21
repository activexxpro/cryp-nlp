import pytest

from nlp_platform.ml.transformer_config import TransformerConfig


def test_transformer_config_defaults() -> None:
    config = TransformerConfig()

    assert config.model_name == "distilbert-base-uncased"
    assert config.max_length == 128
    assert config.num_labels == 4
    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.01
    assert config.warmup_ratio == 0.1


def test_transformer_config_accepts_custom_values() -> None:
    config = TransformerConfig(
        model_name="bert-base-uncased",
        max_length=256,
        num_labels=3,
        learning_rate=1e-5,
        weight_decay=0.02,
        warmup_ratio=0.2,
    )

    assert config.model_name == "bert-base-uncased"
    assert config.max_length == 256
    assert config.num_labels == 3
    assert config.learning_rate == 1e-5
    assert config.weight_decay == 0.02
    assert config.warmup_ratio == 0.2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", ""),
        ("max_length", 0),
        ("num_labels", 1),
        ("learning_rate", 0),
        ("weight_decay", -0.01),
        ("warmup_ratio", -0.1),
        ("warmup_ratio", 1.1),
    ],
)
def test_transformer_config_rejects_invalid_values(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        TransformerConfig(**{field: value})