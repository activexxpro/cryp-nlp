import pytest

from nlp_platform.ml.experiment_config import ExperimentConfig


def test_default_config() -> None:
    config = ExperimentConfig()

    assert config.name == "baseline"
    assert config.seed == 42
    assert config.description == ""


def test_custom_config() -> None:
    config = ExperimentConfig(
        name="News Classifier",
        seed=123,
        description="Baseline news classification experiment",
    )

    assert config.name == "News Classifier"
    assert config.seed == 123
    assert config.description == (
        "Baseline news classification experiment"
    )


def test_identifier_normalizes_name() -> None:
    config = ExperimentConfig(
        name="News Classifier V1"
    )

    assert config.identifier == "news-classifier-v1"


def test_identifier_strips_whitespace() -> None:
    config = ExperimentConfig(
        name="  Baseline Model  "
    )

    assert config.identifier == "baseline-model"


def test_empty_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="")


def test_whitespace_name_is_rejected() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(name="   ")


@pytest.mark.parametrize(
    "seed",
    [-1, -10],
)
def test_negative_seed_is_rejected(seed: int) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(seed=seed)