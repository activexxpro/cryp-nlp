import torch
from torch import nn

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_config import TransformerConfig
from nlp_platform.ml.transformer_pipeline import (
    Prediction,
    TransformerInferencePipeline,
)
from nlp_platform.ml.transformer_tokenizer import TransformerTokenizer


class FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [
                [1, 2, 3, 0],
                [4, 5, 0, 0],
            ],
            "attention_mask": [
                [1, 1, 1, 0],
                [1, 1, 0, 0],
            ],
        }


class FakeOutput:
    def __init__(self) -> None:
        self.logits = torch.tensor(
            [
                [0.0, 2.0, 1.0],
                [3.0, 1.0, 0.0],
            ]
        )


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 3)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> FakeOutput:
        del input_ids, attention_mask, labels
        return FakeOutput()


def create_pipeline() -> TransformerInferencePipeline:
    config = TransformerConfig(
        model_name="test-model",
        max_length=4,
        num_labels=3,
    )

    tokenizer = TransformerTokenizer(
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
        config=config,
    )

    classifier = TransformerClassifier(
        model=FakeModel(),
        config=config,
    )

    return TransformerInferencePipeline(
        tokenizer=tokenizer,
        classifier=classifier,
        device=torch.device("cpu"),
    )


def test_predict_returns_predictions() -> None:
    pipeline = create_pipeline()

    result = pipeline.predict(
        ["first document", "second document"],
    )

    assert result.size == 2
    assert all(
        isinstance(prediction, Prediction)
        for prediction in result.predictions
    )


def test_predict_returns_expected_labels() -> None:
    pipeline = create_pipeline()

    result = pipeline.predict(
        ["first document", "second document"],
    )

    assert result.predictions[0].label == 1
    assert result.predictions[1].label == 0


def test_predictions_have_valid_confidence() -> None:
    pipeline = create_pipeline()

    result = pipeline.predict(
        ["first document", "second document"],
    )

    for prediction in result.predictions:
        assert 0.0 <= prediction.confidence <= 1.0


def test_empty_input_returns_empty_batch() -> None:
    pipeline = create_pipeline()

    result = pipeline.predict([])

    assert result.predictions == ()
    assert result.size == 0


def test_device_property() -> None:
    pipeline = create_pipeline()

    assert pipeline.device == torch.device("cpu")