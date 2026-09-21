import torch
from torch import nn

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_config import TransformerConfig
from nlp_platform.ml.transformer_tokenizer import TransformerTokenizer
from nlp_platform.ml.transformer_training_pipeline import (
    TransformerTrainingConfig,
    TransformerTrainingPipeline,
)


class FakeTokenizer:
    def __call__(
        self,
        texts: list[str],
        **_: object,
    ) -> dict[str, list[list[int]]]:
        return {
            "input_ids": [
                [0, 1, 2, 3]
                for _ in texts
            ],
            "attention_mask": [
                [1, 1, 1, 1]
                for _ in texts
            ],
        }


class FakeOutput:
    def __init__(
        self,
        logits: torch.Tensor,
        loss: torch.Tensor,
    ) -> None:
        self.logits = logits
        self.loss = loss


class FakeTransformerModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> FakeOutput:
        del attention_mask

        embeddings = torch.nn.functional.one_hot(
            input_ids,
            num_classes=4,
        ).float()

        pooled = embeddings.mean(dim=1)
        logits = self.classifier(pooled)

        if labels is None:
            loss = torch.tensor(0.0)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
            )

        return FakeOutput(
            logits=logits,
            loss=loss,
        )


def create_pipeline() -> TransformerTrainingPipeline:
    transformer_config = TransformerConfig(
        model_name="test-model",
        max_length=4,
        num_labels=2,
    )

    tokenizer = TransformerTokenizer(
        tokenizer=FakeTokenizer(),  # type: ignore[arg-type]
        config=transformer_config,
    )

    classifier = TransformerClassifier(
        model=FakeTransformerModel(),
        config=transformer_config,
    )

    return TransformerTrainingPipeline(
        tokenizer=tokenizer,
        classifier=classifier,
        config=TransformerTrainingConfig(
            epochs=2,
            learning_rate=1e-3,
            warmup_ratio=0.5,
        ),
        device=torch.device("cpu"),
    )


def test_pipeline_runs_training_and_evaluation() -> None:
    pipeline = create_pipeline()

    result = pipeline.run(
        train_texts=[
            "first text",
            "second text",
            "third text",
            "fourth text",
        ],
        train_labels=torch.tensor([0, 1, 0, 1]),
        validation_texts=[
            "validation one",
            "validation two",
        ],
        validation_labels=torch.tensor([0, 1]),
    )

    assert result.epochs_completed == 2
    assert result.training.samples == 4
    assert result.evaluation.samples == 2
    assert result.evaluation.loss >= 0.0
    assert 0.0 <= result.evaluation.accuracy <= 1.0


def test_pipeline_config_is_available() -> None:
    pipeline = create_pipeline()

    assert pipeline.config.epochs == 2
    assert pipeline.config.learning_rate == 1e-3


def test_pipeline_device_is_available() -> None:
    pipeline = create_pipeline()

    assert pipeline.device == torch.device("cpu")


def test_empty_training_texts_are_rejected() -> None:
    pipeline = create_pipeline()

    try:
        pipeline.run(
            train_texts=[],
            train_labels=torch.empty(0, dtype=torch.long),
            validation_texts=["validation"],
            validation_labels=torch.tensor([0]),
        )
    except ValueError as exc:
        assert "train_texts" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_empty_validation_texts_are_rejected() -> None:
    pipeline = create_pipeline()

    try:
        pipeline.run(
            train_texts=["training"],
            train_labels=torch.tensor([0]),
            validation_texts=[],
            validation_labels=torch.empty(0, dtype=torch.long),
        )
    except ValueError as exc:
        assert "validation_texts" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mismatched_training_data_is_rejected() -> None:
    pipeline = create_pipeline()

    try:
        pipeline.run(
            train_texts=["one", "two"],
            train_labels=torch.tensor([0]),
            validation_texts=["validation"],
            validation_labels=torch.tensor([0]),
        )
    except ValueError as exc:
        assert "train_texts and train_labels" in str(exc)
    else:
        raise AssertionError("Expected ValueError")