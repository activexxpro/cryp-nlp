import torch
from torch import nn

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_evaluator import TransformerEvaluator


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
        self.layer = nn.Linear(4, 2)

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
        logits = self.layer(pooled)

        if labels is None:
            loss = torch.tensor(0.5)
        else:
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
            )

        return FakeOutput(
            logits=logits,
            loss=loss,
        )


def create_evaluator() -> TransformerEvaluator:
    classifier = TransformerClassifier(
        model=FakeTransformerModel(),
    )

    return TransformerEvaluator(
        classifier=classifier,
        device=torch.device("cpu"),
    )


def create_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    input_ids = torch.tensor(
        [
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=torch.long,
    )

    attention_mask = torch.ones(
        (2, 4),
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1],
        dtype=torch.long,
    )

    return input_ids, attention_mask, labels


def test_evaluate_returns_metrics() -> None:
    evaluator = create_evaluator()
    input_ids, attention_mask, labels = create_batch()

    result = evaluator.evaluate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert result.loss >= 0.0
    assert 0.0 <= result.accuracy <= 1.0
    assert result.samples == 2
    assert result.batches == 1


def test_evaluate_returns_valid_accuracy() -> None:
    evaluator = create_evaluator()
    input_ids, attention_mask, labels = create_batch()

    result = evaluator.evaluate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert 0.0 <= result.accuracy <= 1.0


def test_evaluate_does_not_update_parameters() -> None:
    evaluator = create_evaluator()

    input_ids, attention_mask, labels = create_batch()

    before = [
        parameter.detach().clone()
        for parameter in evaluator._classifier.parameters()
    ]

    evaluator.evaluate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    after = [
        parameter.detach().clone()
        for parameter in evaluator._classifier.parameters()
    ]

    for previous, current in zip(before, after):
        assert torch.equal(previous, current)


def test_device_property() -> None:
    evaluator = create_evaluator()

    assert evaluator.device == torch.device("cpu")


def test_empty_evaluation_data_is_rejected() -> None:
    evaluator = create_evaluator()

    input_ids = torch.empty((0, 4), dtype=torch.long)
    attention_mask = torch.empty((0, 4), dtype=torch.long)
    labels = torch.empty((0,), dtype=torch.long)

    try:
        evaluator.evaluate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")