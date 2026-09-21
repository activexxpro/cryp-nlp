import torch
from torch import nn

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_scheduler import SchedulerConfig
from nlp_platform.ml.transformer_trainer import TransformerTrainer


class FakeOutput:
    def __init__(self, loss: torch.Tensor) -> None:
        self.loss = loss
        self.logits = torch.tensor(
            [[1.0, 2.0]],
            dtype=torch.float32,
        )


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

        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits,
                labels,
            )

        if loss is None:
            loss = torch.tensor(
                0.0,
                requires_grad=True,
            )

        return FakeOutput(loss)


def create_trainer() -> TransformerTrainer:
    classifier = TransformerClassifier(
        model=FakeTransformerModel(),
    )

    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=1e-3,
    )

    return TransformerTrainer(
        classifier=classifier,
        optimizer=optimizer,
        scheduler_config=SchedulerConfig(
            total_steps=10,
            warmup_ratio=0.1,
        ),
        device=torch.device("cpu"),
    )


def create_batch() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    input_ids = torch.tensor(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
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


def test_train_epoch_returns_result() -> None:
    trainer = create_trainer()
    input_ids, attention_mask, labels = create_batch()

    result = trainer.train_epoch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert result.samples == 2
    assert result.steps == 1
    assert result.average_loss >= 0.0


def test_train_epoch_updates_model() -> None:
    trainer = create_trainer()
    input_ids, attention_mask, labels = create_batch()

    before = [
        parameter.detach().clone()
        for parameter in trainer.optimizer.param_groups[0]["params"]
    ]

    trainer.train_epoch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    after = [
        parameter.detach().clone()
        for parameter in trainer.optimizer.param_groups[0]["params"]
    ]

    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, after)
    )


def test_scheduler_advances_after_training_step() -> None:
    trainer = create_trainer()
    input_ids, attention_mask, labels = create_batch()

    initial_step = trainer.scheduler.last_epoch

    trainer.train_epoch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )

    assert trainer.scheduler.last_epoch == initial_step + 1


def test_device_property() -> None:
    trainer = create_trainer()

    assert trainer.device == torch.device("cpu")


def test_empty_training_data_is_rejected() -> None:
    trainer = create_trainer()

    input_ids = torch.empty((0, 4), dtype=torch.long)
    attention_mask = torch.empty((0, 4), dtype=torch.long)
    labels = torch.empty((0,), dtype=torch.long)

    try:
        trainer.train_epoch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    except ValueError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_mismatched_attention_mask_is_rejected() -> None:
    trainer = create_trainer()

    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.ones((1, 4), dtype=torch.long)
    labels = torch.zeros(2, dtype=torch.long)

    try:
        trainer.train_epoch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
    except ValueError as exc:
        assert "same batch size" in str(exc)
    else:
        raise AssertionError("Expected ValueError")