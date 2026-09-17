import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp_platform.ml.evaluator import Evaluator
from nlp_platform.ml.tensor_dataset import TensorDataset
from nlp_platform.ml.torch_dataset import NLPDataset
from nlp_platform.ml.trainer import Trainer
from nlp_platform.ml.training_service import (
    TrainingRunResult,
    TrainingService,
)


class SmallClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            10,
            8,
            padding_idx=0,
        )

        self.classifier = nn.Linear(
            8,
            4,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)

        mask = (
            input_ids != 0
        ).unsqueeze(-1)

        masked = embeddings * mask

        counts = mask.sum(
            dim=1
        ).clamp(min=1)

        pooled = (
            masked.sum(dim=1)
            / counts
        )

        return self.classifier(pooled)


def create_dataloader() -> DataLoader:
    dataset = TensorDataset(
        input_ids=torch.tensor(
            [
                [2, 4, 0, 0],
                [2, 3, 0, 0],
                [4, 5, 0, 0],
                [5, 6, 0, 0],
            ],
            dtype=torch.long,
        ),
        labels=torch.tensor(
            [0, 1, 2, 3],
            dtype=torch.long,
        ),
    )

    return DataLoader(
        NLPDataset(dataset),
        batch_size=2,
        shuffle=False,
    )


def create_service(
    epochs: int = 3,
) -> TrainingService:
    model = SmallClassifier()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    return TrainingService(
        trainer=trainer,
        evaluator=evaluator,
        epochs=epochs,
    )


def test_training_service_returns_result() -> None:
    service = create_service()

    result = service.train(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
    )

    assert isinstance(
        result,
        TrainingRunResult,
    )


def test_training_service_completes_configured_epochs() -> None:
    service = create_service(epochs=3)

    result = service.train(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
    )

    assert result.epochs_completed == 3
    assert result.history.epoch_count == 3


def test_training_history_contains_epoch_numbers() -> None:
    service = create_service(epochs=3)

    result = service.train(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
    )

    assert [
        metrics.epoch
        for metrics in result.history.epochs
    ] == [1, 2, 3]


def test_training_history_contains_losses() -> None:
    service = create_service(epochs=2)

    result = service.train(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
    )

    assert len(result.history.train_losses) == 2
    assert len(result.history.validation_losses) == 2

    assert all(
        loss > 0
        for loss in result.history.train_losses
    )

    assert all(
        loss > 0
        for loss in result.history.validation_losses
    )


def test_training_history_contains_accuracy() -> None:
    service = create_service(epochs=2)

    result = service.train(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
    )

    assert len(
        result.history.validation_accuracies
    ) == 2

    assert all(
        0.0 <= accuracy <= 1.0
        for accuracy in result.history.validation_accuracies
    )


def test_invalid_epoch_count_raises() -> None:
    model = SmallClassifier()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    try:
        TrainingService(
            trainer=trainer,
            evaluator=evaluator,
            epochs=0,
        )
    except ValueError as exc:
        assert str(exc) == "epochs must be at least 1"
    else:
        raise AssertionError(
            "TrainingService should reject zero epochs"
        )