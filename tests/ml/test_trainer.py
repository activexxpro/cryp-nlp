import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp_platform.ml.tensor_dataset import TensorDataset
from nlp_platform.ml.torch_dataset import NLPDataset

from nlp_platform.ml.trainer import Trainer, TrainingResult

def create_dataloader() -> DataLoader:
    input_ids = torch.tensor(
        [
            [2, 4, 0, 0],
            [2, 3, 0, 0],
            [4, 5, 0, 0],
            [5, 6, 0, 0],
        ],
        dtype=torch.long,
    )

    labels = torch.tensor(
        [0, 1, 2, 3],
        dtype=torch.long,
    )

    dataset = TensorDataset(
        input_ids=input_ids,
        labels=labels,
    )

    return DataLoader(
        NLPDataset(dataset),
        batch_size=2,
        shuffle=False,
    )

class SmallClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            10,
            8,
            padding_idx=0,
        )
        self.classifier = nn.Linear(8, 4)

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

        pooled = masked.sum(
            dim=1
        ) / counts

        return self.classifier(pooled)


def test_trainer_returns_training_result() -> None:
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

    result = trainer.train_epoch(
        create_dataloader()
    )

    assert isinstance(
        result,
        TrainingResult,
    )


def test_training_result_contains_sample_count() -> None:
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

    result = trainer.train_epoch(
        create_dataloader()
    )

    assert result.samples == 4
    assert result.batches == 2


def test_training_result_contains_positive_loss() -> None:
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

    result = trainer.train_epoch(
        create_dataloader()
    )

    assert result.loss > 0


def test_training_updates_model_parameters() -> None:
    model = SmallClassifier()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    trainer.train_epoch(
        create_dataloader()
    )

    changed = any(
        not torch.equal(
            before[name],
            parameter,
        )
        for name, parameter in model.named_parameters()
    )

    assert changed


def test_empty_dataloader_returns_zero_loss() -> None:
    input_ids = torch.empty(
        (0, 4),
        dtype=torch.long,
    )

    labels = torch.empty(
        (0,),
        dtype=torch.long,
    )

    dataset = TensorDataset(
        input_ids=input_ids,
        labels=labels,
    )

    dataloader = DataLoader(
        NLPDataset(dataset),
        batch_size=2,
    )

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

    result = trainer.train_epoch(dataloader)

    assert result.loss == 0.0
    assert result.samples == 0
    assert result.batches == 0