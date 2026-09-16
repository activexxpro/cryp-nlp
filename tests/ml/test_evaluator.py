import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp_platform.ml.evaluator import (
    EvaluationResult,
    Evaluator,
)
from nlp_platform.ml.tensor_dataset import TensorDataset
from nlp_platform.ml.torch_dataset import NLPDataset


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


def test_evaluator_returns_evaluation_result() -> None:
    model = SmallClassifier()

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader()
    )

    assert isinstance(
        result,
        EvaluationResult,
    )


def test_evaluation_result_contains_sample_count() -> None:
    model = SmallClassifier()

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader()
    )

    assert result.samples == 4
    assert result.batches == 2


def test_evaluation_result_contains_valid_loss() -> None:
    model = SmallClassifier()

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader()
    )

    assert result.loss > 0


def test_accuracy_is_between_zero_and_one() -> None:
    model = SmallClassifier()

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(
        create_dataloader()
    )

    assert 0.0 <= result.accuracy <= 1.0


def test_evaluation_does_not_update_parameters() -> None:
    model = SmallClassifier()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    evaluator.evaluate(
        create_dataloader()
    )

    for name, parameter in model.named_parameters():
        assert torch.equal(
            before[name],
            parameter,
        )


def test_empty_dataloader_returns_zero_metrics() -> None:
    dataset = TensorDataset(
        input_ids=torch.empty(
            (0, 4),
            dtype=torch.long,
        ),
        labels=torch.empty(
            (0,),
            dtype=torch.long,
        ),
    )

    dataloader = DataLoader(
        NLPDataset(dataset),
        batch_size=2,
    )

    model = SmallClassifier()

    evaluator = Evaluator(
        model=model,
        device=torch.device("cpu"),
    )

    result = evaluator.evaluate(dataloader)

    assert result.loss == 0.0
    assert result.accuracy == 0.0
    assert result.samples == 0
    assert result.batches == 0