from pathlib import Path

import torch
from torch import nn

from nlp_platform.ml.dataloader_config import DataLoaderConfig
from nlp_platform.ml.device import DeviceConfig
from nlp_platform.ml.training_pipeline import TrainingPipeline
from nlp_platform.ml.training_run_config import TrainingRunConfig
from nlp_platform.nlp.encoded_dataset import EncodedDataset
from nlp_platform.nlp.model_input import ModelInput
from nlp_platform.nlp.training_dataset import TrainingDataset


class SmallClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.embedding = nn.Embedding(
            10,
            8,
            padding_idx=0,
        )

        self.classifier = nn.Linear(8, 2)

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


def create_dataset() -> TrainingDataset:
    train = EncodedDataset(
        inputs=tuple(
            ModelInput(
                input_ids=(
                    2,
                    3,
                    0,
                    0,
                ),
                label=0,
            )
            for _ in range(8)
        )
        + tuple(
            ModelInput(
                input_ids=(
                    4,
                    5,
                    0,
                    0,
                ),
                label=1,
            )
            for _ in range(8)
        )
    )

    validation = EncodedDataset(
        inputs=(
            ModelInput(
                input_ids=(2, 3, 0, 0),
                label=0,
            ),
            ModelInput(
                input_ids=(4, 5, 0, 0),
                label=1,
            ),
        )
    )

    test = EncodedDataset(
        inputs=(
            ModelInput(
                input_ids=(2, 3, 0, 0),
                label=0,
            ),
            ModelInput(
                input_ids=(4, 5, 0, 0),
                label=1,
            ),
        )
    )

    return TrainingDataset(
        train=train,
        validation=validation,
        test=test,
    )


def test_training_pipeline_runs() -> None:
    model = SmallClassifier()

    pipeline = TrainingPipeline(
        model=model,
        config=TrainingRunConfig(
            epochs=2,
            learning_rate=0.01,
        ),
        device_config=DeviceConfig(
            device="cpu"
        ),
        dataloader_config=DataLoaderConfig(
            batch_size=4,
            shuffle=True,
        ),
    )

    result = pipeline.run(
        create_dataset()
    )

    assert result.epochs_completed == 2
    assert result.history.epoch_count == 2


def test_training_pipeline_records_metrics() -> None:
    model = SmallClassifier()

    pipeline = TrainingPipeline(
        model=model,
        config=TrainingRunConfig(
            epochs=2,
            learning_rate=0.01,
        ),
        device_config=DeviceConfig(
            device="cpu"
        ),
    )

    result = pipeline.run(
        create_dataset()
    )

    assert len(
        result.history.train_losses
    ) == 2

    assert len(
        result.history.validation_losses
    ) == 2

    assert len(
        result.history.validation_accuracies
    ) == 2


def test_training_pipeline_saves_best_checkpoint(
    tmp_path: Path,
) -> None:
    model = SmallClassifier()

    pipeline = TrainingPipeline(
        model=model,
        config=TrainingRunConfig(
            epochs=2,
            learning_rate=0.01,
        ),
        device_config=DeviceConfig(
            device="cpu"
        ),
        checkpoint_directory=(
            tmp_path / "checkpoints"
        ),
    )

    result = pipeline.run(
        create_dataset()
    )

    assert result.best_epoch is not None
    assert result.best_validation_loss is not None

    checkpoint = (
        tmp_path
        / "checkpoints"
        / "best_model.pt"
    )

    assert checkpoint.exists()


def test_training_pipeline_supports_early_stopping() -> None:
    model = SmallClassifier()

    pipeline = TrainingPipeline(
        model=model,
        config=TrainingRunConfig(
            epochs=10,
            learning_rate=0.01,
            early_stopping_patience=2,
        ),
        device_config=DeviceConfig(
            device="cpu"
        ),
    )

    result = pipeline.run(
        create_dataset()
    )

    assert result.epochs_completed <= 10
    assert isinstance(
        result.stopped_early,
        bool,
    )