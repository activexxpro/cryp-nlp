from dataclasses import dataclass

import torch
from torch.optim import Optimizer

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_scheduler import (
    SchedulerConfig,
    create_transformer_scheduler,
)


@dataclass(frozen=True)
class TransformerTrainingResult:
    """Result of a transformer training run."""

    average_loss: float
    samples: int
    steps: int


class TransformerTrainer:
    """Train a transformer classifier with attention masks and scheduling."""

    def __init__(
        self,
        classifier: TransformerClassifier,
        optimizer: Optimizer,
        scheduler_config: SchedulerConfig,
        device: torch.device | None = None,
    ) -> None:
        self._classifier = classifier
        self._optimizer = optimizer
        self._scheduler = create_transformer_scheduler(
            optimizer,
            scheduler_config,
        )
        self._device = device or torch.device("cpu")

    def train_epoch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> TransformerTrainingResult:
        """Train the transformer for one epoch."""
        if input_ids.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                "input_ids and attention_mask must have the same batch size",
            )

        if input_ids.shape[0] != labels.shape[0]:
            raise ValueError(
                "input_ids and labels must have the same batch size",
            )

        if input_ids.shape[0] == 0:
            raise ValueError("training data must not be empty")

        self._classifier.to(self._device)
        self._classifier.train()

        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        labels = labels.to(self._device)

        self._optimizer.zero_grad()

        outputs = self._classifier.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        if outputs.loss is None:
            raise ValueError("transformer model did not return a loss")

        loss = outputs.loss
        loss.backward()

        self._optimizer.step()
        self._scheduler.step()

        return TransformerTrainingResult(
            average_loss=float(loss.detach().cpu().item()),
            samples=input_ids.shape[0],
            steps=1,
        )

    @property
    def optimizer(self) -> Optimizer:
        """Return the optimizer."""
        return self._optimizer

    @property
    def scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Return the learning-rate scheduler."""
        return self._scheduler

    @property
    def device(self) -> torch.device:
        """Return the training device."""
        return self._device