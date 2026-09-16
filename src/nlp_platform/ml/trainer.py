from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp_platform.ml.loss import calculate_classification_loss


@dataclass(frozen=True)
class TrainingResult:
    """Metrics produced by one training epoch."""

    loss: float
    samples: int
    batches: int


class Trainer:
    """Train a PyTorch text classification model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._device = device

    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> TrainingResult:
        """Train the model for one complete epoch."""

        self._model.train()

        total_loss = 0.0
        total_samples = 0
        total_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self._device)
            labels = batch["labels"].to(self._device)

            self._optimizer.zero_grad()

            logits = self._model(input_ids)

            loss = calculate_classification_loss(
                logits,
                labels,
            )

            loss.backward()
            self._optimizer.step()

            batch_size = labels.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            total_batches += 1

        average_loss = (
            total_loss / total_samples
            if total_samples > 0
            else 0.0
        )

        return TrainingResult(
            loss=average_loss,
            samples=total_samples,
            batches=total_batches,
        )