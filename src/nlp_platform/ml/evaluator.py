from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from nlp_platform.ml.loss import calculate_classification_loss


@dataclass(frozen=True)
class EvaluationResult:
    """Metrics produced by model evaluation."""

    loss: float
    accuracy: float
    samples: int
    batches: int


class Evaluator:
    """Evaluate a PyTorch classification model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
    ) -> None:
        self._model = model
        self._device = device

    def evaluate(
        self,
        dataloader: DataLoader,
    ) -> EvaluationResult:
        """Evaluate the model without updating parameters."""

        self._model.eval()

        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        total_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(
                    self._device
                )
                labels = batch["labels"].to(
                    self._device
                )

                logits = self._model(input_ids)

                loss = calculate_classification_loss(
                    logits,
                    labels,
                )

                predictions = logits.argmax(
                    dim=1
                )

                batch_size = labels.shape[0]

                total_loss += (
                    loss.item() * batch_size
                )

                correct_predictions += (
                    predictions == labels
                ).sum().item()

                total_samples += batch_size
                total_batches += 1

        average_loss = (
            total_loss / total_samples
            if total_samples > 0
            else 0.0
        )

        accuracy = (
            correct_predictions / total_samples
            if total_samples > 0
            else 0.0
        )

        return EvaluationResult(
            loss=average_loss,
            accuracy=accuracy,
            samples=total_samples,
            batches=total_batches,
        )