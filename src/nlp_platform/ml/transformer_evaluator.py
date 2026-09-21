from dataclasses import dataclass

import torch

from nlp_platform.ml.transformer_classifier import TransformerClassifier


@dataclass(frozen=True)
class TransformerEvaluationResult:
    """Evaluation metrics for a transformer classifier."""

    loss: float
    accuracy: float
    samples: int
    batches: int


class TransformerEvaluator:
    """Evaluate a transformer classifier on labeled data."""

    def __init__(
        self,
        classifier: TransformerClassifier,
        device: torch.device | None = None,
    ) -> None:
        self._classifier = classifier
        self._device = device or torch.device("cpu")

    def evaluate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> TransformerEvaluationResult:
        """Evaluate the classifier."""
        if input_ids.shape[0] != attention_mask.shape[0]:
            raise ValueError(
                "input_ids and attention_mask must have the same batch size",
            )

        if input_ids.shape[0] != labels.shape[0]:
            raise ValueError(
                "input_ids and labels must have the same batch size",
            )

        if input_ids.shape[0] == 0:
            raise ValueError("evaluation data must not be empty")

        self._classifier.to(self._device)
        self._classifier.eval()

        input_ids = input_ids.to(self._device)
        attention_mask = attention_mask.to(self._device)
        labels = labels.to(self._device)

        with torch.no_grad():
            outputs = self._classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        if outputs.loss is None:
            raise ValueError("transformer model did not return a loss")

        predictions = outputs.logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        samples = labels.shape[0]

        return TransformerEvaluationResult(
            loss=float(outputs.loss.detach().cpu().item()),
            accuracy=float(correct / samples),
            samples=samples,
            batches=1,
        )

    @property
    def device(self) -> torch.device:
        """Return the evaluation device."""
        return self._device