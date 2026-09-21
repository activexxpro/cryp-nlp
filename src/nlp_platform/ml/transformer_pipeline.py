from dataclasses import dataclass

import torch

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_tokenizer import TransformerTokenizer


@dataclass(frozen=True)
class Prediction:
    """Classification prediction for a single text."""

    label: int
    confidence: float


@dataclass(frozen=True)
class PredictionBatch:
    """Classification predictions for multiple texts."""

    predictions: tuple[Prediction, ...]

    @property
    def size(self) -> int:
        return len(self.predictions)


class TransformerInferencePipeline:
    """Run text classification using a tokenizer and transformer model."""

    def __init__(
        self,
        tokenizer: TransformerTokenizer,
        classifier: TransformerClassifier,
        device: torch.device | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._classifier = classifier
        self._device = device or torch.device("cpu")

    def predict(self, texts: list[str]) -> PredictionBatch:
        """Predict labels and confidence scores for text inputs."""
        if not texts:
            return PredictionBatch(predictions=())

        tokenized = self._tokenizer.encode(texts)

        input_ids = torch.tensor(
            tokenized.input_ids,
            dtype=torch.long,
            device=self._device,
        )
        attention_mask = torch.tensor(
            tokenized.attention_mask,
            dtype=torch.long,
            device=self._device,
        )

        self._classifier.to(self._device)
        self._classifier.eval()

        with torch.no_grad():
            outputs = self._classifier.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        probabilities = torch.softmax(outputs.logits, dim=-1)
        confidences, labels = probabilities.max(dim=-1)

        predictions = tuple(
            Prediction(
                label=int(label),
                confidence=float(confidence),
            )
            for label, confidence in zip(labels, confidences)
        )

        return PredictionBatch(predictions=predictions)

    @property
    def device(self) -> torch.device:
        """Return the inference device."""
        return self._device