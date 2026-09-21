from dataclasses import dataclass

import torch

from nlp_platform.ml.transformer_classifier import TransformerClassifier
from nlp_platform.ml.transformer_config import TransformerConfig
from nlp_platform.ml.transformer_evaluator import (
    TransformerEvaluationResult,
    TransformerEvaluator,
)
from nlp_platform.ml.transformer_result import TransformerRunResult
from nlp_platform.ml.transformer_scheduler import SchedulerConfig
from nlp_platform.ml.transformer_tokenizer import TransformerTokenizer
from nlp_platform.ml.transformer_trainer import (
    TransformerTrainer,
    TransformerTrainingResult,
)


@dataclass(frozen=True)
class TransformerTrainingConfig:
    """Configuration for a transformer training run."""

    epochs: int = 3
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError("warmup_ratio must be between 0 and 1")


class TransformerTrainingPipeline:
    """Coordinate transformer training and evaluation."""

    def __init__(
        self,
        tokenizer: TransformerTokenizer,
        classifier: TransformerClassifier,
        config: TransformerTrainingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._classifier = classifier
        self._config = config or TransformerTrainingConfig()
        self._device = device or torch.device("cpu")

    def run(
        self,
        train_texts: list[str],
        train_labels: torch.Tensor,
        validation_texts: list[str],
        validation_labels: torch.Tensor,
    ) -> TransformerRunResult:
        """Train and evaluate the transformer classifier."""
        if not train_texts:
            raise ValueError("train_texts must not be empty")

        if not validation_texts:
            raise ValueError("validation_texts must not be empty")

        if len(train_texts) != train_labels.shape[0]:
            raise ValueError(
                "train_texts and train_labels must have the same size",
            )

        if len(validation_texts) != validation_labels.shape[0]:
            raise ValueError(
                "validation_texts and validation_labels must have the same size",
            )

        train_batch = self._tokenizer.encode(train_texts)
        validation_batch = self._tokenizer.encode(validation_texts)

        train_input_ids = torch.tensor(
            train_batch.input_ids,
            dtype=torch.long,
        )
        train_attention_mask = torch.tensor(
            train_batch.attention_mask,
            dtype=torch.long,
        )

        validation_input_ids = torch.tensor(
            validation_batch.input_ids,
            dtype=torch.long,
        )
        validation_attention_mask = torch.tensor(
            validation_batch.attention_mask,
            dtype=torch.long,
        )

        optimizer = torch.optim.AdamW(
            self._classifier.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )

        total_steps = self._config.epochs

        scheduler_config = SchedulerConfig(
            total_steps=total_steps,
            warmup_ratio=self._config.warmup_ratio,
        )

        trainer = TransformerTrainer(
            classifier=self._classifier,
            optimizer=optimizer,
            scheduler_config=scheduler_config,
            device=self._device,
        )

        evaluator = TransformerEvaluator(
            classifier=self._classifier,
            device=self._device,
        )

        training_result: TransformerTrainingResult | None = None
        evaluation_result: TransformerEvaluationResult | None = None

        for _ in range(self._config.epochs):
            training_result = trainer.train_epoch(
                input_ids=train_input_ids,
                attention_mask=train_attention_mask,
                labels=train_labels,
            )

            evaluation_result = evaluator.evaluate(
                input_ids=validation_input_ids,
                attention_mask=validation_attention_mask,
                labels=validation_labels,
            )

        if training_result is None or evaluation_result is None:
            raise RuntimeError("transformer training produced no result")

        return TransformerRunResult(
            training=training_result,
            evaluation=evaluation_result,
            epochs_completed=self._config.epochs,
        )

    @property
    def config(self) -> TransformerTrainingConfig:
        """Return the training configuration."""
        return self._config

    @property
    def device(self) -> torch.device:
        """Return the training device."""
        return self._device

    @property
    def tokenizer(self) -> TransformerTokenizer:
        """Return the tokenizer."""
        return self._tokenizer

    @property
    def classifier(self) -> TransformerClassifier:
        """Return the classifier."""
        return self._classifier