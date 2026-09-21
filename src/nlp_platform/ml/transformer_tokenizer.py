from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from nlp_platform.ml.transformer_config import TransformerConfig


@dataclass(frozen=True)
class TokenizedBatch:
    """Batch of transformer tokenized text."""

    input_ids: tuple[tuple[int, ...], ...]
    attention_mask: tuple[tuple[int, ...], ...]

    @property
    def size(self) -> int:
        return len(self.input_ids)

    @property
    def sequence_length(self) -> int:
        if not self.input_ids:
            return 0

        return len(self.input_ids[0])


class TransformerTokenizer:
    """Adapter around a Hugging Face tokenizer."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        config: TransformerConfig | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._config = config or TransformerConfig()

    def encode(self, texts: list[str]) -> TokenizedBatch:
        """Tokenize a batch of text inputs."""
        if not texts:
            return TokenizedBatch(
                input_ids=(),
                attention_mask=(),
            )

        encoded = self._tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self._config.max_length,
            return_attention_mask=True,
        )

        input_ids = tuple(
            tuple(sequence)
            for sequence in encoded["input_ids"]
        )

        attention_mask = tuple(
            tuple(sequence)
            for sequence in encoded["attention_mask"]
        )

        return TokenizedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @property
    def model_max_length(self) -> int:
        """Return the configured maximum sequence length."""
        return self._config.max_length