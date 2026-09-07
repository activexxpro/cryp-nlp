import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from nlp_platform.nlp.tokenization_config import TokenizationConfig

_WORD_PATTERN = re.compile(r"\b\w+\b")


class Tokenizer(ABC):
    """Abstract interface for text tokenization."""

    @abstractmethod
    def tokenize(self, text: str) -> list[str]:
        """Tokenize a single text string."""
        raise NotImplementedError

    def tokenize_many(
        self,
        texts: Sequence[str],
    ) -> list[list[str]]:
        """Tokenize multiple text strings."""

        return [
            self.tokenize(text)
            for text in texts
        ]


class BasicTokenizer(Tokenizer):
    """Configurable word tokenizer."""

    def __init__(
        self,
        config: TokenizationConfig | None = None,
        *,
        lowercase: bool | None = None,
    ) -> None:
        if config is not None and lowercase is not None:
            raise ValueError(
                "provide either config or lowercase, not both"
            )

        if lowercase is not None:
            config = TokenizationConfig(
                lowercase=lowercase,
            )

        self._config = config or TokenizationConfig()

    def tokenize(self, text: str) -> list[str]:
        """Split text into word-like tokens."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self._config.lowercase:
            text = text.lower()

        tokens = _WORD_PATTERN.findall(text)

        return [
            token
            for token in tokens
            if len(token) >= self._config.min_token_length
        ]