import re
from abc import ABC, abstractmethod
from collections.abc import Sequence


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
    """Simple configurable word tokenizer."""

    def __init__(
        self,
        lowercase: bool = True,
    ) -> None:
        self._lowercase = lowercase

    def tokenize(self, text: str) -> list[str]:
        """Split text into word-like tokens."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        if self._lowercase:
            text = text.lower()

        return _WORD_PATTERN.findall(text)