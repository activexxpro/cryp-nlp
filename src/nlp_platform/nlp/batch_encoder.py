from collections.abc import Sequence

from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import Tokenizer


class BatchEncoder:
    """Tokenize, encode, pad, and truncate batches of text."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        encoder: TokenEncoder,
        padder: SequencePadder,
    ) -> None:
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._padder = padder

    def encode(
        self,
        text: str,
    ) -> list[int]:
        """Encode and pad a single text."""

        tokens = self._tokenizer.tokenize(text)
        token_ids = self._encoder.encode(tokens)

        return self._padder.pad_or_truncate(token_ids)

    def encode_many(
        self,
        texts: Sequence[str],
    ) -> list[list[int]]:
        """Encode and pad multiple texts."""

        return [
            self.encode(text)
            for text in texts
        ]