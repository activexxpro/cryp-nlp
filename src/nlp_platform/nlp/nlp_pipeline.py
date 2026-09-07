from collections.abc import Sequence

from nlp_platform.nlp.batch_encoder import BatchEncoder


class NLPPreprocessingPipeline:
    """Convert raw text into model-ready token ID sequences."""

    def __init__(
        self,
        batch_encoder: BatchEncoder,
    ) -> None:
        self._batch_encoder = batch_encoder

    def process(self, text: str) -> list[int]:
        """Process a single text string."""

        if not isinstance(text, str):
            raise TypeError("text must be a string")

        return self._batch_encoder.encode(text)

    def process_many(
        self,
        texts: Sequence[str],
    ) -> list[list[int]]:
        """Process multiple text strings."""

        return self._batch_encoder.encode_many(texts)