from collections.abc import Sequence

from nlp_platform.nlp.encoding_config import EncodingConfig
from nlp_platform.nlp.vocabulary import Vocabulary


class SequencePadder:
    """Pad and truncate encoded token sequences."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        config: EncodingConfig | None = None,
    ) -> None:
        self._vocabulary = vocabulary
        self._config = config or EncodingConfig()

    def pad_or_truncate(
        self,
        token_ids: Sequence[int],
    ) -> list[int]:
        """Pad and truncate an encoded sequence."""

        ids = list(token_ids)

        if (
            self._config.truncation
            and len(ids) > self._config.max_length
        ):
            ids = ids[:self._config.max_length]

        if not self._config.padding:
            return ids

        padding_count = (
            self._config.max_length - len(ids)
        )

        if padding_count <= 0:
            return ids

        pad_id = self._vocabulary.token_to_id(
            Vocabulary.PAD_TOKEN
        )

        return ids + [pad_id] * padding_count

    def pad_or_truncate_many(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Pad and truncate multiple sequences."""

        return [
            self.pad_or_truncate(sequence)
            for sequence in sequences
        ]