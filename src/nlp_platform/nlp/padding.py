from collections.abc import Sequence

from nlp_platform.nlp.vocabulary import Vocabulary


class SequencePadder:
    """Pad and truncate encoded token sequences."""

    def __init__(
        self,
        vocabulary: Vocabulary,
        max_length: int,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be at least 1")

        self._vocabulary = vocabulary
        self._max_length = max_length

    def pad_or_truncate(
        self,
        token_ids: Sequence[int],
    ) -> list[int]:
        """Pad or truncate a single encoded sequence."""

        ids = list(token_ids)

        if len(ids) > self._max_length:
            return ids[:self._max_length]

        padding_count = self._max_length - len(ids)

        return ids + [
            self._vocabulary.token_to_id(
                Vocabulary.PAD_TOKEN
            )
        ] * padding_count

    def pad_or_truncate_many(
        self,
        sequences: Sequence[Sequence[int]],
    ) -> list[list[int]]:
        """Pad or truncate multiple encoded sequences."""

        return [
            self.pad_or_truncate(sequence)
            for sequence in sequences
        ]