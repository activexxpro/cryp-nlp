from collections.abc import Sequence

from nlp_platform.nlp.vocabulary import Vocabulary


class TokenEncoder:
    """Convert token sequences into vocabulary IDs."""

    def __init__(self, vocabulary: Vocabulary) -> None:
        self._vocabulary = vocabulary

    def encode(
        self,
        tokens: Sequence[str],
    ) -> list[int]:
        """Convert tokens into integer IDs."""

        return [
            self._vocabulary.token_to_id(token)
            for token in tokens
        ]

    def encode_many(
        self,
        token_sequences: Sequence[Sequence[str]],
    ) -> list[list[int]]:
        """Convert multiple token sequences into integer IDs."""

        return [
            self.encode(tokens)
            for tokens in token_sequences
        ]

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> list[str]:
        """Convert integer IDs back into tokens."""

        return [
            self._vocabulary.id_to_token(token_id)
            for token_id in token_ids
        ]

    def decode_many(
        self,
        encoded_sequences: Sequence[Sequence[int]],
    ) -> list[list[str]]:
        """Convert multiple ID sequences back into tokens."""

        return [
            self.decode(token_ids)
            for token_ids in encoded_sequences
        ]