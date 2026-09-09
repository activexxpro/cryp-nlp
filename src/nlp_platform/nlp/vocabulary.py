from collections import Counter
from collections.abc import Sequence


class Vocabulary:
    """Map tokens to integer IDs and back."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(
        self,
        token_to_id: dict[str, int],
    ) -> None:
        if not token_to_id:
            raise ValueError("token_to_id must not be empty")

        self._token_to_id = dict(token_to_id)
        self._id_to_token = {
            token_id: token
            for token, token_id in self._token_to_id.items()
        }

    @classmethod
    def build(
        cls,
        token_sequences: Sequence[Sequence[str]],
        min_frequency: int = 1,
    ) -> "Vocabulary":
        """Build a vocabulary from tokenized text."""

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be at least 1"
            )

        counter = Counter(
            token
            for sequence in token_sequences
            for token in sequence
        )

        token_to_id = {
            cls.PAD_TOKEN: 0,
            cls.UNK_TOKEN: 1,
        }

        for token, frequency in sorted(counter.items()):
            if (
                frequency >= min_frequency
                and token not in token_to_id
            ):
                token_to_id[token] = len(token_to_id)

        return cls(token_to_id)

    @property
    def size(self) -> int:
        """Return the vocabulary size."""

        return len(self._token_to_id)

    def token_to_id(self, token: str) -> int:
        """Convert a token to its integer ID."""

        return self._token_to_id.get(
            token,
            self._token_to_id[self.UNK_TOKEN],
        )

    def id_to_token(self, token_id: int) -> str:
        """Convert an integer ID to its token."""

        try:
            return self._id_to_token[token_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown token ID: {token_id}"
            ) from exc

    def contains(self, token: str) -> bool:
        """Return whether a token exists in the vocabulary."""

        return token in self._token_to_id

    def tokens(self) -> list[str]:
        """Return vocabulary tokens in ID order."""

        return [
            self._id_to_token[token_id]
            for token_id in range(self.size)
        ]