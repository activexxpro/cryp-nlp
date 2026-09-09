from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInput:
    """Represent model-ready input for a single text record."""

    input_ids: tuple[int, ...]
    label: int

    @property
    def sequence_length(self) -> int:
        """Return the input sequence length."""

        return len(self.input_ids)