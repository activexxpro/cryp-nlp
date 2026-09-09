from collections.abc import Sequence
from dataclasses import dataclass

from nlp_platform.data.records import TextRecord
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.model_input import ModelInput


@dataclass(frozen=True)
class EncodedDataset:
    """Represent an encoded collection of model inputs."""

    inputs: tuple[ModelInput, ...]

    @property
    def size(self) -> int:
        """Return the number of encoded records."""

        return len(self.inputs)

    @property
    def input_ids(self) -> tuple[tuple[int, ...], ...]:
        """Return encoded input IDs."""

        return tuple(
            item.input_ids
            for item in self.inputs
        )

    @property
    def labels(self) -> tuple[int, ...]:
        """Return labels in dataset order."""

        return tuple(
            item.label
            for item in self.inputs
        )


class DatasetEncodingService:
    """Encode text records into model-ready datasets."""

    def __init__(
        self,
        encoder: DatasetEncoder,
    ) -> None:
        self._encoder = encoder

    def encode(
        self,
        records: Sequence[TextRecord],
    ) -> EncodedDataset:
        """Encode a collection of records."""

        inputs = self._encoder.encode_many(records)

        return EncodedDataset(
            inputs=tuple(inputs),
        )