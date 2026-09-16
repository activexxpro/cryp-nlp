from dataclasses import dataclass

import torch

from nlp_platform.nlp.encoded_dataset import EncodedDataset


@dataclass(frozen=True)
class TensorDataset:
    """Represent an encoded dataset as PyTorch tensors."""

    input_ids: torch.Tensor
    labels: torch.Tensor

    @property
    def size(self) -> int:
        """Return the number of samples."""

        return self.input_ids.shape[0]

    @property
    def sequence_length(self) -> int:
        """Return the sequence length."""

        if self.input_ids.ndim != 2:
            raise ValueError(
                "input_ids must be a two-dimensional tensor"
            )

        return self.input_ids.shape[1]


def to_tensor_dataset(
    dataset: EncodedDataset,
) -> TensorDataset:
    """Convert an encoded dataset into PyTorch tensors."""

    input_ids = torch.tensor(
        dataset.input_ids,
        dtype=torch.long,
    )

    labels = torch.tensor(
        dataset.labels,
        dtype=torch.long,
    )

    return TensorDataset(
        input_ids=input_ids,
        labels=labels,
    )