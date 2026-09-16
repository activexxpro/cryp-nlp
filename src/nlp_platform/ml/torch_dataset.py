import torch
from torch.utils.data import Dataset

from nlp_platform.ml.tensor_dataset import TensorDataset

class NLPDataset(Dataset):
    """PyTorch dataset for NLP model inputs."""

    def __init__(
        self,
        dataset: TensorDataset,
    ) -> None:
        self._dataset = dataset

    def __len__(self) -> int:
        """Return the number of samples."""

        return self._dataset.size

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        """Return one training example."""

        return {
            "input_ids": self._dataset.input_ids[index],
            "labels": self._dataset.labels[index],
        }