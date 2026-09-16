import torch

from nlp_platform.ml.tensor_dataset import TensorDataset


def move_tensor_dataset(
    dataset: TensorDataset,
    device: torch.device,
) -> TensorDataset:
    """Move all tensors in a dataset to the specified device."""

    return TensorDataset(
        input_ids=dataset.input_ids.to(device),
        labels=dataset.labels.to(device),
    )