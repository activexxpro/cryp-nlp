import torch

from nlp_platform.ml.tensor_dataset import TensorDataset
from nlp_platform.ml.tensor_device import move_tensor_dataset


def create_dataset() -> TensorDataset:
    return TensorDataset(
        input_ids=torch.tensor(
            [
                [2, 4, 0, 0],
                [2, 3, 0, 0],
            ],
            dtype=torch.long,
        ),
        labels=torch.tensor(
            [0, 1],
            dtype=torch.long,
        ),
    )


def test_move_tensor_dataset_to_cpu() -> None:
    dataset = create_dataset()

    result = move_tensor_dataset(
        dataset,
        torch.device("cpu"),
    )

    assert result.input_ids.device.type == "cpu"
    assert result.labels.device.type == "cpu"


def test_move_tensor_dataset_preserves_values() -> None:
    dataset = create_dataset()

    result = move_tensor_dataset(
        dataset,
        torch.device("cpu"),
    )

    assert result.input_ids.tolist() == [
        [2, 4, 0, 0],
        [2, 3, 0, 0],
    ]
    assert result.labels.tolist() == [0, 1]


def test_move_tensor_dataset_preserves_dtype() -> None:
    dataset = create_dataset()

    result = move_tensor_dataset(
        dataset,
        torch.device("cpu"),
    )

    assert result.input_ids.dtype == torch.long
    assert result.labels.dtype == torch.long


def test_move_tensor_dataset_preserves_size() -> None:
    dataset = create_dataset()

    result = move_tensor_dataset(
        dataset,
        torch.device("cpu"),
    )

    assert result.size == dataset.size


def test_move_tensor_dataset_preserves_sequence_length() -> None:
    dataset = create_dataset()

    result = move_tensor_dataset(
        dataset,
        torch.device("cpu"),
    )

    assert result.sequence_length == dataset.sequence_length


def test_move_tensor_dataset_to_available_device() -> None:
    dataset = create_dataset()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    result = move_tensor_dataset(dataset, device)

    assert result.input_ids.device == device
    assert result.labels.device == device