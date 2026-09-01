from datasets import Dataset, DatasetDict


def split_train_dataset(
    dataset: Dataset,
    validation_size: float = 0.1,
    seed: int = 42,
) -> DatasetDict:
    """Split the training dataset into train and validation sets."""

    split = dataset.train_test_split(
        test_size=validation_size,
        seed=seed,
    )

    return DatasetDict(
        {
            "train": split["train"],
            "validation": split["test"],
        }
    )