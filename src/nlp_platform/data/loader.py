from datasets import DatasetDict, load_dataset

EXPECTED_COLUMNS = {"text", "label"}


def load_news_dataset() -> DatasetDict:
    """Load the AG News dataset."""

    dataset = load_dataset("ag_news")

    for split in dataset:
        columns = set(dataset[split].column_names)

        if not EXPECTED_COLUMNS.issubset(columns):
            raise ValueError(
                f"Unexpected dataset schema for {split}: {columns}"
            )

    return dataset