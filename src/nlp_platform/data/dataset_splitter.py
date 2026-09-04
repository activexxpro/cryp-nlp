from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from nlp_platform.data.records import TextRecord
from nlp_platform.data.split_config import SplitConfig


@dataclass(frozen=True)
class DatasetSplits:
    """Container for train, validation, and test records."""

    train: list[TextRecord]
    validation: list[TextRecord]
    test: list[TextRecord]


def split_dataset(
    records: Sequence[TextRecord],
    config: SplitConfig | None = None,
) -> DatasetSplits:
    """Split records into stratified train, validation, and test sets."""

    if not records:
        return DatasetSplits(
            train=[],
            validation=[],
            test=[],
        )

    split_config = config or SplitConfig()

    labels = [record.label for record in records]

    train_records, temp_records, train_labels, temp_labels = (
        train_test_split(
            list(records),
            labels,
            test_size=(
                split_config.validation_size
                + split_config.test_size
            ),
            random_state=split_config.seed,
            stratify=labels,
        )
    )

    relative_test_size = (
        split_config.test_size
        / (
            split_config.validation_size
            + split_config.test_size
        )
    )

    validation_records, test_records = train_test_split(
        temp_records,
        test_size=relative_test_size,
        random_state=split_config.seed,
        stratify=temp_labels,
    )

    return DatasetSplits(
        train=train_records,
        validation=validation_records,
        test=test_records,
    )