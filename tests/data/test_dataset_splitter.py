from collections import Counter

from nlp_platform.data.dataset_splitter import (
    DatasetSplits,
    split_dataset,
)
from nlp_platform.data.records import TextRecord
from nlp_platform.data.split_config import SplitConfig


def _build_records() -> list[TextRecord]:
    records = []

    for label in range(4):
        for index in range(25):
            records.append(
                TextRecord(
                    text=f"class {label} sample {index}",
                    label=label,
                )
            )

    return records


def test_split_returns_three_datasets():
    records = _build_records()

    splits = split_dataset(records)

    assert isinstance(splits, DatasetSplits)
    assert len(splits.train) == 80
    assert len(splits.validation) == 10
    assert len(splits.test) == 10


def test_splits_preserve_all_records():
    records = _build_records()

    splits = split_dataset(records)

    combined = (
        splits.train
        + splits.validation
        + splits.test
    )

    assert len(combined) == len(records)
    assert set(combined) == set(records)


def test_splits_are_stratified():
    records = _build_records()

    splits = split_dataset(records)

    expected_distribution = Counter(
        record.label
        for record in records
    )

    for split in (
        splits.train,
        splits.validation,
        splits.test,
    ):
        distribution = Counter(
            record.label
            for record in split
        )

        assert set(distribution) == set(
            expected_distribution
        )

        for label in expected_distribution:
            assert distribution[label] > 0


def test_split_is_deterministic():
    records = _build_records()

    config = SplitConfig(seed=123)

    first = split_dataset(records, config)
    second = split_dataset(records, config)

    assert first == second


def test_different_seeds_can_produce_different_splits():
    records = _build_records()

    first = split_dataset(
        records,
        SplitConfig(seed=1),
    )

    second = split_dataset(
        records,
        SplitConfig(seed=2),
    )

    assert first != second


def test_custom_split_sizes():
    records = _build_records()

    config = SplitConfig(
        validation_size=0.2,
        test_size=0.2,
    )

    splits = split_dataset(records, config)

    assert len(splits.train) == 60
    assert len(splits.validation) == 20
    assert len(splits.test) == 20


def test_empty_dataset():
    splits = split_dataset([])

    assert splits == DatasetSplits(
        train=[],
        validation=[],
        test=[],
    )