from pathlib import Path

import pytest

from nlp_platform.data.persistence import DatasetPersistence
from nlp_platform.data.records import TextRecord


@pytest.fixture
def persistence() -> DatasetPersistence:
    return DatasetPersistence()


@pytest.fixture
def records() -> list[TextRecord]:
    return [
        TextRecord(
            text="hello world",
            label=0,
        ),
        TextRecord(
            text="machine learning",
            label=1,
        ),
        TextRecord(
            text="natural language processing",
            label=2,
        ),
    ]


def test_save_and_load_round_trip(
    persistence: DatasetPersistence,
    records: list[TextRecord],
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    persistence.save(records, path)

    loaded = persistence.load(path)

    assert loaded == records


def test_save_creates_parent_directories(
    persistence: DatasetPersistence,
    records: list[TextRecord],
    tmp_path: Path,
):
    path = tmp_path / "nested" / "data" / "dataset.jsonl"

    persistence.save(records, path)

    assert path.exists()
    assert persistence.load(path) == records


def test_load_skips_blank_lines(
    persistence: DatasetPersistence,
    tmp_path: Path,
):
    path = tmp_path / "dataset.jsonl"

    path.write_text(
        '{"text": "hello", "label": 0}\n'
        "\n"
        '  \n'
        '{"text": "world", "label": 1}\n',
        encoding="utf-8",
    )

    result = persistence.load(path)

    assert result == [
        TextRecord(text="hello", label=0),
        TextRecord(text="world", label=1),
    ]


def test_load_missing_file_raises(
    persistence: DatasetPersistence,
    tmp_path: Path,
):
    path = tmp_path / "missing.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        persistence.load(path)


def test_load_invalid_json_raises(
    persistence: DatasetPersistence,
    tmp_path: Path,
):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        "not valid json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid dataset record at line 1",
    ):
        persistence.load(path)


def test_load_missing_field_raises(
    persistence: DatasetPersistence,
    tmp_path: Path,
):
    path = tmp_path / "invalid.jsonl"

    path.write_text(
        '{"text": "hello"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="invalid dataset record at line 1",
    ):
        persistence.load(path)


def test_unicode_text_round_trip(
    persistence: DatasetPersistence,
    tmp_path: Path,
):
    records = [
        TextRecord(
            text="Café résumé — 日本語",
            label=0,
        ),
    ]

    path = tmp_path / "unicode.jsonl"

    persistence.save(records, path)

    assert persistence.load(path) == records


def test_empty_dataset_round_trip(
    persistence: DatasetPersistence,
    tmp_path: Path,
):
    path = tmp_path / "empty.jsonl"

    persistence.save([], path)

    assert persistence.load(path) == []