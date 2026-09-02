import pytest

from nlp_platform.data.records import TextRecord


def test_valid_record():
    record = TextRecord(
        text="The company reported strong earnings.",
        label=2,
    )

    assert record.text == "The company reported strong earnings."
    assert record.label == 2


def test_empty_text_rejected():
    with pytest.raises(ValueError):
        TextRecord(text="", label=0)


def test_whitespace_text_rejected():
    with pytest.raises(ValueError):
        TextRecord(text="   ", label=0)


def test_invalid_text_type_rejected():
    with pytest.raises(TypeError):
        TextRecord(text=123, label=0)


def test_invalid_label_type_rejected():
    with pytest.raises(TypeError):
        TextRecord(text="hello", label="0")