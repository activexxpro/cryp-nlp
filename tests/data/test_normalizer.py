import pytest

from nlp_platform.data.normalizer import normalize_text


def test_strips_leading_and_trailing_whitespace():
    result = normalize_text("  hello world  ")

    assert result == "hello world"


def test_collapses_multiple_spaces():
    result = normalize_text("hello    world")

    assert result == "hello world"


def test_collapses_tabs():
    result = normalize_text("hello\t\tworld")

    assert result == "hello world"


def test_collapses_newlines():
    result = normalize_text("hello\n\nworld")

    assert result == "hello world"


def test_combined_whitespace_normalization():
    result = normalize_text(
        "  The   company\nreported\tstrong   earnings.  "
    )

    assert result == "The company reported strong earnings."


def test_empty_text_returns_empty_string():
    assert normalize_text("") == ""


def test_whitespace_only_text_returns_empty_string():
    assert normalize_text("   \n\t ") == ""


def test_non_string_input_is_rejected():
    with pytest.raises(TypeError):
        normalize_text(123)