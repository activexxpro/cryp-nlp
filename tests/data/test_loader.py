from nlp_platform.data.loader import EXPECTED_COLUMNS


def test_expected_columns():
    assert EXPECTED_COLUMNS == {"text", "label"}