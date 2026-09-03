import pytest

from nlp_platform.data.labels import LabelMapper


@pytest.fixture
def mapper() -> LabelMapper:
    return LabelMapper(
        labels={
            0: "world",
            1: "sports",
            2: "business",
            3: "science",
        }
    )


def test_label_id_to_name(mapper: LabelMapper):
    assert mapper.to_name(0) == "world"
    assert mapper.to_name(1) == "sports"
    assert mapper.to_name(2) == "business"
    assert mapper.to_name(3) == "science"


def test_label_name_to_id(mapper: LabelMapper):
    assert mapper.to_id("world") == 0
    assert mapper.to_id("sports") == 1
    assert mapper.to_id("business") == 2
    assert mapper.to_id("science") == 3


def test_unknown_label_id_is_rejected(mapper: LabelMapper):
    with pytest.raises(ValueError, match="unknown label ID"):
        mapper.to_name(99)


def test_unknown_label_name_is_rejected(mapper: LabelMapper):
    with pytest.raises(ValueError, match="unknown label name"):
        mapper.to_id("unknown")


def test_num_classes(mapper: LabelMapper):
    assert mapper.num_classes == 4


def test_empty_mapping_is_rejected():
    with pytest.raises(ValueError, match="must not be empty"):
        LabelMapper(labels={})


def test_invalid_label_id_is_rejected():
    with pytest.raises(TypeError, match="must be integers"):
        LabelMapper(
            labels={
                "0": "world",
            }
        )


def test_empty_label_name_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        LabelMapper(
            labels={
                0: "",
            }
        )