from nlp_platform.data.distribution import (
    ClassDistribution,
    calculate_class_distribution,
)
from nlp_platform.data.records import TextRecord


def test_calculate_class_distribution():
    records = [
        TextRecord(text="a", label=0),
        TextRecord(text="b", label=0),
        TextRecord(text="c", label=1),
        TextRecord(text="d", label=1),
        TextRecord(text="e", label=1),
        TextRecord(text="f", label=2),
    ]

    distribution = calculate_class_distribution(records)

    assert distribution.counts == {
        0: 2,
        1: 3,
        2: 1,
    }

    assert distribution.total_records == 6
    assert distribution.num_classes == 3


def test_majority_class():
    distribution = ClassDistribution(
        counts={
            0: 10,
            1: 20,
            2: 5,
        },
        total_records=35,
    )

    assert distribution.majority_class == 1


def test_minority_class():
    distribution = ClassDistribution(
        counts={
            0: 10,
            1: 20,
            2: 5,
        },
        total_records=35,
    )

    assert distribution.minority_class == 2


def test_imbalance_ratio():
    distribution = ClassDistribution(
        counts={
            0: 10,
            1: 20,
            2: 5,
        },
        total_records=35,
    )

    assert distribution.imbalance_ratio == 4.0


def test_empty_distribution():
    distribution = calculate_class_distribution([])

    assert distribution.counts == {}
    assert distribution.total_records == 0
    assert distribution.num_classes == 0
    assert distribution.majority_class is None
    assert distribution.minority_class is None
    assert distribution.imbalance_ratio == 0.0


def test_balanced_distribution():
    distribution = ClassDistribution(
        counts={
            0: 10,
            1: 10,
            2: 10,
        },
        total_records=30,
    )

    assert distribution.imbalance_ratio == 1.0


def test_distribution_is_immutable():
    distribution = ClassDistribution(
        counts={0: 10},
        total_records=10,
    )

    try:
        distribution.total_records = 20
    except AttributeError:
        pass
    else:
        raise AssertionError(
            "ClassDistribution should be immutable"
        )