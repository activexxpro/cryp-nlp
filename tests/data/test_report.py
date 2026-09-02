from nlp_platform.data.records import TextRecord
from nlp_platform.data.report import (
    DataQualityReport,
    generate_quality_report,
)


def test_generate_quality_report():
    records = [
        TextRecord(text="Technology news", label=0),
        TextRecord(text="Sports news", label=1),
        TextRecord(text="Business news", label=2),
    ]

    report = generate_quality_report(records)

    assert isinstance(report, DataQualityReport)
    assert report.statistics.total_records == 3
    assert report.statistics.num_classes == 3
    assert report.empty_text_count == 0
    assert report.negative_label_count == 0
    assert report.is_valid is True


def test_empty_dataset_is_valid():
    report = generate_quality_report([])

    assert report.statistics.total_records == 0
    assert report.empty_text_count == 0
    assert report.negative_label_count == 0
    assert report.is_valid is True


def test_report_statistics_are_correct():
    records = [
        TextRecord(text="Short", label=0),
        TextRecord(text="A longer text", label=0),
        TextRecord(text="Another example", label=1),
    ]

    report = generate_quality_report(records)

    assert report.statistics.total_records == 3
    assert report.statistics.num_classes == 2
    assert report.statistics.label_counts == {
        0: 2,
        1: 1,
    }


def test_report_is_immutable():
    records = [
        TextRecord(text="Example", label=0),
    ]

    report = generate_quality_report(records)

    try:
        report.empty_text_count = 10
    except AttributeError:
        pass
    else:
        raise AssertionError("DataQualityReport should be immutable")