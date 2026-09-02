from nlp_platform.data.batch import prepare_records
from nlp_platform.data.cleaning import (
    TextCleaningPipeline,
    lowercase_text,
)
from nlp_platform.data.report import generate_quality_report


def test_end_to_end_data_pipeline():
    rows = [
        {
            "text": "technology stocks increased.",
            "label": 0,
        },
        {
            "text": "  The TEAM won the championship!  ",
            "label": 1,
        },
        {
            "text": "  Business markets remain stable.  ",
            "label": 2,
        },
    ]

    records = prepare_records(rows)

    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    cleaned_records = [
        record.__class__(
            text=pipeline.transform(record.text),
            label=record.label,
        )
        for record in records
    ]

    report = generate_quality_report(cleaned_records)

    assert report.is_valid is True
    assert report.statistics.total_records == 3
    assert report.statistics.num_classes == 3
    assert report.statistics.label_counts == {
        0: 1,
        1: 1,
        2: 1,
    }

    assert cleaned_records[0].text == (
        "technology stocks increased."
    )
    assert cleaned_records[1].text == (
        "the team won the championship!"
    )
    assert cleaned_records[2].text == (
        "business markets remain stable."
    )


def test_pipeline_preserves_labels():
    rows = [
        {
            "text": " FIRST ARTICLE ",
            "label": 3,
        },
        {
            "text": " SECOND ARTICLE ",
            "label": 1,
        },
    ]

    records = prepare_records(rows)

    pipeline = TextCleaningPipeline(
        transforms=[
            str.strip,
            lowercase_text,
        ]
    )

    cleaned_records = [
        record.__class__(
            text=pipeline.transform(record.text),
            label=record.label,
        )
        for record in records
    ]

    assert [record.label for record in cleaned_records] == [
        3,
        1,
    ]