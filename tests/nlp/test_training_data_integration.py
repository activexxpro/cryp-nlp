from nlp_platform.data.dataset_preparation import (
    DatasetPreparationPipeline,
)
from nlp_platform.data.records import TextRecord
from nlp_platform.data.split_config import SplitConfig
from nlp_platform.nlp.batch_encoder import BatchEncoder
from nlp_platform.nlp.dataset_encoder import DatasetEncoder
from nlp_platform.nlp.encoded_dataset import DatasetEncodingService
from nlp_platform.nlp.encoder import TokenEncoder
from nlp_platform.nlp.nlp_pipeline import NLPPreprocessingPipeline
from nlp_platform.nlp.padding import SequencePadder
from nlp_platform.nlp.tokenizer import BasicTokenizer
from nlp_platform.nlp.training_dataset_encoder import (
    TrainingDatasetEncoder,
)
from nlp_platform.nlp.vocabulary import Vocabulary


def create_training_encoder(
    records: list[TextRecord],
) -> TrainingDatasetEncoder:
    token_sequences = [
        BasicTokenizer().tokenize(record.text)
        for record in records
    ]

    vocabulary = Vocabulary.build(
        token_sequences
    )

    tokenizer = BasicTokenizer()
    token_encoder = TokenEncoder(vocabulary)

    padder = SequencePadder(
        vocabulary=vocabulary,
        max_length=6,
    )

    batch_encoder = BatchEncoder(
        tokenizer=tokenizer,
        encoder=token_encoder,
        padder=padder,
    )

    pipeline = NLPPreprocessingPipeline(
        batch_encoder=batch_encoder,
    )

    dataset_encoder = DatasetEncoder(
        pipeline=pipeline,
    )

    encoding_service = DatasetEncodingService(
        encoder=dataset_encoder,
    )

    return TrainingDatasetEncoder(
        encoding_service=encoding_service,
    )


def create_records() -> list[TextRecord]:
    return [
        TextRecord(
            text=f"Technology news article {index}",
            label=0,
        )
        for index in range(50)
    ] + [
        TextRecord(
            text=f"Sports news article {index}",
            label=1,
        )
        for index in range(50)
    ]


def test_full_training_data_pipeline() -> None:
    records = create_records()

    preparation = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.1,
            test_size=0.1,
            seed=42,
        ),
    )

    splits = preparation.prepare(records)

    encoder = create_training_encoder(records)

    training_dataset = encoder.encode(splits)

    assert training_dataset.train_size == 80
    assert training_dataset.validation_size == 10
    assert training_dataset.test_size == 10
    assert training_dataset.total_size == 100


def test_labels_are_preserved_through_pipeline() -> None:
    records = create_records()

    preparation = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.1,
            test_size=0.1,
            seed=42,
        ),
    )

    splits = preparation.prepare(records)

    encoder = create_training_encoder(records)

    training_dataset = encoder.encode(splits)

    assert set(training_dataset.train.labels) == {0, 1}
    assert set(training_dataset.validation.labels) == {0, 1}
    assert set(training_dataset.test.labels) == {0, 1}


def test_encoded_sequences_have_consistent_length() -> None:
    records = create_records()

    preparation = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.1,
            test_size=0.1,
            seed=42,
        ),
    )

    splits = preparation.prepare(records)

    encoder = create_training_encoder(records)

    training_dataset = encoder.encode(splits)

    for dataset in (
        training_dataset.train,
        training_dataset.validation,
        training_dataset.test,
    ):
        for input_ids in dataset.input_ids:
            assert len(input_ids) == 6


def test_preprocessing_removes_duplicates_before_encoding() -> None:
    records = create_records()

    records.extend(
        [
            TextRecord(
                text="Technology news article 0",
                label=0,
            ),
            TextRecord(
                text="Sports news article 0",
                label=1,
            ),
        ]
    )

    preparation = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.1,
            test_size=0.1,
            seed=42,
        ),
    )

    splits = preparation.prepare(records)

    total_records = (
        len(splits.train)
        + len(splits.validation)
        + len(splits.test)
    )

    assert total_records == 100


def test_preprocessing_normalizes_text_before_encoding() -> None:
    records = [
        TextRecord(
            text="  Technology    News   Article  ",
            label=0,
        ),
        TextRecord(
            text="  Sports    News   Article  ",
            label=1,
        ),
    ]

    preparation = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.2,
            test_size=0.2,
            seed=42,
        ),
    )

    prepared = preparation.build(records)

    assert prepared.size == 2

    texts = prepared.texts

    assert "  " not in texts[0]
    assert "  " not in texts[1]

    assert texts[0] == "technology news article"
    assert texts[1] == "sports news article"


def test_pipeline_is_deterministic() -> None:
    records = create_records()

    config = SplitConfig(
        validation_size=0.1,
        test_size=0.1,
        seed=123,
    )

    preparation = DatasetPreparationPipeline(
        split_config=config,
    )

    first = preparation.prepare(records)
    second = preparation.prepare(records)

    first_train = [
        record.text
        for record in first.train
    ]

    second_train = [
        record.text
        for record in second.train
    ]

    assert first_train == second_train


def test_different_seed_changes_dataset_split() -> None:
    records = create_records()

    first = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.1,
            test_size=0.1,
            seed=42,
        ),
    ).prepare(records)

    second = DatasetPreparationPipeline(
        split_config=SplitConfig(
            validation_size=0.1,
            test_size=0.1,
            seed=99,
        ),
    ).prepare(records)

    first_train = [
        record.text
        for record in first.train
    ]

    second_train = [
        record.text
        for record in second.train
    ]

    assert first_train != second_train