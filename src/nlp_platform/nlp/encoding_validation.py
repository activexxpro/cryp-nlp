from collections.abc import Sequence

from nlp_platform.nlp.encoded_dataset import EncodedDataset


class EncodingValidationError(ValueError):
    """Raised when encoded dataset validation fails."""


def validate_encoded_dataset(
    dataset: EncodedDataset,
    expected_sequence_length: int | None = None,
) -> None:
    """Validate an encoded dataset."""

    if dataset.size != len(dataset.inputs):
        raise EncodingValidationError(
            "dataset size does not match number of inputs"
        )

    if dataset.size == 0:
        return

    sequence_lengths = {
        len(item.input_ids)
        for item in dataset.inputs
    }

    if expected_sequence_length is not None:
        if expected_sequence_length < 1:
            raise ValueError(
                "expected_sequence_length must be at least 1"
            )

        if sequence_lengths != {expected_sequence_length}:
            raise EncodingValidationError(
                "encoded sequences do not have the expected length"
            )

    labels = dataset.labels

    if len(labels) != dataset.size:
        raise EncodingValidationError(
            "number of labels does not match dataset size"
        )


def validate_batch_dimensions(
    input_ids: Sequence[Sequence[int]],
    labels: Sequence[int],
) -> None:
    """Validate that model inputs and labels have matching dimensions."""

    if len(input_ids) != len(labels):
        raise EncodingValidationError(
            "number of inputs must match number of labels"
        )

    if not input_ids:
        return

    sequence_lengths = {
        len(sequence)
        for sequence in input_ids
    }

    if len(sequence_lengths) != 1:
        raise EncodingValidationError(
            "all input sequences must have the same length"
        )