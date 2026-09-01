from datasets import Dataset

from nlp_platform.data.splitter import split_train_dataset


def test_train_validation_split():
    dataset = Dataset.from_dict(
        {
            "text": [
                "text one",
                "text two",
                "text three",
                "text four",
                "text five",
                "text six",
                "text seven",
                "text eight",
                "text nine",
                "text ten",
            ],
            "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    result = split_train_dataset(
        dataset,
        validation_size=0.2,
        seed=42,
    )

    assert len(result["train"]) == 8
    assert len(result["validation"]) == 2