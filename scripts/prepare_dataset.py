from pathlib import Path

from nlp_platform.data.loader import load_news_dataset
from nlp_platform.data.splitter import split_train_dataset


OUTPUT_DIR = Path("data/processed")


def main() -> None:
    dataset = load_news_dataset()

    train_validation = split_train_dataset(
        dataset["train"],
        validation_size=0.1,
        seed=42,
    )

    processed = {
        "train": train_validation["train"],
        "validation": train_validation["validation"],
        "test": dataset["test"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_name, split_dataset in processed.items():
        path = OUTPUT_DIR / split_name
        split_dataset.save_to_disk(str(path))

        print(
            f"{split_name}: "
            f"{len(split_dataset)} samples"
        )


if __name__ == "__main__":
    main()