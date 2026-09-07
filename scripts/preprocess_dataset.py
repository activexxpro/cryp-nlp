import argparse
import json
from pathlib import Path

from nlp_platform.data.dataset_preparation import (
    DatasetPreparationPipeline,
)
from nlp_platform.data.persistence import DatasetPersistence


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Prepare an NLP dataset stored as JSONL.",
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL dataset path.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for prepared dataset splits.",
    )

    parser.add_argument(
        "--name",
        default="dataset",
        help="Dataset name.",
    )

    parser.add_argument(
        "--version",
        default="1.0.0",
        help="Dataset version.",
    )

    parser.add_argument(
        "--source",
        default="unknown",
        help="Dataset source.",
    )

    parser.add_argument(
        "--min-length",
        type=int,
        default=1,
        help="Minimum text length.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maximum text length.",
    )

    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.1,
        help="Validation split fraction.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test split fraction.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate records.",
    )

    return parser


def save_split(
    persistence: DatasetPersistence,
    records,
    output_dir: Path,
    split_name: str,
) -> None:
    """Save one prepared dataset split."""

    persistence.save(
        records,
        output_dir / f"{split_name}.jsonl",
    )


def main() -> None:
    """Run dataset preparation from the command line."""

    parser = build_parser()
    args = parser.parse_args()

    persistence = DatasetPersistence()

    records = persistence.load(args.input)

    pipeline = DatasetPreparationPipeline(
        min_length=args.min_length,
        max_length=args.max_length,
        remove_duplicate_records=not args.keep_duplicates,
    )

    splits = pipeline.prepare(
        records,
        name=args.name,
        version=args.version,
        source=args.source,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_split(
        persistence,
        splits.train,
        args.output_dir,
        "train",
    )

    save_split(
        persistence,
        splits.validation,
        args.output_dir,
        "validation",
    )

    save_split(
        persistence,
        splits.test,
        args.output_dir,
        "test",
    )

    metadata = pipeline.create_metadata(
        records,
        name=args.name,
        version=args.version,
        source=args.source,
    )

    metadata_path = args.output_dir / "metadata.json"

    metadata_path.write_text(
        json.dumps(
            {
                "name": metadata.name,
                "version": metadata.version,
                "source": metadata.source,
                "created_at": metadata.created_at,
                "num_records": metadata.num_records,
                "num_classes": metadata.num_classes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Input records: {len(records)}")
    print(f"Train records: {len(splits.train)}")
    print(f"Validation records: {len(splits.validation)}")
    print(f"Test records: {len(splits.test)}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()