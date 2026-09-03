import argparse
from pathlib import Path

from nlp_platform.data.persistence import DatasetPersistence
from nlp_platform.data.preprocessing import DatasetPreprocessingPipeline


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Preprocess an NLP dataset stored as JSONL.",
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL dataset path.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL dataset path.",
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
        "--keep-duplicates",
        action="store_true",
        help="Keep duplicate records.",
    )

    return parser


def main() -> None:
    """Run dataset preprocessing from the command line."""

    parser = build_parser()
    args = parser.parse_args()

    persistence = DatasetPersistence()

    records = persistence.load(args.input)

    pipeline = DatasetPreprocessingPipeline(
        min_length=args.min_length,
        max_length=args.max_length,
        remove_duplicate_records=not args.keep_duplicates,
    )

    processed_records = pipeline.process(records)

    persistence.save(
        processed_records,
        args.output,
    )

    print(f"Input records: {len(records)}")
    print(f"Output records: {len(processed_records)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()