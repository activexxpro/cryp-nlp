import json
import os
import subprocess
import sys
from pathlib import Path


def _cli_environment() -> dict[str, str]:
    """Build an environment for running project CLI commands."""

    env = os.environ.copy()

    project_root = Path(__file__).resolve().parents[2]
    src_path = project_root / "src"

    env["PYTHONPATH"] = str(src_path)

    return env


def test_preprocess_cli(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"

    records = [
        {
            "text": f"Example document number {index}",
            "label": index % 2,
        }
        for index in range(100)
    ]

    input_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
        env=_cli_environment(),
        check=False,
    )

    assert result.returncode == 0, result.stderr

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "metadata.json").exists()

    train_records = (output_dir / "train.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    validation_records = (
        output_dir / "validation.jsonl"
    ).read_text(
        encoding="utf-8"
    ).strip().splitlines()

    test_records = (output_dir / "test.jsonl").read_text(
        encoding="utf-8"
    ).strip().splitlines()

    assert len(train_records) == 80
    assert len(validation_records) == 10
    assert len(test_records) == 10


def test_preprocess_cli_can_keep_duplicates(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"

    records = [
        {
            "text": f"Class zero document {index}",
            "label": 0,
        }
        for index in range(50)
    ]

    records += [
        {
            "text": f"Class one document {index}",
            "label": 1,
        }
        for index in range(50)
    ]

    records += [
        {
            "text": "duplicate document",
            "label": 0,
        },
        {
            "text": "duplicate document",
            "label": 0,
        },
    ]

    input_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--keep-duplicates",
        ],
        capture_output=True,
        text=True,
        env=_cli_environment(),
        check=False,
    )

    assert result.returncode == 0, result.stderr

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()

    all_output_records = []

    for split_name in (
        "train",
        "validation",
        "test",
    ):
        split_path = output_dir / f"{split_name}.jsonl"

        lines = split_path.read_text(
            encoding="utf-8"
        ).strip().splitlines()

        all_output_records.extend(
            json.loads(line)
            for line in lines
        )

    assert len(all_output_records) == 102

    duplicate_count = sum(
        record["text"] == "duplicate document"
        for record in all_output_records
    )

    assert duplicate_count == 2
    

def test_preprocess_cli_missing_input_fails(tmp_path: Path):
    input_path = tmp_path / "missing.jsonl"
    output_path = tmp_path / "output.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0

def test_preprocess_cli_creates_dataset_splits(tmp_path):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "prepared"

    records = []

    for index in range(100):
        records.append(
            {
                "text": f"Example document {index}",
                "label": index % 2,
            }
        )

    input_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in records
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--name",
            "test-dataset",
            "--version",
            "1.0.0",
            "--source",
            "unit-test",
        ],
        capture_output=True,
        text=True,
        env=_cli_environment(),
        check=False,
    )

    assert result.returncode == 0

    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "validation.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "metadata.json").exists()