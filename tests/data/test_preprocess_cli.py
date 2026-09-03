import os
import json
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


def test_preprocess_cli(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        '{"text": "  HELLO    WORLD  ", "label": 0}\n'
        '{"text": "HELLO WORLD", "label": 0}\n'
        '{"text": "Hi", "label": 1}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-length",
            "5",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_cli_environment(),
    )

    print("STDOUT:")
    print(result.stdout)

    print("STDERR:")
    print(result.stderr)

    assert result.returncode == 0
    assert "Input records: 3" in result.stdout
    assert "Output records: 1" in result.stdout

    output_records = [
        json.loads(line)
        for line in output_path.read_text(
            encoding="utf-8"
        ).splitlines()
    ]

    assert output_records == [
        {
            "text": "hello world",
            "label": 0,
        }
    ]


def test_preprocess_cli_can_keep_duplicates(tmp_path: Path):
    input_path = tmp_path / "input.jsonl"
    output_path = tmp_path / "output.jsonl"

    input_path.write_text(
        '{"text": "Hello", "label": 0}\n'
        '{"text": "hello", "label": 0}\n',
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--keep-duplicates",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=_cli_environment(),
    )

    assert result.returncode == 0
    assert "Input records: 2" in result.stdout
    assert "Output records: 2" in result.stdout


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