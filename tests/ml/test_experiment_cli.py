import json
import subprocess
import sys
from pathlib import Path


def create_experiment(path: Path, name: str) -> None:
    payload = {
        "experiment_name": name,
        "seed": 42,
        "epochs_completed": 2,
        "stopped_early": False,
        "best_epoch": 2,
        "best_validation_loss": 0.25,
        "best_checkpoint": None,
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.5,
                "validation_loss": 0.4,
                "validation_accuracy": 0.8,
            },
            {
                "epoch": 2,
                "train_loss": 0.3,
                "validation_loss": 0.25,
                "validation_accuracy": 0.9,
            },
        ],
    }

    path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def run_cli(
    directory: Path,
    *arguments: str,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "scripts/experiments.py",
            "--directory",
            str(directory),
            *arguments,
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_lists_experiments(tmp_path: Path) -> None:
    create_experiment(
        tmp_path / "baseline.json",
        "baseline",
    )
    create_experiment(
        tmp_path / "transformer.json",
        "transformer",
    )

    result = run_cli(tmp_path, "list")

    assert result.returncode == 0
    assert "baseline.json" in result.stdout
    assert "transformer.json" in result.stdout


def test_cli_reports_empty_directory(tmp_path: Path) -> None:
    result = run_cli(tmp_path, "list")

    assert result.returncode == 0
    assert "No experiments found." in result.stdout


def test_cli_shows_experiment(tmp_path: Path) -> None:
    create_experiment(
        tmp_path / "baseline.json",
        "baseline",
    )

    result = run_cli(
        tmp_path,
        "show",
        "--name",
        "baseline.json",
    )

    assert result.returncode == 0
    assert "Experiment: baseline" in result.stdout
    assert "Seed: 42" in result.stdout
    assert "Epochs completed: 2" in result.stdout
    assert "Best epoch: 2" in result.stdout
    assert "Best validation loss: 0.25" in result.stdout
    assert "Final validation accuracy: 0.9" in result.stdout