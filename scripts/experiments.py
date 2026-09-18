import argparse
from pathlib import Path

from nlp_platform.ml.experiment_registry import ExperimentRegistry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect persisted NLP experiments.",
    )

    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Directory containing experiment JSON files.",
    )

    parser.add_argument(
        "command",
        choices=("list", "show"),
        help="Experiment operation to perform.",
    )

    parser.add_argument(
        "--name",
        help="Experiment JSON filename for the show command.",
    )

    return parser


def list_experiments(registry: ExperimentRegistry) -> None:
    experiments = registry.list_experiments()

    if not experiments:
        print("No experiments found.")
        return

    for filename in experiments:
        print(filename)


def show_experiment(
    registry: ExperimentRegistry,
    filename: str | None,
) -> None:
    if not filename:
        raise ValueError("--name is required for the show command.")

    result = registry.load_experiment(filename)

    print(f"Experiment: {result.experiment_name}")
    print(f"Seed: {result.seed}")
    print(f"Epochs completed: {result.epochs_completed}")
    print(f"Stopped early: {result.stopped_early}")
    print(f"Best epoch: {result.best_epoch}")
    print(f"Best validation loss: {result.best_validation_loss}")
    print(f"Final validation loss: {result.final_validation_loss}")
    print(
        "Final validation accuracy: "
        f"{result.final_validation_accuracy}"
    )
    print(f"Best checkpoint: {result.best_checkpoint}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    registry = ExperimentRegistry(args.directory)

    if args.command == "list":
        list_experiments(registry)
        return

    show_experiment(registry, args.name)


if __name__ == "__main__":
    main()