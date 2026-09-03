import json
from pathlib import Path
from collections.abc import Sequence

from nlp_platform.data.records import TextRecord


class DatasetPersistence:
    """Persist text records as JSON Lines."""

    def save(
        self,
        records: Sequence[TextRecord],
        path: str | Path,
    ) -> None:
        """Save records to a JSONL file."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            for record in records:
                json.dump(
                    {
                        "text": record.text,
                        "label": record.label,
                    },
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")

    def load(
        self,
        path: str | Path,
    ) -> list[TextRecord]:
        """Load records from a JSONL file."""

        input_path = Path(path)

        if not input_path.exists():
            raise FileNotFoundError(
                f"dataset file does not exist: {input_path}"
            )

        records: list[TextRecord] = []

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    data = json.loads(line)
                    records.append(
                        TextRecord(
                            text=data["text"],
                            label=data["label"],
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"invalid dataset record at line {line_number}"
                    ) from exc

        return records