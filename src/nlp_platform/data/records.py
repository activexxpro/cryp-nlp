from dataclasses import dataclass


@dataclass(frozen=True)
class TextRecord:
    text: str
    label: int

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")

        if not self.text.strip():
            raise ValueError("text must not be empty")

        if not isinstance(self.label, int):
            raise TypeError("label must be an integer")