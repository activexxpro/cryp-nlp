import re

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and surrounding characters."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    normalized = text.strip()
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)

    return normalized