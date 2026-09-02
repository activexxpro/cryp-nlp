from collections.abc import Callable, Sequence

from nlp_platform.data.normalizer import normalize_text


TextTransform = Callable[[str], str]


def lowercase_text(text: str) -> str:
    """Convert text to lowercase."""

    return text.lower()


def remove_extra_punctuation(text: str) -> str:
    """Normalize repeated punctuation marks."""

    result = text

    while "!!" in result:
        result = result.replace("!!", "!")

    while "??" in result:
        result = result.replace("??", "?")

    return result


class TextCleaningPipeline:
    """Apply a sequence of text transformations."""

    def __init__(
        self,
        transforms: Sequence[TextTransform] | None = None,
    ) -> None:
        self._transforms = list(
            transforms
            if transforms is not None
            else [normalize_text]
        )

    def transform(self, text: str) -> str:
        """Apply all configured transformations."""

        result = text

        for transform in self._transforms:
            result = transform(result)

        return result

    def transform_many(
        self,
        texts: Sequence[str],
    ) -> list[str]:
        """Apply the pipeline to multiple texts."""

        return [self.transform(text) for text in texts]