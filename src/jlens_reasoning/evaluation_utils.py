import re
import unicodedata
from enum import StrEnum


class ReasoningStatus(StrEnum):
    NOT_PRESENT = "not_present"
    PARSED = "parsed"
    MALFORMED = "malformed_reasoning"


def no_reasoning(text: str) -> tuple[str, ReasoningStatus]:
    return text, ReasoningStatus.NOT_PRESENT


# Matches complete, non-nested reasoning spans. Any leftover tag is malformed.
_THINK_SPAN = re.compile(r"<think>(?:(?!</?think>).)*</think>", re.DOTALL)


def parse_think_tags(text: str) -> tuple[str, ReasoningStatus]:
    if "<think>" not in text and "</think>" not in text:
        return text, ReasoningStatus.NOT_PRESENT
    visible = _THINK_SPAN.sub("", text)
    if "<think>" in visible or "</think>" in visible:
        return "", ReasoningStatus.MALFORMED
    return visible, ReasoningStatus.PARSED


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFC", text).strip().casefold().rstrip(".!?")


def extract_answer(evaluation_text: str) -> str | None:
    answer = re.split(r"[.!?\n]", evaluation_text, maxsplit=1)[0].strip()
    return answer or None


def matches_reference(normalized_answer: str, references: tuple[str, ...]) -> bool:
    return any(
        normalize_text(reference) == normalized_answer for reference in references
    )


def safe_truncated_text(text: str) -> str | None:
    boundary = max(text.rfind(character) for character in ".!?\n")
    return None if boundary < 0 else text[: boundary + 1].strip()
