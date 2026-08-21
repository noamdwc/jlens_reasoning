"""Immutable character/token span primitives."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

SpanKind = Literal["fact", "question", "rule", "context"]


@dataclass(frozen=True, slots=True)
class CharSpan:
    """A half-open character or token interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if (
            type(self.start) is not int
            or type(self.end) is not int
            or self.start < 0
            or self.end < self.start
        ):
            raise ValueError("span bounds must be non-negative ordered integers")


class SpanStatus(StrEnum):
    OK = "ok"
    AMBIGUOUS = "ambiguous"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class SpanDiagnostic:
    """Arrow-friendly resolution result for one logical prompt span."""

    kind: SpanKind
    ordinal: int
    surface: str
    status: SpanStatus
    match_count: int
    char_start: int | None
    char_end: int | None
    token_start: int | None
    token_end: int | None

    def __post_init__(self) -> None:
        if self.kind not in {"fact", "question", "rule", "context"}:
            raise ValueError(f"unknown span kind: {self.kind!r}")
        if type(self.ordinal) is not int or self.ordinal < 0:
            raise ValueError("span ordinal must be a non-negative integer")
        if type(self.match_count) is not int or self.match_count < 0:
            raise ValueError("span match count must be a non-negative integer")
        _validate_optional_bounds(
            self.char_start,
            self.char_end,
            name="character",
        )
        _validate_optional_bounds(
            self.token_start,
            self.token_end,
            name="token",
        )


def _validate_optional_bounds(
    start: int | None,
    end: int | None,
    *,
    name: str,
) -> None:
    if (start is None) != (end is None):
        raise ValueError(f"{name} span bounds must both be null or both be integers")
    if start is not None:
        CharSpan(start, end)  # type: ignore[arg-type]


def find_all_spans(text: str, needle: str) -> tuple[CharSpan, ...]:
    """Return all occurrences, including occurrences with overlapping starts."""
    if not needle:
        raise ValueError("cannot find spans for an empty needle")
    spans: list[CharSpan] = []
    start = 0
    while True:
        match_start = text.find(needle, start)
        if match_start < 0:
            return tuple(spans)
        spans.append(CharSpan(match_start, match_start + len(needle)))
        start = match_start + 1


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    char_span: CharSpan,
) -> CharSpan:
    """Map a character span through one full-prompt offset mapping."""
    covered = [
        token_index
        for token_index, (start, end) in enumerate(offsets)
        if end > start and start < char_span.end and end > char_span.start
    ]
    if not covered:
        raise ValueError(f"character span {char_span!r} has no token coverage")
    return CharSpan(covered[0], covered[-1] + 1)


def parse_paragraph_payload_spans(text: str) -> tuple[CharSpan, ...]:
    """Return trimmed payloads from newline-delimited lines.

    Bounds are relative to ``text``. Newline delimiters, blank lines, and
    leading/trailing horizontal whitespace are structural gaps; whitespace
    inside a payload is retained.
    """
    payloads: list[CharSpan] = []
    for line_match in re.finditer(r"[^\r\n]+", text):
        raw_start, raw_end = line_match.span()
        start = raw_start
        end = raw_end
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            payloads.append(CharSpan(start, end))
    return tuple(payloads)
