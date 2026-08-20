"""Mechanical helpers for resolving and sampling FLenQA positions."""

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from jlens_reasoning.experiments_utils.spans import (
    CharSpan,
    char_span_to_token_span,
    find_all_spans,
)


@dataclass(frozen=True, slots=True)
class ResolvedSpan:
    """One labeled span resolved in character and token coordinates."""

    label: str
    char_span: CharSpan
    token_span: CharSpan


def find_required_span(
    text: str,
    needle: str,
    *,
    name: str,
    occurrence: Literal["unique", "first", "last"] = "unique",
) -> CharSpan:
    """Find required text using the requested occurrence policy."""
    matches = find_all_spans(text, needle) if needle else ()
    if not matches:
        raise ValueError(f"Required FLenQA {name} span is unresolved")
    if len(matches) > 1 and occurrence == "unique":
        raise ValueError(f"Required FLenQA {name} span is ambiguous")
    return matches[-1] if occurrence == "last" else matches[0]


def resolve_span(
    label: str,
    char_span: CharSpan,
    offsets: Sequence[tuple[int, int]],
) -> ResolvedSpan:
    """Attach token coordinates to a labeled character span."""
    return ResolvedSpan(
        label=label,
        char_span=char_span,
        token_span=char_span_to_token_span(offsets, char_span),
    )


def non_fact_paragraphs(
    paragraphs: Sequence[CharSpan],
    facts: Sequence[ResolvedSpan],
) -> tuple[CharSpan, ...]:
    """Return context paragraphs that are not resolved fact paragraphs."""
    fact_bounds = {(fact.char_span.start, fact.char_span.end) for fact in facts}
    return tuple(
        paragraph
        for paragraph in paragraphs
        if (paragraph.start, paragraph.end) not in fact_bounds
    )


def _eligible_padding_position(
    *,
    token_id: int,
    offset: tuple[int, int],
    text: str,
    padding_spans: Sequence[CharSpan],
    special_ids: frozenset[int],
) -> bool:
    """Return whether one token is usable as padding content."""
    start, end = offset
    return (
        token_id not in special_ids
        and end > start
        and bool(text[start:end].strip())
        and any(start < span.end and end > span.start for span in padding_spans)
    )


def padding_token_positions(
    *,
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    text: str,
    padding_spans: Sequence[CharSpan],
    special_ids: frozenset[int],
) -> tuple[int, ...]:
    """Return token indices that overlap eligible padding content."""
    positions = []
    for position, (token_id, offset) in enumerate(zip(input_ids, offsets, strict=True)):
        if _eligible_padding_position(
            token_id=token_id,
            offset=offset,
            text=text,
            padding_spans=padding_spans,
            special_ids=special_ids,
        ):
            positions.append(position)
    return tuple(sorted(positions))


def sample_padding_positions(
    positions: Sequence[int],
    *,
    prompt_id: str,
    sample_seed: int,
    count: int,
) -> tuple[int, ...]:
    """Sample sorted padding positions from a prompt-stable random seed."""
    rng = random.Random(int(prompt_id[:16], 16) ^ sample_seed)
    sample_size = min(count, len(positions))
    return tuple(sorted(rng.sample(positions, sample_size)))
