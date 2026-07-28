from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from jlens_reasoning.experiments_utils.spans import (
    CharSpan,
    SpanDiagnostic,
    SpanStatus,
    char_span_to_token_span,
    find_all_spans,
    parse_paragraph_payload_spans,
)


def test_span_contracts_are_immutable_and_use_stable_status_values() -> None:
    span = CharSpan(2, 5)
    diagnostic = SpanDiagnostic(
        kind="fact",
        ordinal=0,
        fact_ordinal=None,
        surface="cat",
        status=SpanStatus.OK,
        match_count=1,
        char_start=2,
        char_end=5,
        token_start=3,
        token_end=4,
    )

    with pytest.raises(FrozenInstanceError):
        span.start = 3  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        diagnostic.match_count = 2  # type: ignore[misc]
    assert tuple(status.value for status in SpanStatus) == (
        "ok",
        "ambiguous",
        "unresolved",
    )


def test_find_all_spans_includes_overlapping_occurrences() -> None:
    assert find_all_spans("ababa", "aba") == (
        CharSpan(0, 3),
        CharSpan(2, 5),
    )


def test_find_all_spans_rejects_empty_needle() -> None:
    with pytest.raises(ValueError, match="empty"):
        find_all_spans("anything", "")


def test_char_span_to_token_span_ignores_special_tokens_and_uses_overlap() -> None:
    offsets = ((0, 0), (0, 2), (2, 4), (4, 4), (4, 7), (0, 0))

    assert char_span_to_token_span(offsets, CharSpan(1, 5)) == CharSpan(1, 5)
    assert char_span_to_token_span(offsets, CharSpan(2, 4)) == CharSpan(2, 3)


def test_char_span_to_token_span_rejects_spans_without_token_coverage() -> None:
    with pytest.raises(ValueError, match="coverage"):
        char_span_to_token_span(((0, 0), (2, 4)), CharSpan(0, 2))


def test_paragraph_payload_parser_preserves_offsets_and_excludes_structure() -> None:
    text = "\n  first  payload \r\n \t\r\nsecond   payload\t\n"

    spans = parse_paragraph_payload_spans(text)

    assert spans == (
        CharSpan(text.index("first"), text.index("payload") + len("payload")),
        CharSpan(
            text.index("second"), text.index("payload\t", text.index("second")) + 7
        ),
    )
    assert tuple(text[span.start : span.end] for span in spans) == (
        "first  payload",
        "second   payload",
    )


def test_paragraph_payload_parser_treats_whitespace_only_lines_as_structure() -> None:
    assert parse_paragraph_payload_spans("\n \t\r\n") == ()
