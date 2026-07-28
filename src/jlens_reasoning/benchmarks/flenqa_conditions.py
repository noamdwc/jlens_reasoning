"""Token-position conditions derived from prepared FLenQA prompts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jlens_reasoning.benchmarks.flenqa_preparation import PreparedPrompt
from jlens_reasoning.experiments_utils.spans import CharSpan, SpanStatus

PLACEMENT_EPSILON = 0.02


@dataclass(frozen=True, slots=True)
class PromptConditions:
    padding_type_effective: str | None
    dispersion_effective: str
    frac_padding_before: float
    frac_padding_between: float
    frac_padding_after: float
    n_padding_tokens: int


def _fact_diagnostics(prepared: PreparedPrompt):
    return tuple(
        diagnostic for diagnostic in prepared.diagnostics if diagnostic.kind == "fact"
    )


def _facts_resolved(prepared: PreparedPrompt) -> bool:
    facts = _fact_diagnostics(prepared)
    return bool(facts) and all(
        diagnostic.status is SpanStatus.OK
        and diagnostic.char_start is not None
        and diagnostic.char_end is not None
        and diagnostic.token_start is not None
        and diagnostic.token_end is not None
        for diagnostic in facts
    )


def _overlaps(offset: tuple[int, int], span: CharSpan) -> bool:
    start, end = offset
    return end > start and start < span.end and end > span.start


def build_padding_positions(prepared: PreparedPrompt) -> tuple[int, ...]:
    """Return tokens overlapping non-key paragraph payloads in ``mixin``."""
    if not _facts_resolved(prepared):
        return ()
    key_payloads = {
        (diagnostic.char_start, diagnostic.char_end)
        for diagnostic in _fact_diagnostics(prepared)
    }
    padding_payloads = tuple(
        span
        for span in prepared.paragraph_payload_spans
        if (span.start, span.end) not in key_payloads
    )
    return tuple(
        token_position
        for token_position, offset in enumerate(prepared.offsets)
        if any(_overlaps(offset, payload) for payload in padding_payloads)
    )


def _placement_label(before: float, between: float, after: float) -> str:
    if before <= PLACEMENT_EPSILON:
        return "first"
    if after <= PLACEMENT_EPSILON:
        return "last"
    if abs(before - after) <= PLACEMENT_EPSILON and between <= PLACEMENT_EPSILON:
        return "middle"
    return "scattered"


def derive_conditions(prepared: PreparedPrompt) -> PromptConditions:
    """Derive effective conditions from token positions and unanimous provenance."""
    if not _facts_resolved(prepared):
        return PromptConditions(
            padding_type_effective=None,
            dispersion_effective="unresolved",
            frac_padding_before=0.0,
            frac_padding_between=0.0,
            frac_padding_after=0.0,
            n_padding_tokens=0,
        )

    padding = build_padding_positions(prepared)
    if not padding:
        return PromptConditions(
            padding_type_effective="none",
            dispersion_effective="not_applicable",
            frac_padding_before=0.0,
            frac_padding_between=0.0,
            frac_padding_after=0.0,
            n_padding_tokens=0,
        )

    declared = prepared.prompt.padding_type_declared
    if len(declared) != 1 or declared[0] not in {"books", "same"}:
        raise ValueError(
            "Padded prompt source rows must agree on one declared "
            f"padding type; received {declared!r}"
        )

    facts = prepared.fact_token_spans
    first_start = min(span.start for span in facts)
    last_end = max(span.end for span in facts)
    total = len(padding)
    before_count = sum(position < first_start for position in padding)
    after_count = sum(position >= last_end for position in padding)
    between_count = total - before_count - after_count
    before = before_count / total
    between = between_count / total
    after = after_count / total
    return PromptConditions(
        padding_type_effective=declared[0],
        dispersion_effective=_placement_label(before, between, after),
        frac_padding_before=before,
        frac_padding_between=between,
        frac_padding_after=after,
        n_padding_tokens=total,
    )


def assert_unpadded_prompt_count(
    prepared_prompts: Sequence[PreparedPrompt],
    *,
    expected: int = 300,
) -> None:
    """Assert the content-verified unpadded baseline has exactly ``expected`` rows."""
    count = sum(
        _facts_resolved(prepared) and not build_padding_positions(prepared)
        for prepared in prepared_prompts
    )
    if count != expected:
        raise ValueError(
            f"Expected exactly {expected} content-verified unpadded prompts; "
            f"found {count}"
        )
