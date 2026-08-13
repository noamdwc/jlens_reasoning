"""Readable FLenQA span resolution and experimental position selection."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaPrompt
from jlens_reasoning.benchmarks.flenqa.tokenization import tokenize_with_offsets
from jlens_reasoning.experiments_utils.spans import (
    CharSpan,
    char_span_to_token_span,
    find_all_spans,
    parse_paragraph_payload_spans,
)

PIR_TASK = "PIR"
MONOREL_TASK = "MonoRel"
RULETAKER_TASK = "Simplified RuleTaker"

PADDING_SAMPLE_COUNT = 4
FACT_LABELS = ("fact_a_end", "fact_b_end")
QUESTION_LABEL = "question_end"
FINAL_LABEL = "final_prompt"
PADDING_LABEL = "sampled_padding"


@dataclass(frozen=True, slots=True)
class ResolvedSpan:
    """One named surface resolved in both character and token coordinates."""

    label: str
    surface: str
    char_span: CharSpan
    token_span: CharSpan


@dataclass(frozen=True, slots=True)
class LabeledPosition:
    label: str
    position: int


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """A fully positioned prompt ready for both lens passes."""

    prompt: FlenqaPrompt
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    token_signature: str
    context: ResolvedSpan
    paragraph_payload_spans: tuple[CharSpan, ...]
    facts: tuple[ResolvedSpan, ...]
    question: ResolvedSpan
    rule: ResolvedSpan | None
    positions: tuple[LabeledPosition, ...]
    special_token_ids: frozenset[int]

    @property
    def unique_positions(self) -> tuple[int, ...]:
        return unique_positions(self.positions)


def unique_positions(positions: Sequence[LabeledPosition]) -> tuple[int, ...]:
    return tuple(sorted({item.position for item in positions}))


FactMatch = tuple[str, str, CharSpan]


def _match_facts(
    prompt: FlenqaPrompt,
    paragraphs: Sequence[CharSpan],
    *,
    contains: bool,
) -> tuple[FactMatch, ...]:
    if len(prompt.key_texts) > len(FACT_LABELS):
        raise ValueError("FLenQA positions support at most two key facts")

    matches: list[FactMatch] = []
    for label, surface in zip(FACT_LABELS, prompt.key_texts, strict=False):
        fact_paragraphs = tuple(
            paragraph
            for paragraph in paragraphs
            if (
                surface in prompt.mixin[paragraph.start : paragraph.end]
                if contains
                else surface == prompt.mixin[paragraph.start : paragraph.end]
            )
        )
        if not fact_paragraphs:
            raise ValueError(f"Required FLenQA fact span is unresolved: {surface!r}")
        matches.extend((label, surface, paragraph) for paragraph in fact_paragraphs)
    return tuple(matches)


def _pir_facts(
    prompt: FlenqaPrompt,
    paragraphs: Sequence[CharSpan],
) -> tuple[FactMatch, ...]:
    return _match_facts(prompt, paragraphs, contains=False)


def _monorel_facts(
    prompt: FlenqaPrompt,
    paragraphs: Sequence[CharSpan],
) -> tuple[FactMatch, ...]:
    return _match_facts(prompt, paragraphs, contains=False)


def _ruletaker_facts(
    prompt: FlenqaPrompt,
    paragraphs: Sequence[CharSpan],
) -> tuple[FactMatch, ...]:
    return _match_facts(prompt, paragraphs, contains=True)


_FACT_RESOLVERS: dict[
    str,
    Callable[[FlenqaPrompt, Sequence[CharSpan]], tuple[FactMatch, ...]],
] = {
    PIR_TASK: _pir_facts,
    MONOREL_TASK: _monorel_facts,
    RULETAKER_TASK: _ruletaker_facts,
}


def resolve_key_paragraphs(prompt: FlenqaPrompt) -> tuple[FactMatch, ...]:
    """Resolve each logical fact to its task-specific context paragraphs."""
    paragraphs = parse_paragraph_payload_spans(prompt.mixin)
    try:
        resolver = _FACT_RESOLVERS[prompt.task]
    except KeyError as exc:
        raise ValueError(f"Unknown FLenQA task: {prompt.task!r}") from exc
    return resolver(prompt, paragraphs)


def _find_span(
    text: str,
    surface: str,
    *,
    name: str,
    choose_last: bool = False,
) -> CharSpan:
    matches = find_all_spans(text, surface) if surface else ()
    if not matches:
        raise ValueError(f"Required FLenQA {name} span is unresolved")
    if len(matches) > 1 and not choose_last:
        raise ValueError(f"Required FLenQA {name} span is ambiguous")
    return matches[-1] if choose_last else matches[0]


def _resolved_span(
    label: str,
    surface: str,
    char_span: CharSpan,
    offsets: Sequence[tuple[int, int]],
) -> ResolvedSpan:
    return ResolvedSpan(
        label=label,
        surface=surface,
        char_span=char_span,
        token_span=char_span_to_token_span(offsets, char_span),
    )


def _padding_spans(
    paragraphs: Sequence[CharSpan],
    facts: Sequence[ResolvedSpan],
) -> tuple[CharSpan, ...]:
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
    start, end = offset
    return (
        token_id not in special_ids
        and end > start
        and bool(text[start:end].strip())
        and any(start < span.end and end > span.start for span in padding_spans)
    )


def _padding_content_positions(
    *,
    input_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    text: str,
    padding_spans: Sequence[CharSpan],
    special_ids: frozenset[int],
) -> tuple[int, ...]:
    return tuple(
        position
        for position, (token_id, offset) in enumerate(
            zip(input_ids, offsets, strict=True)
        )
        if _eligible_padding_position(
            token_id=token_id,
            offset=offset,
            text=text,
            padding_spans=padding_spans,
            special_ids=special_ids,
        )
    )


def padding_content_positions(prepared: PreparedPrompt) -> tuple[int, ...]:
    """Return tokens that overlap non-fact context paragraphs."""
    return _padding_content_positions(
        input_ids=prepared.input_ids,
        offsets=prepared.offsets,
        text=prepared.prompt.text,
        padding_spans=_padding_spans(prepared.paragraph_payload_spans, prepared.facts),
        special_ids=prepared.special_token_ids,
    )


def sample_padding_positions(
    positions: Sequence[int],
    *,
    prompt_id: str,
    sample_seed: int,
    count: int = PADDING_SAMPLE_COUNT,
) -> tuple[int, ...]:
    """Sample sorted padding positions from a prompt-stable random seed."""
    unique = sorted(set(positions))
    if type(count) is not int or count < 0:
        raise ValueError("padding sample count must be a non-negative integer")
    if type(sample_seed) is not int:
        raise ValueError("padding sample seed must be an integer")
    if not unique or count == 0:
        return ()
    try:
        prompt_seed = int(prompt_id[:16], 16)
    except ValueError as exc:
        raise ValueError("prompt_id must begin with hexadecimal characters") from exc
    rng = random.Random(prompt_seed ^ sample_seed)
    return tuple(sorted(rng.sample(unique, min(count, len(unique)))))


def prepare_prompt(
    prompt: FlenqaPrompt,
    tokenizer: Any,
    max_seq_len: int = 4096,
    sample_seed: int = 1729,
) -> PreparedPrompt:
    """Resolve one prompt and select every position used by the experiment."""
    # Tokenize the exact author prompt once. Tokenizer-shape validation lives in
    # tokenization.py so the rest of this function can describe the experiment.
    tokenized = tokenize_with_offsets(prompt.text, tokenizer)
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if len(tokenized.input_ids) > max_seq_len:
        raise ValueError(
            f"untruncated prompt has {len(tokenized.input_ids)} tokens, "
            f"exceeding {max_seq_len}"
        )

    # Resolve the context once, then map its newline-delimited paragraphs into
    # final-prompt coordinates. Task-specific fact rules are explicit above.
    context_chars = _find_span(prompt.text, prompt.mixin, name="context")
    context = _resolved_span(
        "context",
        prompt.mixin,
        context_chars,
        tokenized.offsets,
    )
    local_paragraphs = parse_paragraph_payload_spans(prompt.mixin)
    paragraphs = tuple(
        CharSpan(
            context_chars.start + paragraph.start,
            context_chars.start + paragraph.end,
        )
        for paragraph in local_paragraphs
    )

    fact_matches = resolve_key_paragraphs(prompt)
    facts = tuple(
        _resolved_span(
            label,
            surface,
            CharSpan(
                context_chars.start + local_span.start,
                context_chars.start + local_span.end,
            ),
            tokenized.offsets,
        )
        for label, surface, local_span in fact_matches
    )

    # The author template repeats MonoRel questions, so the final occurrence is
    # the experimental question position. Rules are recorded but not selected.
    question_chars = _find_span(
        prompt.text,
        prompt.question,
        name="question",
        choose_last=True,
    )
    question = _resolved_span(
        QUESTION_LABEL,
        prompt.question,
        question_chars,
        tokenized.offsets,
    )
    rule = None
    if prompt.rule is not None:
        rule_prefix = "Rule: "
        declaration = _find_span(
            prompt.text,
            f"{rule_prefix}{prompt.rule}",
            name="rule declaration",
        )
        rule_chars = CharSpan(declaration.start + len(rule_prefix), declaration.end)
        rule = _resolved_span("rule", prompt.rule, rule_chars, tokenized.offsets)

    # Position selection is intentionally direct: semantic span ends, the final
    # model position, and four deterministic samples from non-fact paragraphs.
    positions = [
        LabeledPosition(span.label, span.token_span.end - 1)
        for span in (*facts, question)
    ]
    positions.append(LabeledPosition(FINAL_LABEL, len(tokenized.input_ids) - 1))
    padding_candidates = _padding_content_positions(
        input_ids=tokenized.input_ids,
        offsets=tokenized.offsets,
        text=prompt.text,
        padding_spans=_padding_spans(paragraphs, facts),
        special_ids=tokenized.special_token_ids,
    )
    positions.extend(
        LabeledPosition(PADDING_LABEL, position)
        for position in sample_padding_positions(
            padding_candidates,
            prompt_id=prompt.prompt_id,
            sample_seed=sample_seed,
        )
    )

    return PreparedPrompt(
        prompt=prompt,
        input_ids=tokenized.input_ids,
        offsets=tokenized.offsets,
        token_signature=tokenized.signature,
        context=context,
        paragraph_payload_spans=paragraphs,
        facts=facts,
        question=question,
        rule=rule,
        positions=tuple(positions),
        special_token_ids=tokenized.special_token_ids,
    )
