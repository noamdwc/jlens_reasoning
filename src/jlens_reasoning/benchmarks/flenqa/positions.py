"""Build token and position assets for FLenQA prompts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaPrompt
from jlens_reasoning.benchmarks.flenqa.positions_utils import (
    find_required_span,
    non_fact_paragraphs,
    padding_token_positions,
    resolve_span,
    sample_padding_positions,
)
from jlens_reasoning.benchmarks.flenqa.tokenization import tokenize_with_offsets
from jlens_reasoning.experiments_utils.spans import (
    CharSpan,
    parse_paragraph_payload_spans,
)

PIR_TASK = "PIR"
MONOREL_TASK = "MonoRel"
RULETAKER_TASK = "Simplified RuleTaker"

FACT_LABELS = ("fact_a_end", "fact_b_end")
QUESTION_LABEL = "question_end"
FINAL_LABEL = "final_prompt"
PADDING_LABEL = "sampled_padding"

PADDING_SAMPLE_COUNT = 4


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """Token and labeled-position assets consumed by benchmark runners."""

    prompt: FlenqaPrompt
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    positions: dict[str, tuple[int, ...]]

    @property
    def unique_positions(self) -> tuple[int, ...]:
        """Return sorted token positions without duplicate labels."""
        return tuple(
            sorted(
                {
                    position
                    for labeled_positions in self.positions.values()
                    for position in labeled_positions
                }
            )
        )


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


def _exact_facts(
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
    PIR_TASK: _exact_facts,
    MONOREL_TASK: _exact_facts,
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


def prepare_prompt(
    prompt: FlenqaPrompt,
    tokenizer: Any,
    max_seq_len: int = 4096,
    sample_seed: int = 1729,
) -> PreparedPrompt:
    """Build reusable token and labeled-position assets for one FLenQA prompt.

    The prompt is tokenized once without truncation. Its fact paragraphs and
    final question occurrence are resolved against that tokenization, then
    grouped by label alongside the final token and deterministic samples from
    non-fact context paragraphs. Character offsets are retained for debugging.

    Args:
        prompt: Normalized FLenQA prompt containing the final rendered text and
            the task metadata needed to locate facts and the question.
        tokenizer: Callable tokenizer that accepts the Hugging Face-style
            arguments used by ``tokenize_with_offsets`` and returns one input-ID
            sequence with character offset mappings.
        max_seq_len: Maximum accepted token count. The prompt is rejected rather
            than truncated when it exceeds this value.
        sample_seed: Integer combined with ``prompt.prompt_id`` to choose stable
            padding positions.

    Returns:
        A ``PreparedPrompt`` containing the source prompt, input IDs, debugging
        offsets, and a mapping from position labels to token-index tuples.

    Raises:
        ValueError: If the token limit is invalid or exceeded, tokenizer output
            is malformed, the task is unknown, or a required prompt span cannot
            be resolved unambiguously.
    """
    # Tokenize the exact author prompt once. Tokenizer-shape validation lives in
    # tokenization.py so this function can focus on building reusable assets.
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
    context_chars = find_required_span(prompt.text, prompt.mixin, name="context")
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
        resolve_span(
            label,
            CharSpan(
                context_chars.start + local_span.start,
                context_chars.start + local_span.end,
            ),
            tokenized.offsets,
        )
        for label, _, local_span in fact_matches
    )

    # The author template repeats MonoRel questions, so the final occurrence is
    # the question asset used by downstream analyses.
    question_chars = find_required_span(
        prompt.text,
        prompt.question,
        name="question",
        choose_last=True,
    )
    question = resolve_span(
        QUESTION_LABEL,
        question_chars,
        tokenized.offsets,
    )
    positions = {
        label: tuple(fact.token_span.end - 1 for fact in facts if fact.label == label)
        for label, _ in zip(FACT_LABELS, prompt.key_texts, strict=False)
    }
    positions[QUESTION_LABEL] = (question.token_span.end - 1,)
    positions[FINAL_LABEL] = (len(tokenized.input_ids) - 1,)
    padding_candidates = padding_token_positions(
        input_ids=tokenized.input_ids,
        offsets=tokenized.offsets,
        text=prompt.text,
        padding_spans=non_fact_paragraphs(paragraphs, facts),
        special_ids=tokenized.special_token_ids,
    )
    positions[PADDING_LABEL] = sample_padding_positions(
        padding_candidates,
        prompt_id=prompt.prompt_id,
        sample_seed=sample_seed,
        count=PADDING_SAMPLE_COUNT,
    )

    return PreparedPrompt(
        prompt=prompt,
        input_ids=tokenized.input_ids,
        offsets=tokenized.offsets,
        positions=positions,
    )
