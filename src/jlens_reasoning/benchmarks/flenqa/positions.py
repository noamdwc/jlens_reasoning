"""Task-aware FLenQA span resolution over one full-prompt tokenization."""

from __future__ import annotations

import hashlib
import operator
import random
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from jlens_reasoning.benchmarks.flenqa.dataset import FlenqaPrompt
from jlens_reasoning.experiments_utils.spans import (
    CharSpan,
    SpanDiagnostic,
    SpanStatus,
    char_span_to_token_span,
    find_all_spans,
    parse_paragraph_payload_spans,
)

PIR_TASK = "PIR"
MONOREL_TASK = "MonoRel"
RULETAKER_TASK = "Simplified RuleTaker"
PADDING_SAMPLE_COUNT = 4
FACT_LABELS = ("fact_a_end", "fact_b_end")
BRIDGE_LABELS = ("bridge_fact_a", "bridge_fact_b")
QUESTION_LABEL = "question_end"
FINAL_LABEL = "final_prompt"
PADDING_LABEL = "sampled_padding"

_PERSON = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
_POSSESSIVE_PHRASE = re.compile(
    r"\b[A-Z][A-Za-z'-]*'s(?:\s+[a-z][A-Za-z'-]*)+?"
    r"(?=\s+(?:is|was|has|contains|appears|looks)\b|[,.;])"
)


@dataclass(frozen=True, slots=True)
class LabeledPosition:
    """One semantic label attached to a token position."""

    label: str
    position: int


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    """An immutable prompt with full-text tokenization and resolved spans.

    ``paragraph_payload_spans`` uses final-prompt character coordinates. It is
    empty when the context itself cannot be mapped uniquely into the prompt.
    """

    prompt: FlenqaPrompt
    input_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    token_signature: str
    context_char_span: CharSpan | None
    context_token_span: CharSpan | None
    paragraph_payload_spans: tuple[CharSpan, ...]
    diagnostics: tuple[SpanDiagnostic, ...]
    bridge: str | None
    positions: tuple[LabeledPosition, ...]
    special_token_ids: frozenset[int]

    @property
    def fact_token_spans(self) -> tuple[CharSpan, ...]:
        return tuple(
            CharSpan(diagnostic.token_start, diagnostic.token_end)
            for diagnostic in self.diagnostics
            if diagnostic.kind == "fact"
            and diagnostic.status is SpanStatus.OK
            and diagnostic.token_start is not None
            and diagnostic.token_end is not None
        )

    @property
    def question_token_span(self) -> CharSpan | None:
        for diagnostic in self.diagnostics:
            if (
                diagnostic.kind == "question"
                and diagnostic.status is SpanStatus.OK
                and diagnostic.token_start is not None
                and diagnostic.token_end is not None
            ):
                return CharSpan(diagnostic.token_start, diagnostic.token_end)
        return None

    @property
    def unique_positions(self) -> tuple[int, ...]:
        return unique_positions(self.positions)


def unique_positions(
    positions: Sequence[LabeledPosition],
) -> tuple[int, ...]:
    """Return sorted positions while preserving labels in the source sequence."""
    return tuple(sorted({item.position for item in positions}))


def _shared_candidates(
    prompt: FlenqaPrompt,
    pattern: re.Pattern[str],
) -> tuple[str, ...]:
    if not prompt.key_texts:
        return ()
    per_fact = [set(pattern.findall(fact)) for fact in prompt.key_texts]
    shared = set.intersection(*per_fact)
    return tuple(
        sorted(
            (
                candidate
                for candidate in shared
                if candidate.casefold() not in prompt.question.casefold()
            ),
            key=lambda candidate: (-len(candidate), candidate),
        )
    )


def extract_bridge(prompt: FlenqaPrompt) -> str | None:
    """Return the unique task-specific bridge, or ``None`` when unresolved."""
    if prompt.task == RULETAKER_TASK:
        return None
    if prompt.task == PIR_TASK:
        candidates = _shared_candidates(prompt, _POSSESSIVE_PHRASE)
    elif prompt.task == MONOREL_TASK:
        candidates = _shared_candidates(prompt, _PERSON)
    else:
        raise ValueError(f"Unknown FLenQA task: {prompt.task!r}")
    return candidates[0] if len(candidates) == 1 else None


@dataclass(frozen=True, slots=True)
class BridgeGateResult:
    applicable: int
    resolved: int


def bridge_gate(
    prompts: Sequence[FlenqaPrompt],
    *,
    expected_applicable: int,
) -> BridgeGateResult:
    """Require one consistent, non-leaking bridge per applicable problem."""
    by_problem: dict[int, list[FlenqaPrompt]] = {}
    for prompt in prompts:
        if prompt.task in {PIR_TASK, MONOREL_TASK}:
            by_problem.setdefault(prompt.problem_id, []).append(prompt)
    applicable = len(by_problem)
    if applicable != expected_applicable:
        raise ValueError(
            f"Bridge gate expected {expected_applicable} applicable problems; "
            f"found {applicable}"
        )

    for problem_id, problem_prompts in by_problem.items():
        bridges = {extract_bridge(prompt) for prompt in problem_prompts}
        if None in bridges or len(bridges) != 1:
            raise ValueError(
                f"Bridge unresolved or inconsistent for problem {problem_id}"
            )
        bridge = next(iter(bridges))
        assert bridge is not None
        if any(
            bridge.casefold() in prompt.question.casefold()
            for prompt in problem_prompts
        ):
            raise ValueError(f"Bridge leaks into question for problem {problem_id}")
    return BridgeGateResult(applicable=applicable, resolved=applicable)


def _diagnostic(
    *,
    kind: str,
    ordinal: int,
    surface: str,
    status: SpanStatus,
    match_count: int,
    span: CharSpan | None = None,
    fact_ordinal: int | None = None,
) -> SpanDiagnostic:
    return SpanDiagnostic(
        kind=kind,  # type: ignore[arg-type]
        ordinal=ordinal,
        fact_ordinal=fact_ordinal,
        surface=surface,
        status=status,
        match_count=match_count,
        char_start=None if span is None else span.start,
        char_end=None if span is None else span.end,
        token_start=None,
        token_end=None,
    )


def _unique_status(match_count: int) -> SpanStatus:
    if match_count == 1:
        return SpanStatus.OK
    if match_count:
        return SpanStatus.AMBIGUOUS
    return SpanStatus.UNRESOLVED


def resolve_key_paragraphs(prompt: FlenqaPrompt) -> tuple[SpanDiagnostic, ...]:
    """Resolve logical fact targets to mixin-relative paragraph bounds."""
    payload_spans = parse_paragraph_payload_spans(prompt.mixin)
    diagnostics: list[SpanDiagnostic] = []
    for ordinal, surface in enumerate(prompt.key_texts):
        if not surface:
            matches: tuple[CharSpan, ...] = ()
        elif prompt.task == "Simplified RuleTaker":
            matches = tuple(
                payload_span
                for payload_span in payload_spans
                if surface in prompt.mixin[payload_span.start : payload_span.end]
            )
        else:
            matches = tuple(
                payload_span
                for payload_span in payload_spans
                if prompt.mixin[payload_span.start : payload_span.end] == surface
            )
        status = _unique_status(len(matches))
        diagnostics.append(
            _diagnostic(
                kind="fact",
                ordinal=ordinal,
                surface=surface,
                status=status,
                match_count=len(matches),
                span=matches[0] if status is SpanStatus.OK else None,
            )
        )
    return tuple(diagnostics)


def _to_builtin(value: Any) -> Any:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _as_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    raise ValueError(f"tokenizer {name} must be a sequence")


def _normalize_input_ids(value: Any) -> tuple[int, ...]:
    raw = _as_sequence(_to_builtin(value), name="input_ids")
    if raw and isinstance(raw[0], Sequence) and not isinstance(raw[0], (str, bytes)):
        if len(raw) != 1:
            raise ValueError("tokenizer input_ids must contain exactly one batch")
        raw = _as_sequence(raw[0], name="input_ids batch")
    ids: list[int] = []
    for value in raw:
        try:
            token_id = operator.index(value)
        except TypeError as exc:
            raise ValueError("tokenizer input_ids must contain integers") from exc
        ids.append(token_id)
    if not ids:
        raise ValueError("tokenizer input_ids must be nonempty")
    return tuple(ids)


def _looks_like_offset(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and not isinstance(value[0], Sequence)
    )


def _normalize_offsets(value: Any) -> tuple[tuple[int, int], ...]:
    raw = _as_sequence(_to_builtin(value), name="offset_mapping")
    if raw and not _looks_like_offset(raw[0]):
        if len(raw) != 1:
            raise ValueError("tokenizer offset_mapping must contain exactly one batch")
        raw = _as_sequence(raw[0], name="offset_mapping batch")
    offsets: list[tuple[int, int]] = []
    for raw_offset in raw:
        if not _looks_like_offset(raw_offset):
            raise ValueError("tokenizer offsets must be integer pairs")
        try:
            start = operator.index(raw_offset[0])
            end = operator.index(raw_offset[1])
        except TypeError as exc:
            raise ValueError("tokenizer offsets must be integer pairs") from exc
        if start < 0 or end < start:
            raise ValueError("tokenizer offsets must be non-negative and ordered")
        offsets.append((start, end))
    return tuple(offsets)


def _tokenize_once(
    prompt: FlenqaPrompt,
    tokenizer: Any,
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    encoded = tokenizer(
        prompt.text,
        return_tensors="pt",
        truncation=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoded, Mapping) and not hasattr(encoded, "__getitem__"):
        raise ValueError("tokenizer output must provide input_ids and offset_mapping")
    try:
        raw_ids = encoded["input_ids"]
        raw_offsets = encoded["offset_mapping"]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            "tokenizer output must provide input_ids and offset_mapping"
        ) from exc
    input_ids = _normalize_input_ids(raw_ids)
    offsets = _normalize_offsets(raw_offsets)
    if len(offsets) != len(input_ids):
        raise ValueError(
            "tokenizer input_ids and offset_mapping must have the same length"
        )
    previous = (0, 0)
    for offset in offsets:
        start, end = offset
        if end > len(prompt.text):
            raise ValueError("tokenizer offset extends beyond the prompt text")
        if end > start:
            if start < previous[0] or end < previous[1]:
                raise ValueError("tokenizer nonzero offsets must be monotonic")
            previous = offset
    return input_ids, offsets


def _token_signature(input_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    try:
        for token_id in input_ids:
            digest.update(struct.pack(">q", token_id))
    except struct.error as exc:
        raise ValueError("tokenizer token ID is outside signed 64-bit range") from exc
    return digest.hexdigest()


def _add_token_bounds(
    diagnostic: SpanDiagnostic,
    offsets: tuple[tuple[int, int], ...],
) -> SpanDiagnostic:
    if (
        diagnostic.status is not SpanStatus.OK
        or diagnostic.char_start is None
        or diagnostic.char_end is None
    ):
        return diagnostic
    token_span = char_span_to_token_span(
        offsets,
        CharSpan(diagnostic.char_start, diagnostic.char_end),
    )
    return replace(
        diagnostic,
        token_start=token_span.start,
        token_end=token_span.end,
    )


def _context_diagnostic(prompt: FlenqaPrompt) -> SpanDiagnostic:
    matches = find_all_spans(prompt.text, prompt.mixin) if prompt.mixin else ()
    status = _unique_status(len(matches))
    return _diagnostic(
        kind="context",
        ordinal=0,
        surface=prompt.mixin,
        status=status,
        match_count=len(matches),
        span=matches[0] if status is SpanStatus.OK else None,
    )


def _map_mixin_diagnostic_to_prompt(
    local: SpanDiagnostic,
    context: SpanDiagnostic,
) -> SpanDiagnostic:
    if local.status is not SpanStatus.OK:
        return local
    if context.status is not SpanStatus.OK:
        return replace(
            local,
            status=context.status,
            char_start=None,
            char_end=None,
        )
    assert context.char_start is not None
    assert local.char_start is not None
    assert local.char_end is not None
    return replace(
        local,
        char_start=context.char_start + local.char_start,
        char_end=context.char_start + local.char_end,
    )


def _last_occurrence_diagnostic(
    *,
    kind: str,
    surface: str,
    text: str,
) -> SpanDiagnostic:
    matches = find_all_spans(text, surface) if surface else ()
    return _diagnostic(
        kind=kind,
        ordinal=0,
        surface=surface,
        status=SpanStatus.OK if matches else SpanStatus.UNRESOLVED,
        match_count=len(matches),
        span=matches[-1] if matches else None,
    )


def _rule_diagnostic(prompt: FlenqaPrompt) -> SpanDiagnostic | None:
    if prompt.rule is None:
        return None
    matches = find_all_spans(prompt.text, prompt.rule) if prompt.rule else ()
    status = _unique_status(len(matches))
    return _diagnostic(
        kind="rule",
        ordinal=0,
        surface=prompt.rule,
        status=status,
        match_count=len(matches),
        span=matches[0] if status is SpanStatus.OK else None,
    )


def _bridge_diagnostic(
    *,
    bridge: str,
    fact: SpanDiagnostic,
    fact_ordinal: int,
    text: str,
) -> SpanDiagnostic:
    if (
        fact.status is not SpanStatus.OK
        or fact.char_start is None
        or fact.char_end is None
    ):
        return _diagnostic(
            kind="bridge",
            ordinal=fact_ordinal,
            fact_ordinal=fact_ordinal,
            surface=bridge,
            status=fact.status,
            match_count=0,
        )
    fact_text = text[fact.char_start : fact.char_end]
    matches = find_all_spans(fact_text, bridge)
    if not matches:
        return _diagnostic(
            kind="bridge",
            ordinal=fact_ordinal,
            fact_ordinal=fact_ordinal,
            surface=bridge,
            status=SpanStatus.UNRESOLVED,
            match_count=0,
        )
    chosen = matches[-1]
    return _diagnostic(
        kind="bridge",
        ordinal=fact_ordinal,
        fact_ordinal=fact_ordinal,
        surface=bridge,
        status=SpanStatus.OK,
        match_count=len(matches),
        span=CharSpan(
            fact.char_start + chosen.start,
            fact.char_start + chosen.end,
        ),
    )


def _fact_diagnostics(prepared: PreparedPrompt) -> tuple[SpanDiagnostic, ...]:
    return tuple(
        diagnostic for diagnostic in prepared.diagnostics if diagnostic.kind == "fact"
    )


def _padding_content_spans(prepared: PreparedPrompt) -> tuple[CharSpan, ...]:
    key_spans = {
        (diagnostic.char_start, diagnostic.char_end)
        for diagnostic in _fact_diagnostics(prepared)
        if diagnostic.status is SpanStatus.OK
    }
    return tuple(
        span
        for span in prepared.paragraph_payload_spans
        if (span.start, span.end) not in key_spans
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
    if token_id in special_ids or end <= start:
        return False
    if not text[start:end].strip():
        return False
    return any(start < span.end and end > span.start for span in padding_spans)


def padding_content_positions(prepared: PreparedPrompt) -> tuple[int, ...]:
    """Return eligible tokens overlapping declared padding-content spans."""
    padding_spans = _padding_content_spans(prepared)
    return tuple(
        position
        for position, (token_id, offset) in enumerate(
            zip(prepared.input_ids, prepared.offsets, strict=True)
        )
        if _eligible_padding_position(
            token_id=token_id,
            offset=offset,
            text=prepared.prompt.text,
            padding_spans=padding_spans,
            special_ids=prepared.special_token_ids,
        )
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


def _select_positions(
    prepared: PreparedPrompt,
    *,
    sample_seed: int,
) -> tuple[LabeledPosition, ...]:
    selected: list[LabeledPosition] = []
    for label, diagnostic in zip(
        FACT_LABELS,
        _fact_diagnostics(prepared),
        strict=False,
    ):
        if diagnostic.status is SpanStatus.OK and diagnostic.token_end is not None:
            selected.append(LabeledPosition(label, diagnostic.token_end - 1))
    bridge_diagnostics = tuple(
        diagnostic
        for diagnostic in prepared.diagnostics
        if diagnostic.kind == "bridge"
    )
    for label, diagnostic in zip(
        BRIDGE_LABELS,
        bridge_diagnostics,
        strict=False,
    ):
        if diagnostic.status is SpanStatus.OK and diagnostic.token_end is not None:
            selected.append(LabeledPosition(label, diagnostic.token_end - 1))
    question = prepared.question_token_span
    if question is not None:
        selected.append(LabeledPosition(QUESTION_LABEL, question.end - 1))
    selected.append(LabeledPosition(FINAL_LABEL, len(prepared.input_ids) - 1))
    selected.extend(
        LabeledPosition(PADDING_LABEL, position)
        for position in sample_padding_positions(
            padding_content_positions(prepared),
            prompt_id=prepared.prompt.prompt_id,
            sample_seed=sample_seed,
        )
    )
    if any(
        item.position < 0 or item.position >= len(prepared.input_ids)
        for item in selected
    ):
        raise ValueError("Semantic position is outside the prepared prompt")
    return tuple(selected)


def validate_prepared_prompt(prepared: PreparedPrompt) -> PreparedPrompt:
    """Require every semantic input needed by the benchmark positions."""
    facts = _fact_diagnostics(prepared)
    if len(facts) != len(prepared.prompt.key_texts) or any(
        diagnostic.status is not SpanStatus.OK
        or diagnostic.token_start is None
        or diagnostic.token_end is None
        for diagnostic in facts
    ):
        raise ValueError("Required FLenQA fact spans are unresolved")
    questions = tuple(
        diagnostic
        for diagnostic in prepared.diagnostics
        if diagnostic.kind == "question"
    )
    if len(questions) != 1 or any(
        diagnostic.status is not SpanStatus.OK
        or diagnostic.token_start is None
        or diagnostic.token_end is None
        for diagnostic in questions
    ):
        raise ValueError("Required FLenQA question span is unresolved")
    if prepared.prompt.task in {PIR_TASK, MONOREL_TASK}:
        bridges = tuple(
            diagnostic
            for diagnostic in prepared.diagnostics
            if diagnostic.kind == "bridge"
        )
        if prepared.bridge is None or len(bridges) != len(facts) or any(
            diagnostic.status is not SpanStatus.OK
            or diagnostic.token_start is None
            or diagnostic.token_end is None
            for diagnostic in bridges
        ):
            raise ValueError("Required FLenQA bridge spans are unresolved")
    expected_labels = {
        *FACT_LABELS[: len(facts)],
        QUESTION_LABEL,
        FINAL_LABEL,
    }
    if prepared.prompt.task in {PIR_TASK, MONOREL_TASK}:
        expected_labels.update(BRIDGE_LABELS[: len(facts)])
    actual_labels = {item.label for item in prepared.positions}
    if not expected_labels <= actual_labels:
        missing = sorted(expected_labels - actual_labels)
        raise ValueError(f"Required FLenQA position labels are missing: {missing}")
    if prepared.unique_positions != tuple(sorted(set(prepared.unique_positions))):
        raise ValueError("FLenQA execution positions must be sorted and unique")
    return prepared


def prepare_prompt(
    prompt: FlenqaPrompt,
    tokenizer: Any,
    max_seq_len: int = 4096,
    bridge: str | None = None,
    sample_seed: int = 1729,
) -> PreparedPrompt:
    """Prepare one final author prompt without truncation or re-tokenization."""
    input_ids, offsets = _tokenize_once(prompt, tokenizer)
    if type(max_seq_len) is not int or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive integer")
    if len(input_ids) > max_seq_len:
        raise ValueError(
            f"untruncated prompt has {len(input_ids)} tokens, exceeding {max_seq_len}"
        )
    if bridge == "":
        raise ValueError("bridge must be nonempty when provided")
    if bridge is None:
        bridge = extract_bridge(prompt)

    context = _add_token_bounds(_context_diagnostic(prompt), offsets)
    local_facts = resolve_key_paragraphs(prompt)
    facts = tuple(
        _add_token_bounds(_map_mixin_diagnostic_to_prompt(local, context), offsets)
        for local in local_facts
    )

    diagnostics: list[SpanDiagnostic] = [context]
    diagnostics.extend(facts)
    if bridge is not None:
        diagnostics.extend(
            _add_token_bounds(
                _map_mixin_diagnostic_to_prompt(
                    _bridge_diagnostic(
                        bridge=bridge,
                        fact=local_fact,
                        fact_ordinal=fact_ordinal,
                        text=prompt.mixin,
                    ),
                    context,
                ),
                offsets,
            )
            for fact_ordinal, local_fact in enumerate(local_facts)
        )
    rule = _rule_diagnostic(prompt)
    if rule is not None:
        diagnostics.append(_add_token_bounds(rule, offsets))
    diagnostics.append(
        _add_token_bounds(
            _last_occurrence_diagnostic(
                kind="question",
                surface=prompt.question,
                text=prompt.text,
            ),
            offsets,
        )
    )

    if context.status is SpanStatus.OK:
        assert context.char_start is not None
        paragraph_payload_spans = tuple(
            CharSpan(
                context.char_start + span.start,
                context.char_start + span.end,
            )
            for span in parse_paragraph_payload_spans(prompt.mixin)
        )
        context_char_span = CharSpan(context.char_start, context.char_end)  # type: ignore[arg-type]
        context_token_span = CharSpan(context.token_start, context.token_end)  # type: ignore[arg-type]
    else:
        paragraph_payload_spans = ()
        context_char_span = None
        context_token_span = None

    raw_special_ids = getattr(tokenizer, "all_special_ids", ())
    special_token_ids = frozenset(
        int(token_id) for token_id in (() if raw_special_ids is None else raw_special_ids)
    )
    prepared = PreparedPrompt(
        prompt=prompt,
        input_ids=input_ids,
        offsets=offsets,
        token_signature=_token_signature(input_ids),
        context_char_span=context_char_span,
        context_token_span=context_token_span,
        paragraph_payload_spans=paragraph_payload_spans,
        diagnostics=tuple(diagnostics),
        bridge=bridge,
        positions=(),
        special_token_ids=special_token_ids,
    )
    return replace(
        prepared,
        positions=_select_positions(prepared, sample_seed=sample_seed),
    )
