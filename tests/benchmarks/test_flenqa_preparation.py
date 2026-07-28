from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from typing import Any

import pytest
import torch

from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt
from jlens_reasoning.benchmarks.flenqa_preparation import (
    PreparedPrompt,
    prepare_prompt,
    resolve_key_paragraphs,
)
from jlens_reasoning.benchmarks.flenqa_prompts import build_prompt_text
from jlens_reasoning.experiments_utils.spans import CharSpan, SpanStatus


def _prompt(
    *,
    task: str = "PIR",
    mixin: str = "Alpha fact.\nBeta fact.",
    key_texts: tuple[str, ...] = ("Alpha fact.", "Beta fact."),
    question: str = "Is beta true?",
    rule: str | None = None,
    text: str | None = None,
) -> FlenqaPrompt:
    final_text = (
        build_prompt_text(task=task, question=question, mixin=mixin, rule=rule)
        if text is None
        else text
    )
    return FlenqaPrompt(
        canonical_index=0,
        prompt_id="prompt-0",
        problem_id=1,
        task=task,
        text=final_text,
        question=question,
        key_texts=key_texts,
        rule=rule,
        label=True,
        mixin=mixin,
        ctx_size_declared=250,
        source_row_ids=(0,),
        padding_type_declared=("books",),
        dispersion_declared=("first",),
    )


class RecordingCharTokenizer:
    def __init__(self, *, tensor_output: bool = False) -> None:
        self.tensor_output = tensor_output
        self.calls: list[tuple[str, dict[str, object]]] = []

    def __call__(self, text: str, **kwargs: object) -> Mapping[str, Any]:
        self.calls.append((text, kwargs))
        input_ids = [900, *(1000 + index for index in range(len(text))), 901]
        offsets = [(0, 0), *((index, index + 1) for index in range(len(text))), (0, 0)]
        if self.tensor_output:
            return {
                "input_ids": torch.tensor([input_ids]),
                "offset_mapping": torch.tensor([offsets]),
            }
        return {"input_ids": [input_ids], "offset_mapping": [offsets]}


def _diagnostics(prepared: PreparedPrompt, kind: str) -> list[Any]:
    return [
        diagnostic for diagnostic in prepared.diagnostics if diagnostic.kind == kind
    ]


def test_resolve_pir_facts_records_exact_ambiguous_and_unresolved_separately() -> None:
    prompt = _prompt(
        mixin="Exact.\nRepeated.\nRepeated.",
        key_texts=("Exact.", "Repeated.", "Missing."),
    )

    diagnostics = resolve_key_paragraphs(prompt)

    assert [(item.status, item.match_count) for item in diagnostics] == [
        (SpanStatus.OK, 1),
        (SpanStatus.AMBIGUOUS, 2),
        (SpanStatus.UNRESOLVED, 0),
    ]
    assert diagnostics[0].char_start == 0
    assert diagnostics[0].char_end == len("Exact.")
    assert diagnostics[1].char_start is None
    assert diagnostics[2].char_start is None


def test_resolve_pir_requires_an_exact_full_paragraph_payload() -> None:
    prompt = _prompt(
        mixin="Prefix Exact. Suffix\nExact.",
        key_texts=("Exact.",),
    )

    (diagnostic,) = resolve_key_paragraphs(prompt)

    assert diagnostic.status is SpanStatus.OK
    assert diagnostic.match_count == 1
    assert diagnostic.char_start == prompt.mixin.rindex("Exact.")
    assert diagnostic.char_end == len(prompt.mixin)


def test_ruletaker_expands_statement_to_unique_enclosing_paragraph() -> None:
    mixin = "The cow is young and the cow is kind.\nThe dog sleeps."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=mixin,
        key_texts=("The cow is young",),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    (diagnostic,) = resolve_key_paragraphs(prompt)

    paragraph = "The cow is young and the cow is kind."
    assert diagnostic.status is SpanStatus.OK
    assert diagnostic.match_count == 1
    assert diagnostic.surface == "The cow is young"
    assert (diagnostic.char_start, diagnostic.char_end) == (0, len(paragraph))


def test_ruletaker_counts_containing_paragraphs_not_sentence_repetitions() -> None:
    statement = "The cow is young."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{statement} Also, {statement}\nThe dog sleeps.",
        key_texts=(statement,),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    (diagnostic,) = resolve_key_paragraphs(prompt)

    assert diagnostic.status is SpanStatus.OK
    assert diagnostic.match_count == 1


@pytest.mark.parametrize(
    ("mixin", "expected_status", "expected_count"),
    [
        (
            "The cow is young.\nThe cow is young and kind.",
            SpanStatus.AMBIGUOUS,
            2,
        ),
        ("The dog sleeps.", SpanStatus.UNRESOLVED, 0),
    ],
)
def test_ruletaker_reports_ambiguous_or_missing_enclosing_paragraphs(
    mixin: str,
    expected_status: SpanStatus,
    expected_count: int,
) -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=mixin,
        key_texts=("The cow is young",),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    (diagnostic,) = resolve_key_paragraphs(prompt)

    assert diagnostic.status is expected_status
    assert diagnostic.match_count == expected_count
    assert diagnostic.char_start is None


def test_prepare_prompt_tokenizes_final_text_once_and_retains_tokenization() -> None:
    prompt = _prompt()
    tokenizer = RecordingCharTokenizer(tensor_output=True)

    prepared = prepare_prompt(prompt, tokenizer)

    assert tokenizer.calls == [
        (
            prompt.text,
            {
                "return_tensors": "pt",
                "truncation": False,
                "return_offsets_mapping": True,
            },
        )
    ]
    expected_ids = (900, *(1000 + index for index in range(len(prompt.text))), 901)
    expected_offsets = (
        (0, 0),
        *((index, index + 1) for index in range(len(prompt.text))),
        (0, 0),
    )
    assert prepared.input_ids == expected_ids
    assert prepared.offsets == expected_offsets
    packed = b"".join(struct.pack(">q", token_id) for token_id in expected_ids)
    assert prepared.token_signature == hashlib.sha256(packed).hexdigest()
    with pytest.raises(FrozenInstanceError):
        prepared.token_signature = "changed"  # type: ignore[misc]


def test_prepare_prompt_maps_context_payload_facts_and_question_from_one_offsets_map() -> (
    None
):
    prompt = _prompt(mixin="  Alpha fact.  \n\nBeta fact.\n")

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    context_start = prompt.text.index(prompt.mixin)
    assert prepared.context_char_span == CharSpan(
        context_start,
        context_start + len(prompt.mixin),
    )
    assert prepared.context_token_span == CharSpan(
        context_start + 1,
        context_start + len(prompt.mixin) + 1,
    )
    assert tuple(
        prompt.text[span.start : span.end] for span in prepared.paragraph_payload_spans
    ) == ("Alpha fact.", "Beta fact.")
    facts = _diagnostics(prepared, "fact")
    assert len(facts) == 2
    assert all(item.status is SpanStatus.OK for item in facts)
    assert prompt.text[facts[0].char_start : facts[0].char_end] == "Alpha fact."
    assert prepared.fact_token_spans == (
        CharSpan(facts[0].char_start + 1, facts[0].char_end + 1),
        CharSpan(facts[1].char_start + 1, facts[1].char_end + 1),
    )
    question = _diagnostics(prepared, "question")
    assert len(question) == 1
    assert prepared.question_token_span == CharSpan(
        question[0].char_start + 1,
        question[0].char_end + 1,
    )


def test_monorel_question_chooses_last_of_two_author_template_occurrences() -> None:
    prompt = _prompt(task="MonoRel")

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    (question,) = _diagnostics(prepared, "question")
    assert question.status is SpanStatus.OK
    assert question.match_count == 2
    assert question.char_start == prompt.text.rindex(prompt.question)
    assert question.char_end == question.char_start + len(prompt.question)


def test_prepare_prompt_records_rule_and_each_logical_target_separately() -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin="The cow is young and kind.\nThe dog is quiet.",
        key_texts=("The cow is young", "The dog is quiet"),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    assert len(_diagnostics(prepared, "context")) == 1
    assert len(_diagnostics(prepared, "fact")) == 2
    assert len(_diagnostics(prepared, "question")) == 1
    assert len(_diagnostics(prepared, "rule")) == 1
    first_fact = _diagnostics(prepared, "fact")[0]
    assert (
        prompt.text[first_fact.char_start : first_fact.char_end]
        == "The cow is young and kind."
    )


def test_prepare_prompt_records_bridge_per_fact_and_chooses_last_occurrence() -> None:
    prompt = _prompt(
        mixin="Ada called Bob, then Bob thanked Bob.\nBob greeted Ada.",
        key_texts=(
            "Ada called Bob, then Bob thanked Bob.",
            "Bob greeted Ada.",
        ),
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer(), bridge="Bob")

    bridges = _diagnostics(prepared, "bridge")
    facts = _diagnostics(prepared, "fact")
    assert [(item.fact_ordinal, item.status, item.match_count) for item in bridges] == [
        (0, SpanStatus.OK, 3),
        (1, SpanStatus.OK, 1),
    ]
    assert bridges[0].char_start == prompt.text.rindex(
        "Bob",
        facts[0].char_start,
        facts[0].char_end,
    )


def test_prepare_prompt_records_unresolved_bridge_for_each_resolved_fact() -> None:
    prepared = prepare_prompt(_prompt(), RecordingCharTokenizer(), bridge="Nobody")

    bridges = _diagnostics(prepared, "bridge")
    assert len(bridges) == 2
    assert all(item.status is SpanStatus.UNRESOLVED for item in bridges)
    assert all(item.match_count == 0 for item in bridges)


def test_ambiguous_context_nulls_dependent_final_bounds() -> None:
    prompt = _prompt()
    malformed_text = f"{prompt.mixin}\nscaffold\n{prompt.mixin}\n{prompt.question}"
    prompt = _prompt(text=malformed_text)

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    context = _diagnostics(prepared, "context")[0]
    assert context.status is SpanStatus.AMBIGUOUS
    assert context.match_count == 2
    assert prepared.context_char_span is None
    assert prepared.context_token_span is None
    assert prepared.paragraph_payload_spans == ()
    assert all(
        item.status is SpanStatus.AMBIGUOUS for item in _diagnostics(prepared, "fact")
    )
    assert all(item.char_start is None for item in _diagnostics(prepared, "fact"))


def test_ambiguous_context_preserves_local_bridge_resolution_semantics() -> None:
    prompt = _prompt(
        mixin="Ada called Bob.",
        key_texts=("Ada called Bob.",),
    )
    prompt = _prompt(
        mixin=prompt.mixin,
        key_texts=prompt.key_texts,
        text=f"{prompt.mixin}\nscaffold\n{prompt.mixin}\n{prompt.question}",
    )

    found = prepare_prompt(prompt, RecordingCharTokenizer(), bridge="Bob")
    missing = prepare_prompt(prompt, RecordingCharTokenizer(), bridge="Nobody")

    (found_bridge,) = _diagnostics(found, "bridge")
    (missing_bridge,) = _diagnostics(missing, "bridge")
    assert (found_bridge.status, found_bridge.match_count) == (
        SpanStatus.AMBIGUOUS,
        1,
    )
    assert found_bridge.char_start is None
    assert (missing_bridge.status, missing_bridge.match_count) == (
        SpanStatus.UNRESOLVED,
        0,
    )


class FixedLengthTokenizer:
    def __init__(self, n_tokens: int) -> None:
        self.n_tokens = n_tokens
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        n_tokens = (
            min(self.n_tokens, 4096)
            if kwargs.get("truncation") is True
            else self.n_tokens
        )
        return {
            "input_ids": [[*range(n_tokens)]],
            "offset_mapping": [[(0, len(text))] * n_tokens],
        }


def test_prepare_prompt_accepts_exactly_4096_untruncated_tokens() -> None:
    tokenizer = FixedLengthTokenizer(4096)

    prepared = prepare_prompt(_prompt(), tokenizer)

    assert len(prepared.input_ids) == 4096
    assert tokenizer.calls == [
        {
            "return_tensors": "pt",
            "truncation": False,
            "return_offsets_mapping": True,
        }
    ]


def test_prepare_prompt_rejects_4097_without_allowing_tokenizer_truncation() -> None:
    tokenizer = FixedLengthTokenizer(4097)

    with pytest.raises(ValueError, match="4097.*4096"):
        prepare_prompt(_prompt(), tokenizer)

    assert tokenizer.calls[0]["truncation"] is False


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"input_ids": [[]], "offset_mapping": [[]]},
        {"input_ids": [[1, 2]], "offset_mapping": [[(0, 1)]]},
        {"input_ids": [[1], [2]], "offset_mapping": [[(0, 1)], [(0, 1)]]},
        {"input_ids": [[1]], "offset_mapping": [[(1, 0)]]},
        {"input_ids": [[1]], "offset_mapping": [[(0, len(_prompt().text) + 1)]]},
        {"input_ids": [[1, 2]], "offset_mapping": [[(2, 3), (1, 2)]]},
    ],
)
def test_prepare_prompt_rejects_malformed_tokenizer_outputs(
    payload: dict[str, object],
) -> None:
    class MalformedTokenizer:
        def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
            return payload

    with pytest.raises(ValueError, match="tokenizer"):
        prepare_prompt(_prompt(), MalformedTokenizer())
