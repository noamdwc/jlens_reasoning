from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from jlens_reasoning.benchmarks.flenqa.dataset import (
    FlenqaPrompt,
    SourceProvenance,
    build_prompt_text,
)
from jlens_reasoning.benchmarks.flenqa.positions import (
    PreparedPrompt,
    prepare_prompt,
    resolve_key_paragraphs,
    sample_padding_positions,
)
from jlens_reasoning.benchmarks.flenqa.positions_utils import (
    _eligible_padding_position,
)
from jlens_reasoning.experiments_utils.spans import CharSpan


def _prompt(
    *,
    task: str = "Simplified RuleTaker",
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
        prompt_id="0" * 64,
        problem_id=1,
        task=task,
        text=final_text,
        question=question,
        key_texts=key_texts,
        rule=rule,
        label=True,
        mixin=mixin,
        provenance=(SourceProvenance(0, 250, "books", "first"),),
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


def test_resolve_facts_preserves_every_matching_paragraph() -> None:
    prompt = _prompt(
        mixin="Exact.\nRepeated.\nRepeated.",
        key_texts=("Exact.", "Missing."),
    )

    with pytest.raises(ValueError, match="Missing"):
        resolve_key_paragraphs(prompt)

    resolved = resolve_key_paragraphs(
        _prompt(
            mixin="Exact.\nRepeated.\nRepeated.",
            key_texts=("Exact.", "Repeated."),
        )
    )
    assert [prompt.mixin[span.start : span.end] for _, _, span in resolved] == [
        "Exact.",
        "Repeated.",
        "Repeated.",
    ]


def test_resolve_pir_requires_an_exact_full_paragraph_payload() -> None:
    prompt = _prompt(
        task="PIR",
        mixin="Prefix Exact. Suffix\nExact.",
        key_texts=("Exact.",),
    )

    ((label, surface, span),) = resolve_key_paragraphs(prompt)

    assert (label, surface) == ("fact_a_end", "Exact.")
    assert span == CharSpan(prompt.mixin.rindex("Exact."), len(prompt.mixin))


def test_ruletaker_expands_statement_to_unique_enclosing_paragraph() -> None:
    mixin = "The cow is young and the cow is kind.\nThe dog sleeps."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=mixin,
        key_texts=("The cow is young",),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    ((label, surface, span),) = resolve_key_paragraphs(prompt)

    paragraph = "The cow is young and the cow is kind."
    assert (label, surface) == ("fact_a_end", "The cow is young")
    assert span == CharSpan(0, len(paragraph))


def test_ruletaker_counts_containing_paragraphs_not_sentence_repetitions() -> None:
    statement = "The cow is young."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{statement} Also, {statement}\nThe dog sleeps.",
        key_texts=(statement,),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    ((_, _, span),) = resolve_key_paragraphs(prompt)

    assert span == CharSpan(0, len(prompt.mixin.splitlines()[0]))


def test_ruletaker_preserves_multiple_containing_paragraphs() -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin="The cow is young.\nThe cow is young and kind.",
        key_texts=("The cow is young",),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    resolved = resolve_key_paragraphs(prompt)

    assert len(resolved) == 2
    assert [surface for _, surface, _ in resolved] == [
        "The cow is young",
        "The cow is young",
    ]


def test_ruletaker_reports_missing_enclosing_paragraph() -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin="The dog sleeps.",
        key_texts=("The cow is young",),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    with pytest.raises(ValueError, match="fact"):
        resolve_key_paragraphs(prompt)


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


def test_prepared_prompt_only_retains_assets_needed_by_runners_and_debugging() -> None:
    prepared = prepare_prompt(_prompt(), RecordingCharTokenizer())

    assert set(prepared.__dataclass_fields__) == {
        "prompt",
        "input_ids",
        "offsets",
        "positions",
    }


def test_prepare_prompt_maps_facts_and_question_from_one_offsets_map() -> None:
    prompt = _prompt(mixin="  Alpha fact.  \n\nBeta fact.\n")

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    assert prepared.positions["fact_a_end"] == (
        prompt.text.index("Alpha fact.") + len("Alpha fact."),
    )
    assert prepared.positions["fact_b_end"] == (
        prompt.text.index("Beta fact.") + len("Beta fact."),
    )
    assert prepared.positions["question_end"] == (
        prompt.text.rindex(prompt.question) + len(prompt.question),
    )


def test_monorel_question_chooses_last_of_two_author_template_occurrences() -> None:
    prompt = _prompt(task="MonoRel")

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    assert prepared.positions["question_end"] == (
        prompt.text.rindex(prompt.question) + len(prompt.question),
    )


def test_prepare_prompt_uses_the_full_ruletaker_fact_paragraph() -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin="The cow is young and kind.\nThe dog is quiet.",
        key_texts=("The cow is young", "The dog is quiet"),
        question="The cow is blue.",
        rule="If someone is young then they are blue.",
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    first_fact = "The cow is young and kind."
    assert prepared.positions["fact_a_end"] == (
        prompt.text.index(first_fact) + len(first_fact),
    )


def test_prepare_prompt_does_not_create_unused_rule_positions() -> None:
    rule = "The cow is young."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{rule}\nThe dog is quiet.",
        key_texts=(rule, "The dog is quiet."),
        question="The cow is blue.",
        rule=rule,
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    assert "rule" not in prepared.positions


def test_prepare_prompt_rejects_ambiguous_context() -> None:
    prompt = _prompt()
    malformed_text = f"{prompt.mixin}\nscaffold\n{prompt.mixin}\n{prompt.question}"
    prompt = _prompt(text=malformed_text)

    with pytest.raises(ValueError, match="context.*ambiguous"):
        prepare_prompt(prompt, RecordingCharTokenizer())


def test_prepare_prompt_rejects_ambiguous_context_before_fact_resolution() -> None:
    prompt = _prompt(
        mixin="Ada called Bob.",
        key_texts=("Ada called Bob.",),
    )
    prompt = _prompt(
        mixin=prompt.mixin,
        key_texts=prompt.key_texts,
        text=f"{prompt.mixin}\nscaffold\n{prompt.mixin}\n{prompt.question}",
    )

    with pytest.raises(ValueError, match="context.*ambiguous"):
        prepare_prompt(prompt, RecordingCharTokenizer())


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


@pytest.mark.parametrize(
    ("token_id", "offset", "special_ids", "expected"),
    [
        (0, (0, 4), frozenset({0}), False),
        (10, (4, 6), frozenset(), False),
        (11, (6, 7), frozenset(), True),
        (12, (7, 11), frozenset(), True),
    ],
)
def test_padding_candidates_exclude_special_and_whitespace_only_tokens(
    token_id: int,
    offset: tuple[int, int],
    special_ids: frozenset[int],
    expected: bool,
) -> None:
    assert (
        _eligible_padding_position(
            token_id=token_id,
            offset=offset,
            text="word \n,tail",
            padding_spans=(CharSpan(0, 11),),
            special_ids=special_ids,
        )
        is expected
    )


def test_unique_positions_keeps_labels_but_deduplicates_execution_positions() -> None:
    prepared = PreparedPrompt(
        prompt=_prompt(),
        input_ids=(1,),
        offsets=((0, 1),),
        positions={
            "question_end": (9,),
            "final_prompt": (9,),
        },
    )

    assert prepared.unique_positions == (9,)


def test_padding_sample_is_deterministic_for_prompt_id_and_seed() -> None:
    candidates = tuple(range(20))

    first = sample_padding_positions(
        candidates,
        prompt_id="a" * 64,
        sample_seed=1729,
        count=4,
    )
    second = sample_padding_positions(
        candidates,
        prompt_id="a" * 64,
        sample_seed=1729,
        count=4,
    )

    assert first == second
    assert len(first) == 4


def test_prepare_prompt_rejects_unresolved_required_fact() -> None:
    with pytest.raises(ValueError, match="fact"):
        prepare_prompt(
            _prompt(key_texts=("Missing fact.", "Beta fact.")),
            RecordingCharTokenizer(),
        )


def test_prepare_prompt_labels_all_repeated_ruletaker_facts() -> None:
    first = "Dave is good. First expansion."
    second = "Dave is small. Second expansion."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{first}\nPadding content.\n{second}\n{first}",
        key_texts=("Dave is good.", "Dave is small."),
        question="Dave is loud.",
        rule="If someone is good and small then they are loud.",
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    assert set(prepared.positions["fact_a_end"]) == {
        prompt.text.index(first) + len(first),
        prompt.text.rindex(first) + len(first),
    }
    fact_ranges = (
        (prompt.text.index(first), prompt.text.index(first) + len(first)),
        (prompt.text.index(second), prompt.text.index(second) + len(second)),
        (prompt.text.rindex(first), prompt.text.rindex(first) + len(first)),
    )
    assert all(
        not any(
            start < fact_end and end > fact_start
            for fact_start, fact_end in fact_ranges
        )
        for position in prepared.positions["sampled_padding"]
        for start, end in (prepared.offsets[position],)
    )


def test_identical_logical_facts_share_positions_without_duplicate_execution() -> None:
    fact = "Gary is dumb. Expanded fact."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{fact}\n{fact}",
        key_texts=("Gary is dumb.", "Gary is dumb."),
        question="Gary is red.",
        rule="If someone is dumb then they are red.",
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    fact_a = set(prepared.positions["fact_a_end"])
    fact_b = set(prepared.positions["fact_b_end"])
    assert fact_a == fact_b
    assert len(fact_a) == 2
    assert set(fact_a) <= set(prepared.unique_positions)


def test_prepare_prompt_selects_only_semantic_and_padding_content_positions() -> None:
    key_texts = (
        "John's living room is marble-floored.",
        "Ethan Washington is in John's living room.",
    )
    padding = "Padding, text."
    mixin = f"{key_texts[0]}\n{padding}\n{key_texts[1]}"
    question = "Is Ethan Washington in a marble-floored room?"
    prompt = FlenqaPrompt(
        canonical_index=0,
        prompt_id="1" * 64,
        problem_id=0,
        task="PIR",
        text=build_prompt_text(
            task="PIR",
            question=question,
            mixin=mixin,
            rule=None,
        ),
        question=question,
        key_texts=key_texts,
        rule=None,
        label=True,
        mixin=mixin,
        provenance=(SourceProvenance(0, 500, "books", "middle"),),
    )

    prepared = prepare_prompt(prompt, RecordingCharTokenizer())

    assert {
        "fact_a_end",
        "fact_b_end",
        "question_end",
        "final_prompt",
        "sampled_padding",
    } <= prepared.positions.keys()
    padding_start = prompt.text.index(padding)
    padding_end = padding_start + len(padding)
    assert all(
        padding_start <= prepared.offsets[position][0] < padding_end
        for position in prepared.positions["sampled_padding"]
    )
