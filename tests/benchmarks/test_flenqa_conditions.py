from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt
from jlens_reasoning.benchmarks.flenqa_conditions import (
    PromptConditions,
    assert_unpadded_prompt_count,
    build_padding_positions,
    derive_conditions,
)
from jlens_reasoning.benchmarks.flenqa_preparation import prepare_prompt
from jlens_reasoning.benchmarks.flenqa_prompts import build_prompt_text


class CharTokenizer:
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, Any]:
        assert kwargs["truncation"] is False
        ids = [0, *(index + 10 for index in range(len(text)))]
        offsets = [(0, 0), *((index, index + 1) for index in range(len(text)))]
        return {"input_ids": [ids], "offset_mapping": [offsets]}


def _prompt(
    mixin: str,
    key_texts: tuple[str, ...],
    *,
    declared: tuple[str, ...] = ("books",),
    canonical_index: int = 0,
) -> FlenqaPrompt:
    question = "Is the conclusion true?"
    text = build_prompt_text(
        task="PIR",
        question=question,
        mixin=mixin,
        rule=None,
    )
    return FlenqaPrompt(
        canonical_index=canonical_index,
        prompt_id=f"{canonical_index:064x}",
        problem_id=canonical_index,
        task="PIR",
        text=text,
        question=question,
        key_texts=key_texts,
        rule=None,
        label=True,
        mixin=mixin,
        ctx_size_declared=250,
        source_row_ids=(canonical_index,),
        padding_type_declared=declared,
        dispersion_declared=("first",),
    )


def _prepare(
    mixin: str,
    key_texts: tuple[str, ...],
    *,
    declared: tuple[str, ...] = ("books",),
    canonical_index: int = 0,
):
    return prepare_prompt(
        _prompt(
            mixin,
            key_texts,
            declared=declared,
            canonical_index=canonical_index,
        ),
        CharTokenizer(),
    )


def test_unpadded_key_paragraphs_have_zero_padding_despite_separator_tokens() -> None:
    prepared = _prepare(
        "First key paragraph.\n \nSecond key paragraph.",
        ("First key paragraph.", "Second key paragraph."),
    )

    assert build_padding_positions(prepared) == ()
    conditions = derive_conditions(prepared)
    assert conditions == PromptConditions(
        padding_type_effective="none",
        dispersion_effective="not_applicable",
        frac_padding_before=0.0,
        frac_padding_between=0.0,
        frac_padding_after=0.0,
        n_padding_tokens=0,
    )


def test_padding_positions_overlap_only_non_key_paragraph_payloads() -> None:
    mixin = "First key.\n\n  Filler paragraph.  \nSecond key."
    prepared = _prepare(mixin, ("First key.", "Second key."))

    positions = build_padding_positions(prepared)
    chars = {
        prepared.prompt.text[start:end]
        for position in positions
        for start, end in (prepared.offsets[position],)
    }

    assert chars
    assert chars <= set("Filler paragraph. ")
    assert "\n" not in chars
    context_end = prepared.context_char_span.end
    assert all(prepared.offsets[position][1] <= context_end for position in positions)


def test_padded_type_comes_from_one_unanimous_declared_value() -> None:
    same = _prepare(
        "First key.\nFiller.\nSecond key.",
        ("First key.", "Second key."),
        declared=("same",),
    )

    assert derive_conditions(same).padding_type_effective == "same"

    disagreeing = _prepare(
        "First key.\nFiller.\nSecond key.",
        ("First key.", "Second key."),
        declared=("books", "same"),
    )
    with pytest.raises(ValueError, match="agree"):
        derive_conditions(disagreeing)


def test_unresolved_key_span_does_not_misclassify_prompt_as_unpadded() -> None:
    prepared = _prepare("First key.\nFiller.", ("Missing key.",))

    assert build_padding_positions(prepared) == ()
    conditions = derive_conditions(prepared)
    assert conditions.padding_type_effective is None
    assert conditions.dispersion_effective == "unresolved"


def test_placement_fractions_are_counts_of_model_token_positions() -> None:
    prepared = _prepare(
        "Before.\nFirst key.\nBetween.\nSecond key.\nAfter.",
        ("First key.", "Second key."),
    )

    conditions = derive_conditions(prepared)

    assert conditions.n_padding_tokens > 0
    assert (
        conditions.frac_padding_before
        + conditions.frac_padding_between
        + conditions.frac_padding_after
    ) == pytest.approx(1.0)
    assert conditions.frac_padding_between > 0
    assert conditions.dispersion_effective == "scattered"


def test_all_300_content_verified_unpadded_prompts_have_empty_padding_sets() -> None:
    prepared = tuple(
        _prepare(
            "First key.\n\nSecond key.",
            ("First key.", "Second key."),
            declared=("books", "same"),
            canonical_index=index,
        )
        for index in range(300)
    )

    assert_unpadded_prompt_count(prepared, expected=300)


def test_unpadded_count_invariant_rejects_a_structural_false_positive() -> None:
    prompts = [
        _prepare(
            "First key.\n\nSecond key.",
            ("First key.", "Second key."),
            canonical_index=index,
        )
        for index in range(299)
    ]
    prompts.append(
        _prepare(
            "First key.\nActual filler.\nSecond key.",
            ("First key.", "Second key."),
            canonical_index=299,
        )
    )

    with pytest.raises(ValueError, match="300"):
        assert_unpadded_prompt_count(prompts, expected=300)
