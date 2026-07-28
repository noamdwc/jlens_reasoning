from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.flenqa_length_drift.anchors import (
    ANCHOR_FINAL_PROMPT,
    ANCHOR_SAMPLED_PADDING,
    Anchor,
    prompt_seed,
    select_anchors,
    select_summary_positions,
)
from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt
from jlens_reasoning.benchmarks.flenqa_conditions import build_padding_positions
from jlens_reasoning.benchmarks.flenqa_preparation import prepare_prompt
from jlens_reasoning.benchmarks.flenqa_prompts import build_prompt_text


class CharTokenizer:
    def __call__(self, text: str, **kwargs: object) -> Mapping[str, Any]:
        ids = [0, *(index + 10 for index in range(len(text)))]
        offsets = [(0, 0), *((index, index + 1) for index in range(len(text)))]
        return {"input_ids": [ids], "offset_mapping": [offsets]}


def _prepared():
    mixin = "Ada called Bob.\nPadding words here.\nBob greeted Ada."
    question = "Did Ada meet someone?"
    prompt = FlenqaPrompt(
        canonical_index=0,
        prompt_id="12" * 32,
        problem_id=0,
        task="PIR",
        text=build_prompt_text(
            task="PIR",
            question=question,
            mixin=mixin,
            rule=None,
        ),
        question=question,
        key_texts=("Ada called Bob.", "Bob greeted Ada."),
        rule=None,
        label=True,
        mixin=mixin,
        ctx_size_declared=500,
        source_row_ids=(0,),
        padding_type_declared=("books",),
        dispersion_declared=("middle",),
    )
    return prepare_prompt(prompt, CharTokenizer(), bridge="Bob")


def test_anchor_selection_is_labelled_bounded_and_padding_safe() -> None:
    prepared = _prepared()
    padding = build_padding_positions(prepared)

    anchors = select_anchors(
        prepared,
        padding_positions=padding,
        seed=prompt_seed(prepared.prompt.prompt_id),
    )

    assert Anchor(ANCHOR_FINAL_PROMPT, len(prepared.input_ids) - 1) in anchors
    assert len(anchors) <= 12
    assert all(0 <= anchor.position < len(prepared.input_ids) for anchor in anchors)
    assert {
        anchor.position for anchor in anchors if anchor.label == ANCHOR_SAMPLED_PADDING
    } <= set(padding)


def test_anchor_and_summary_sampling_are_deterministic_and_padding_only() -> None:
    prepared = _prepared()
    padding = build_padding_positions(prepared)
    seed = prompt_seed(prepared.prompt.prompt_id)
    anchors = select_anchors(
        prepared,
        padding_positions=padding,
        seed=seed,
    )

    first = select_summary_positions(
        prepared,
        anchors=anchors,
        padding_positions=padding,
        seed=seed,
    )
    second = select_summary_positions(
        prepared,
        anchors=anchors,
        padding_positions=padding,
        seed=seed,
    )

    assert first == second
    assert {anchor.position for anchor in anchors} <= set(first)
    protected = {
        position
        for span in prepared.fact_token_spans
        for position in range(span.start, span.end)
    }
    sampled_fill = set(first) - {anchor.position for anchor in anchors} - protected
    sampled_fill -= set(
        range(max(0, len(prepared.input_ids) - 4), len(prepared.input_ids))
    )
    assert sampled_fill <= set(padding)


def test_unpadded_prompt_never_gets_sampled_padding_anchor() -> None:
    prepared = _prepared()
    anchors = select_anchors(
        prepared,
        padding_positions=(),
        seed=prompt_seed(prepared.prompt.prompt_id),
    )

    assert all(anchor.label != ANCHOR_SAMPLED_PADDING for anchor in anchors)
