from __future__ import annotations

from experiments.flenqa_length_drift.bridges import (
    bridge_candidate_surfaces,
    extract_bridge,
)
from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt
from jlens_reasoning.benchmarks.flenqa_prompts import build_prompt_text


def _prompt(
    *,
    task: str = "PIR",
    key_texts: tuple[str, ...],
    question: str,
    problem_id: int = 0,
) -> FlenqaPrompt:
    mixin = "\n".join(key_texts)
    rule = None
    text = build_prompt_text(
        task=task,
        question=question,
        mixin=mixin,
        rule=rule,
    )
    return FlenqaPrompt(
        canonical_index=problem_id,
        prompt_id=f"{problem_id:064x}",
        problem_id=problem_id,
        task=task,
        text=text,
        question=question,
        key_texts=key_texts,
        rule=rule,
        label=True,
        mixin=mixin,
        ctx_size_declared=250,
        source_row_ids=(problem_id,),
        padding_type_declared=("books",),
        dispersion_declared=("first",),
    )


def test_extracts_pir_possessive_room_bridge() -> None:
    prompt = _prompt(
        key_texts=(
            "John's living room is marble-floored, a reality that is clear.",
            "Ethan Washington is in John's living room, a fact well known.",
        ),
        question="Is Ethan Washington in a marble-floored room?",
    )

    assert extract_bridge(prompt) == "John's living room"


def test_extracts_monorel_middle_person_bridge() -> None:
    prompt = _prompt(
        task="MonoRel",
        key_texts=(
            "Julie Baker is younger than Julian Barton, a known fact.",
            "Samantha Arnold is younger than Julie Baker, also known.",
        ),
        question="Is Samantha Arnold younger than Julian Barton?",
    )

    assert extract_bridge(prompt) == "Julie Baker"


def test_ruletaker_has_no_bridge() -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        key_texts=("The cow is young.", "The cow is kind."),
        question="The cow is blue.",
    )

    assert extract_bridge(prompt) is None


def test_bridge_candidate_surfaces_include_full_and_head_variants() -> None:
    surfaces = bridge_candidate_surfaces("John's living room")

    assert "John's living room" in surfaces
    assert " John's living room" in surfaces
    assert "room" in surfaces
    assert " room" in surfaces
