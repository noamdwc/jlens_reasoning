from __future__ import annotations

import pytest

from experiments.flenqa_length_drift.gate import bridge_gate
from tests.experiments.flenqa_length_drift.test_bridges import _prompt


def _applicable_prompts() -> list:
    pir = [
        _prompt(
            key_texts=(
                "John's living room is marble-floored.",
                "Ethan Washington is in John's living room.",
            ),
            question="Is Ethan Washington in a marble-floored room?",
            problem_id=index,
        )
        for index in range(100)
    ]
    monorel = [
        _prompt(
            task="MonoRel",
            key_texts=(
                "Julie Baker is younger than Julian Barton.",
                "Samantha Arnold is younger than Julie Baker.",
            ),
            question="Is Samantha Arnold younger than Julian Barton?",
            problem_id=100 + index,
        )
        for index in range(100)
    ]
    return [*pir, *monorel]


def test_bridge_gate_requires_and_resolves_200_applicable_problems() -> None:
    result = bridge_gate(_applicable_prompts())

    assert result.applicable == 200
    assert result.resolved == 200


def test_bridge_gate_cannot_pass_after_silently_skipping_one_task() -> None:
    with pytest.raises(ValueError, match="200"):
        bridge_gate(_applicable_prompts()[:100])


def test_bridge_gate_rejects_a_bridge_that_leaks_into_question() -> None:
    prompts = _applicable_prompts()
    prompts[0] = _prompt(
        key_texts=(
            "John's living room is marble-floored.",
            "Ethan Washington is in John's living room.",
        ),
        question="Is John's living room marble-floored?",
        problem_id=0,
    )

    with pytest.raises(ValueError, match="unresolved|question"):
        bridge_gate(prompts)
