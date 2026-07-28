"""Offline bridge-resolution gate for the 200 applicable FLenQA problems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from experiments.flenqa_length_drift.bridges import extract_bridge
from experiments.flenqa_length_drift.constants import MONOREL_TASK, PIR_TASK
from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt


@dataclass(frozen=True, slots=True)
class BridgeGateResult:
    applicable: int
    resolved: int


def bridge_gate(
    prompts: Sequence[FlenqaPrompt],
    *,
    expected_applicable: int = 200,
) -> BridgeGateResult:
    """Require one resolved, non-leaking bridge for every applicable problem."""
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

    resolved = 0
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
        resolved += 1
    return BridgeGateResult(applicable=applicable, resolved=resolved)
