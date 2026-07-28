"""Task-specific extraction of the unspoken FLenQA bridge entity."""

from __future__ import annotations

import re

from experiments.flenqa_length_drift.constants import (
    MONOREL_TASK,
    PIR_TASK,
    RULETAKER_TASK,
)
from jlens_reasoning.benchmarks.flenqa import FlenqaPrompt

_PERSON = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")
_POSSESSIVE_PHRASE = re.compile(
    r"\b[A-Z][A-Za-z'-]*'s(?:\s+[a-z][A-Za-z'-]*)+?"
    r"(?=\s+(?:is|was|has|contains|appears|looks)\b|[,.;])"
)


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


def bridge_candidate_surfaces(bridge: str) -> tuple[str, ...]:
    """Return ordered full-bridge and head-token surface variants."""
    if not bridge.strip():
        raise ValueError("bridge must be nonempty")
    head = bridge.split()[-1]
    ordered: list[str] = []
    for base in (bridge, bridge.lower(), bridge.capitalize(), head, head.capitalize()):
        for surface in (base, f" {base}"):
            if surface not in ordered:
                ordered.append(surface)
    return tuple(ordered)
