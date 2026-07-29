"""Deterministic labelled anchor and summary-position selection."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from experiments.flenqa_length_drift.constants import (
    ANCHOR_BUDGET,
    ANCHOR_PADDING_COUNT,
    FINAL_POSITION_COUNT,
    KEY_SPAN_SUMMARY_CAP,
    SUMMARY_POSITION_BUDGET,
)
from jlens_reasoning.benchmarks.flenqa_preparation import PreparedPrompt
from jlens_reasoning.experiments_utils.spans import SpanStatus

ANCHOR_FACT_A_END = "fact_a_end"
ANCHOR_FACT_B_END = "fact_b_end"
ANCHOR_BRIDGE_FACT_A = "bridge_fact_a"
ANCHOR_BRIDGE_FACT_B = "bridge_fact_b"
ANCHOR_QUESTION_END = "question_end"
ANCHOR_FINAL_PROMPT = "final_prompt"
ANCHOR_SAMPLED_PADDING = "sampled_padding"


@dataclass(frozen=True, slots=True)
class Anchor:
    label: str
    position: int


def prompt_seed(prompt_id: str) -> int:
    """Derive a stable sampling seed from a content hash."""
    try:
        return int(prompt_id[:16], 16)
    except ValueError as exc:
        raise ValueError("prompt_id must begin with hexadecimal characters") from exc


def _sample(
    positions: Sequence[int],
    *,
    count: int,
    seed: int,
) -> tuple[int, ...]:
    unique = sorted(set(positions))
    if count <= 0 or not unique:
        return ()
    rng = random.Random(seed)
    return tuple(sorted(rng.sample(unique, min(count, len(unique)))))


def select_anchors(
    prepared: PreparedPrompt,
    *,
    padding_positions: Sequence[int],
    seed: int,
) -> tuple[Anchor, ...]:
    """Select all semantic anchors plus seeded positions from explicit padding."""
    if not prepared.input_ids:
        raise ValueError("Cannot select anchors for an empty prompt")
    anchors: list[Anchor] = []
    fact_labels = (ANCHOR_FACT_A_END, ANCHOR_FACT_B_END)
    bridge_labels = (ANCHOR_BRIDGE_FACT_A, ANCHOR_BRIDGE_FACT_B)
    facts = [
        diagnostic for diagnostic in prepared.diagnostics if diagnostic.kind == "fact"
    ]
    for label, diagnostic in zip(fact_labels, facts, strict=False):
        if diagnostic.status is SpanStatus.OK and diagnostic.token_end is not None:
            anchors.append(Anchor(label, diagnostic.token_end - 1))
    bridges = [
        diagnostic for diagnostic in prepared.diagnostics if diagnostic.kind == "bridge"
    ]
    for label, diagnostic in zip(bridge_labels, bridges, strict=False):
        if diagnostic.status is SpanStatus.OK and diagnostic.token_end is not None:
            anchors.append(Anchor(label, diagnostic.token_end - 1))
    question = prepared.question_token_span
    if question is not None:
        anchors.append(Anchor(ANCHOR_QUESTION_END, question.end - 1))
    anchors.append(Anchor(ANCHOR_FINAL_PROMPT, len(prepared.input_ids) - 1))
    anchors.extend(
        Anchor(ANCHOR_SAMPLED_PADDING, position)
        for position in _sample(
            padding_positions,
            count=ANCHOR_PADDING_COUNT,
            seed=seed,
        )
    )
    if len(anchors) > ANCHOR_BUDGET:
        raise ValueError(
            f"Selected {len(anchors)} anchors, exceeding budget {ANCHOR_BUDGET}"
        )
    if any(
        anchor.position < 0 or anchor.position >= len(prepared.input_ids)
        for anchor in anchors
    ):
        raise ValueError("Anchor position is outside the prepared prompt")
    return tuple(anchors)


def select_summary_positions(
    prepared: PreparedPrompt,
    *,
    anchors: Sequence[Anchor],
    padding_positions: Sequence[int],
    seed: int,
) -> tuple[int, ...]:
    """Combine anchors, fact tails, final tokens, then explicit-padding fill."""
    positions = {anchor.position for anchor in anchors}
    for span in prepared.fact_token_spans:
        positions.update(
            range(max(span.start, span.end - KEY_SPAN_SUMMARY_CAP), span.end)
        )
    positions.update(
        range(
            max(0, len(prepared.input_ids) - FINAL_POSITION_COUNT),
            len(prepared.input_ids),
        )
    )
    if len(positions) > SUMMARY_POSITION_BUDGET:
        raise ValueError(
            f"Selected {len(positions)} mandatory summary positions, "
            f"exceeding budget {SUMMARY_POSITION_BUDGET}"
        )
    remaining = SUMMARY_POSITION_BUDGET - len(positions)
    candidates = tuple(
        position for position in padding_positions if position not in positions
    )
    positions.update(_sample(candidates, count=remaining, seed=seed))
    return tuple(sorted(positions))
