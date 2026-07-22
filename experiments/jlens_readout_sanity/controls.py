"""Stable facade and orchestration for J-Lens negative controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch

from experiments.jlens_readout_sanity.constants import CONTROL_REQUIRED_CASE_COUNT
from experiments.jlens_readout_sanity.control_analysis import (
    _control_metadata,
    _wrong_reference_contexts,
    aggregate_all_checks,
    assemble_control_results,
    controls_passed,
    real_rank_gain_cases,
    require_exact_cases,
    summarize_wrong_concept,
)
from experiments.jlens_readout_sanity.control_execution import (
    analyze_identity_case,
    run_identity_control,
    run_matched_random_vector_control,
    run_random_target_control,
    run_wrong_concept_control,
)
from experiments.jlens_readout_sanity.types import InterventionContext

__all__ = [
    "_control_metadata",
    "_wrong_reference_contexts",
    "aggregate_all_checks",
    "analyze_identity_case",
    "controls_passed",
    "require_exact_cases",
    "run_negative_controls",
    "summarize_wrong_concept",
]


def run_negative_controls(
    *,
    contexts: Sequence[InterventionContext],
    swap_results: Sequence[Mapping[str, Any]],
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Run the four fixed deterministic controls and assemble their report."""
    if len(contexts) != CONTROL_REQUIRED_CASE_COUNT:
        raise ValueError(
            f"Negative controls require exactly {CONTROL_REQUIRED_CASE_COUNT} cases"
        )
    metadata = _control_metadata(contexts)
    real_cases, real_mean = real_rank_gain_cases(
        swap_results,
        expected_keys=metadata.expected_keys,
    )
    identity = run_identity_control(
        contexts=contexts,
        expected_keys=metadata.expected_keys,
        model=model,
        forward_next_token=forward_next_token,
        layers=layers,
    )
    matched_random_vector = run_matched_random_vector_control(
        contexts=contexts,
        expected_keys=metadata.expected_keys,
        real_cases=real_cases,
        real_mean=real_mean,
        model=model,
        forward_next_token=forward_next_token,
        layers=layers,
    )
    wrong_concept = run_wrong_concept_control(
        contexts=contexts,
        wrong_references=metadata.wrong_references,
        expected_keys=metadata.expected_keys,
        real_cases=real_cases,
        model=model,
        forward_next_token=forward_next_token,
        layers=layers,
    )
    random_target = run_random_target_control(
        contexts=contexts,
        metadata=metadata,
        expected_keys=metadata.expected_keys,
        real_cases=real_cases,
        real_mean=real_mean,
        model=model,
        lens=lens,
        tokenizer=tokenizer,
        unembedding_weight=unembedding_weight,
        forward_next_token=forward_next_token,
        layers=layers,
    )
    return assemble_control_results(
        expected_keys=metadata.expected_keys,
        identity=identity,
        matched_random_vector=matched_random_vector,
        wrong_concept=wrong_concept,
        random_target=random_target,
    )
