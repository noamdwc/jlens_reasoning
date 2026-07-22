"""Model-backed negative-control execution for J-Lens readout sanity."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    LOW_PRECISION_NORM_ATOL,
    LOW_PRECISION_NORM_RTOL,
    MAX_RANDOM_VECTOR_ATTEMPTS,
    NORM_ATOL,
    NORM_RTOL,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
    RANDOM_TARGET_NAMESPACE,
    RANDOM_VECTOR_NAMESPACE,
    WRONG_CONCEPT_REQUIRED_CASE_WINS,
)
from experiments.jlens_readout_sanity.control_analysis import (
    ControlMetadata,
    require_exact_cases,
    summarize_wrong_concept,
)
from experiments.jlens_readout_sanity.types import InterventionContext
from jlens_reasoning.experiments_utils.controls import (
    build_random_target_exclusions,
    log_rank_gain,
    matched_random_vectors,
    mean,
    select_random_targets,
    strict_percentile_gate,
)
from jlens_reasoning.experiments_utils.interventions import (
    execute_intervention,
    single_token_vectors_by_layer,
)
from jlens_reasoning.experiments_utils.tokens import best_target_rank


def analyze_identity_case(
    *,
    key: str,
    clean_logits: torch.Tensor,
    scoring_input: torch.Tensor,
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    real_vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    target_ids: Sequence[int],
) -> dict[str, Any]:
    """Verify that swapping a source direction with itself is invariant."""
    identity_vectors = {
        layer: (source_vector, source_vector)
        for layer, (source_vector, _) in sorted(real_vectors_by_layer.items())
    }
    intervened_logits = execute_intervention(
        model=model,
        forward_next_token=forward_next_token,
        scoring_input=scoring_input,
        vectors_by_layer=identity_vectors,
        alpha=CONTROL_ALPHA,
    )
    clean = clean_logits.detach().float().cpu()
    intervened = intervened_logits.detach().float().cpu()
    clean_top1_id = int(clean.argmax().item())
    intervened_top1_id = int(intervened.argmax().item())
    clean_target_rank = best_target_rank(clean, target_ids)
    intervened_target_rank = best_target_rank(intervened, target_ids)
    maximum_difference = float((clean - intervened).abs().max().item())
    top1_unchanged = clean_top1_id == intervened_top1_id
    target_rank_unchanged = clean_target_rank == intervened_target_rank
    logits_close = bool(
        torch.allclose(
            clean,
            intervened,
            atol=IDENTITY_ATOL,
            rtol=IDENTITY_RTOL,
        )
    )
    return {
        "key": key,
        "workspace_layers": sorted(identity_vectors),
        "alpha": CONTROL_ALPHA,
        "atol": IDENTITY_ATOL,
        "rtol": IDENTITY_RTOL,
        "clean_top1_id": clean_top1_id,
        "intervened_top1_id": intervened_top1_id,
        "top1_unchanged": top1_unchanged,
        "clean_target_rank": clean_target_rank,
        "intervened_target_rank": intervened_target_rank,
        "target_rank_unchanged": target_rank_unchanged,
        "logits_close": logits_close,
        "maximum_absolute_logit_difference": maximum_difference,
        "passed": top1_unchanged and target_rank_unchanged and logits_close,
    }


def _rank_gain_payload(
    context: InterventionContext,
    intervened_logits: torch.Tensor,
) -> dict[str, Any]:
    clean = context.clean_logits.detach().float().cpu()
    intervened = intervened_logits.detach().float().cpu()
    clean_rank = best_target_rank(clean, context.target_ids)
    intervened_rank = best_target_rank(intervened, context.target_ids)
    return {
        "key": context.resolved.case.key,
        "intended_target_ids": list(context.target_ids),
        "clean_rank": clean_rank,
        "intervened_rank": intervened_rank,
        "intervened_top1_id": int(intervened.argmax().item()),
        "log_rank_gain": log_rank_gain(clean_rank, intervened_rank),
    }


def run_identity_control(
    *,
    contexts: Sequence[InterventionContext],
    expected_keys: Sequence[str],
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Run and aggregate source-to-self identity interventions."""
    cases = [
        analyze_identity_case(
            key=context.resolved.case.key,
            clean_logits=context.clean_logits,
            scoring_input=context.scoring_input,
            model=model,
            forward_next_token=forward_next_token,
            real_vectors_by_layer=context.real_vectors_by_layer,
            target_ids=context.target_ids,
        )
        for context in contexts
    ]
    require_exact_cases(cases, expected_keys=expected_keys)
    passed_count = sum(bool(case["passed"]) for case in cases)
    return {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "operation": "source concept to the same source concept",
            "workspace_layers": list(layers),
            "activation_positions": "all",
        },
        "cases": cases,
        "passed_case_count": passed_count,
        "required_case_count": len(expected_keys),
        "maximum_absolute_logit_difference": max(
            case["maximum_absolute_logit_difference"] for case in cases
        ),
        "passed": passed_count == len(expected_keys),
    }


def run_matched_random_vector_control(
    *,
    contexts: Sequence[InterventionContext],
    expected_keys: Sequence[str],
    real_cases: Sequence[Mapping[str, Any]],
    real_mean: float,
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Compare real swaps with deterministic matched-norm random vectors."""
    seed_results = []
    for seed in CONTROL_SEEDS:
        seed_cases = []
        norms_by_case = {}
        for context in contexts:
            random_vectors, norm_report = matched_random_vectors(
                context.real_vectors_by_layer,
                base_seed=seed,
                namespace=RANDOM_VECTOR_NAMESPACE,
                norm_atol=NORM_ATOL,
                norm_rtol=NORM_RTOL,
                low_precision_norm_atol=LOW_PRECISION_NORM_ATOL,
                low_precision_norm_rtol=LOW_PRECISION_NORM_RTOL,
                max_attempts=MAX_RANDOM_VECTOR_ATTEMPTS,
            )
            norms_by_case[context.resolved.case.key] = norm_report
            intervened_logits = execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=random_vectors,
                alpha=CONTROL_ALPHA,
            )
            seed_cases.append(_rank_gain_payload(context, intervened_logits))
            del intervened_logits, random_vectors
        require_exact_cases(seed_cases, expected_keys=expected_keys)
        seed_results.append(
            {
                "seed": seed,
                "cases": seed_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in seed_cases]
                ),
                "norms_by_case": norms_by_case,
            }
        )
    control_means = [result["mean_log_rank_gain"] for result in seed_results]
    gate = strict_percentile_gate(
        real_mean,
        control_means,
        quantile=PERCENTILE_QUANTILE,
        interpretation=PERCENTILE_INTERPRETATION,
    )
    return {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "generation_device": "cpu",
            "generation_dtype": "torch.float32",
            "output_device_dtype": "same as corresponding real vector",
        },
        "real_cases": real_cases,
        "real_mean_log_rank_gain": real_mean,
        "seeds": seed_results,
        "control_mean_log_rank_gains": control_means,
        "percentile_95_threshold": gate["threshold"],
        "gate": gate,
        "passed": gate["passed"],
    }


def run_wrong_concept_control(
    *,
    contexts: Sequence[InterventionContext],
    wrong_references: Sequence[InterventionContext],
    expected_keys: Sequence[str],
    real_cases: Sequence[Mapping[str, Any]],
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Compare matched directions with deliberately mismatched directions."""
    mismatched_cases = []
    mismatch_config = []
    for context, wrong_reference in zip(contexts, wrong_references, strict=True):
        intervened_logits = execute_intervention(
            model=model,
            forward_next_token=forward_next_token,
            scoring_input=context.scoring_input,
            vectors_by_layer=wrong_reference.real_vectors_by_layer,
            alpha=CONTROL_ALPHA,
        )
        mismatched_cases.append(_rank_gain_payload(context, intervened_logits))
        mismatch_config.append(
            {
                "key": context.resolved.case.key,
                "source": asdict(wrong_reference.resolved.source),
                "target": asdict(wrong_reference.resolved.target),
            }
        )
        del intervened_logits
    require_exact_cases(mismatched_cases, expected_keys=expected_keys)
    summary = summarize_wrong_concept(
        real_cases,
        mismatched_cases,
        expected_keys=expected_keys,
        required_winning_case_count=WRONG_CONCEPT_REQUIRED_CASE_WINS,
    )
    return {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "mismatches": mismatch_config,
        },
        "matched_cases": real_cases,
        "mismatched_cases": mismatched_cases,
        **summary,
    }


def run_random_target_control(
    *,
    contexts: Sequence[InterventionContext],
    metadata: ControlMetadata,
    expected_keys: Sequence[str],
    real_cases: Sequence[Mapping[str, Any]],
    real_mean: float,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    """Compare real targets with deterministic unrelated vocabulary targets."""
    exclusions = build_random_target_exclusions(
        tokenizer,
        source_surfaces=metadata.source_surfaces,
        target_surfaces=metadata.target_surfaces,
        clean_answer_surfaces=metadata.clean_answer_surfaces,
        intended_answer_surfaces=metadata.intended_answer_surfaces,
        formatting_token_ids=metadata.formatting_token_ids,
    )
    output_vocab_size = int(unembedding_weight.shape[0])
    selected_targets = select_random_targets(
        tokenizer,
        excluded_ids=exclusions["all"],
        seeds=CONTROL_SEEDS,
        output_vocab_size=output_vocab_size,
        namespace=RANDOM_TARGET_NAMESPACE,
    )
    target_results = []
    for selected in selected_targets:
        target_vectors_by_layer = single_token_vectors_by_layer(
            lens=lens,
            unembedding_weight=unembedding_weight,
            layers=layers,
            token_id=selected["token_id"],
        )
        target_cases = []
        for context in contexts:
            vectors_by_layer = {
                layer: (
                    context.real_vectors_by_layer[layer][0],
                    target_vectors_by_layer[layer],
                )
                for layer in layers
            }
            intervened_logits = execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=vectors_by_layer,
                alpha=CONTROL_ALPHA,
            )
            target_cases.append(_rank_gain_payload(context, intervened_logits))
            del intervened_logits, vectors_by_layer
        require_exact_cases(target_cases, expected_keys=expected_keys)
        target_results.append(
            {
                **selected,
                "cases": target_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in target_cases]
                ),
            }
        )
    control_means = [result["mean_log_rank_gain"] for result in target_results]
    gate = strict_percentile_gate(
        real_mean,
        control_means,
        quantile=PERCENTILE_QUANTILE,
        interpretation=PERCENTILE_INTERPRETATION,
    )
    return {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "selection": "SHA-256 index into ascending eligible token IDs",
            "output_vocab_size": output_vocab_size,
        },
        "exclusions": exclusions,
        "real_cases": real_cases,
        "real_mean_log_rank_gain": real_mean,
        "targets": target_results,
        "control_mean_log_rank_gains": control_means,
        "percentile_95_threshold": gate["threshold"],
        "gate": gate,
        "passed": gate["passed"],
    }
