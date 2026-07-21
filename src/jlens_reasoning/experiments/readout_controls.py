"""Execution and assembly of readout sanity negative controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from jlens_reasoning.experiments import intervention_utils as interventions
from jlens_reasoning.experiments.readout_cases import (
    READOUT_CASES,
    SWAP_CASES,
    _concept_surfaces,
)
from jlens_reasoning.experiments.sanity_constants import (
    CONTROL_ALPHA,
    CONTROL_CASE_KEYS,
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    LOW_PRECISION_NORM_ATOL,
    LOW_PRECISION_NORM_RTOL,
    NORM_ATOL,
    NORM_RTOL,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
    WRONG_CONCEPT_REQUIRED_CASE_WINS,
)
from jlens_reasoning.experiments.sanity_controls import (
    build_random_target_exclusions,
    controls_passed,
    log_rank_gain,
    matched_random_vectors,
    mean,
    require_exact_cases,
    select_random_targets,
    strict_percentile_gate,
    summarize_wrong_concept,
)


def run_negative_controls(
    *,
    contexts: Sequence[interventions.InterventionContext],
    swap_results: Sequence[Mapping[str, Any]],
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    require_exact_cases([{"key": context.resolved.case.key} for context in contexts])
    expected_keys = CONTROL_CASE_KEYS
    require_exact_cases(swap_results)
    real_cases = []
    for result in swap_results:
        alpha_one = interventions._intervention_payload_at_alpha(
            result["interventions"], CONTROL_ALPHA
        )
        real_cases.append(
            {
                "key": result["key"],
                "clean_rank": result["clean"]["target_rank"],
                "intervened_rank": alpha_one["target_rank"],
                "intervened_top1_id": alpha_one["top1_id"],
                "log_rank_gain": log_rank_gain(
                    result["clean"]["target_rank"],
                    alpha_one["target_rank"],
                ),
            }
        )
    require_exact_cases(real_cases)
    real_mean = mean([case["log_rank_gain"] for case in real_cases])

    identity_cases = [
        interventions.analyze_identity_case(
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
    require_exact_cases(identity_cases)
    identity_passed_count = sum(bool(case["passed"]) for case in identity_cases)
    identity = {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "operation": "source concept to the same source concept",
            "workspace_layers": list(layers),
            "activation_positions": "all",
        },
        "cases": identity_cases,
        "passed_case_count": identity_passed_count,
        "required_case_count": len(expected_keys),
        "maximum_absolute_logit_difference": max(
            case["maximum_absolute_logit_difference"] for case in identity_cases
        ),
        "passed": identity_passed_count == len(expected_keys),
    }

    random_vector_seed_results = []
    for seed in CONTROL_SEEDS:
        seed_cases = []
        norms_by_case = {}
        for context in contexts:
            random_vectors, norm_report = matched_random_vectors(
                context.real_vectors_by_layer,
                base_seed=seed,
            )
            norms_by_case[context.resolved.case.key] = norm_report
            intervened_logits = interventions.execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=random_vectors,
                alpha=CONTROL_ALPHA,
            )
            seed_cases.append(
                interventions._rank_gain_payload(context, intervened_logits)
            )
            del intervened_logits, random_vectors
        require_exact_cases(seed_cases)
        random_vector_seed_results.append(
            {
                "seed": seed,
                "cases": seed_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in seed_cases]
                ),
                "norms_by_case": norms_by_case,
            }
        )
    random_vector_means = [
        result["mean_log_rank_gain"] for result in random_vector_seed_results
    ]
    random_vector_gate = strict_percentile_gate(real_mean, random_vector_means)
    matched_random_vector = {
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
        "seeds": random_vector_seed_results,
        "control_mean_log_rank_gains": random_vector_means,
        "percentile_95_threshold": random_vector_gate["threshold"],
        "gate": random_vector_gate,
        "passed": random_vector_gate["passed"],
    }

    contexts_by_key = {context.resolved.case.key: context for context in contexts}
    spider_context = contexts_by_key["spider"]
    france_reference = contexts_by_key["france_capital"]
    mismatched_cases = []
    mismatch_config = []
    for context in contexts:
        wrong_reference = (
            france_reference
            if context.resolved.case.key == "spider"
            else spider_context
        )
        intervened_logits = interventions.execute_intervention(
            model=model,
            forward_next_token=forward_next_token,
            scoring_input=context.scoring_input,
            vectors_by_layer=wrong_reference.real_vectors_by_layer,
            alpha=CONTROL_ALPHA,
        )
        mismatched_cases.append(
            interventions._rank_gain_payload(context, intervened_logits)
        )
        mismatch_config.append(
            {
                "key": context.resolved.case.key,
                "source": asdict(wrong_reference.resolved.source),
                "target": asdict(wrong_reference.resolved.target),
            }
        )
        del intervened_logits
    require_exact_cases(mismatched_cases)
    wrong_summary = summarize_wrong_concept(
        real_cases,
        mismatched_cases,
        required_winning_case_count=WRONG_CONCEPT_REQUIRED_CASE_WINS,
    )
    wrong_concept = {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "mismatches": mismatch_config,
        },
        "matched_cases": real_cases,
        "mismatched_cases": mismatched_cases,
        **wrong_summary,
    }

    source_surfaces = tuple(
        surface
        for case in SWAP_CASES
        for surface in _concept_surfaces(case.source_surface.strip())
    )
    target_surfaces = tuple(
        surface
        for case in SWAP_CASES
        for surface in _concept_surfaces(case.target_surface.strip())
    )
    clean_answer_surfaces = tuple(
        surface
        for case in READOUT_CASES
        for answer in case.expected_answers
        for surface in _concept_surfaces(answer)
    )
    intended_answer_surfaces = tuple(
        surface
        for case in SWAP_CASES
        for answer in case.target_answers
        for surface in _concept_surfaces(answer)
    )
    formatting_token_ids = tuple(
        item["token_id"] for context in contexts for item in context.formatting_prefix
    )
    exclusions = build_random_target_exclusions(
        tokenizer,
        source_surfaces=source_surfaces,
        target_surfaces=target_surfaces,
        clean_answer_surfaces=clean_answer_surfaces,
        intended_answer_surfaces=intended_answer_surfaces,
        formatting_token_ids=formatting_token_ids,
    )
    output_vocab_size = int(unembedding_weight.shape[0])
    selected_targets = select_random_targets(
        tokenizer,
        excluded_ids=exclusions["all"],
        seeds=CONTROL_SEEDS,
        output_vocab_size=output_vocab_size,
    )
    random_target_results = []
    for selected in selected_targets:
        target_vectors_by_layer = interventions._single_token_vectors_by_layer(
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
            intervened_logits = interventions.execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=vectors_by_layer,
                alpha=CONTROL_ALPHA,
            )
            target_cases.append(
                interventions._rank_gain_payload(context, intervened_logits)
            )
            del intervened_logits, vectors_by_layer
        require_exact_cases(target_cases)
        random_target_results.append(
            {
                **selected,
                "cases": target_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in target_cases]
                ),
            }
        )
    random_target_means = [
        result["mean_log_rank_gain"] for result in random_target_results
    ]
    random_target_gate = strict_percentile_gate(real_mean, random_target_means)
    random_target = {
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
        "targets": random_target_results,
        "control_mean_log_rank_gains": random_target_means,
        "percentile_95_threshold": random_target_gate["threshold"],
        "gate": random_target_gate,
        "passed": random_target_gate["passed"],
    }

    control_results = {
        "identity": identity,
        "matched_random_vector": matched_random_vector,
        "wrong_concept": wrong_concept,
        "random_target": random_target,
    }
    return {
        "seeds": list(CONTROL_SEEDS),
        "definitions": {
            "log_rank_gain": "log(clean_rank) - log(intervened_rank)",
            "logarithm": "natural",
            "aggregate": "arithmetic mean across exactly five cases",
            "expected_case_keys": list(expected_keys),
            "percentile": "sort ascending; linear interpolation at (n - 1) * 0.95",
            "percentile_interpretation": PERCENTILE_INTERPRETATION,
            "comparison": "strictly greater than",
        },
        "thresholds": {
            "percentile_quantile": PERCENTILE_QUANTILE,
            "wrong_concept_required_case_wins": WRONG_CONCEPT_REQUIRED_CASE_WINS,
        },
        "tolerances": {
            "identity_logits": {
                "atol": IDENTITY_ATOL,
                "rtol": IDENTITY_RTOL,
            },
            "random_vector_norm_float32": {
                "atol": NORM_ATOL,
                "rtol": NORM_RTOL,
            },
            "random_vector_norm_low_precision": {
                "atol": LOW_PRECISION_NORM_ATOL,
                "rtol": LOW_PRECISION_NORM_RTOL,
            },
        },
        **control_results,
        "passed": controls_passed(control_results),
    }
