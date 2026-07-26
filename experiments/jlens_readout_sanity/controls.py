"""Stable facade and orchestration for J-Lens negative controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    CONTROL_CHECK_MAP,
    CONTROL_REQUIRED_CASE_COUNT,
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
from experiments.jlens_readout_sanity.experiment import (
    ExperimentRuntime,
    InterventionResult,
    PreparedIntervention,
)
from jlens_reasoning.evaluation import (
    NextTokenEvaluation,
    RankComparison,
    compare_token_ranks,
    evaluate_next_token,
)
from jlens_reasoning.experiments_utils.controls import (
    build_random_target_exclusions,
    matched_random_vectors,
    mean,
    percentile_label,
    require_exact_case_keys,
    select_random_targets,
    strict_percentile_gate,
)
from jlens_reasoning.experiments_utils.interventions import (
    execute_intervention,
    single_token_vectors_by_layer,
)
from jlens_reasoning.experiments_utils.tokens import concept_surfaces

__all__ = [
    "aggregate_all_checks",
    "controls_passed",
    "evaluate_control_condition",
    "require_exact_cases",
    "run_control_suite",
    "summarize_wrong_concept",
]


@dataclass(frozen=True, slots=True)
class ControlConditionResult:
    evaluation: NextTokenEvaluation
    comparison: RankComparison


def evaluate_control_condition(
    *,
    clean: NextTokenEvaluation,
    intervened_logits: torch.Tensor,
    accepted_references: Sequence[str],
    tokenizer: object,
    top_k: int,
) -> ControlConditionResult:
    """Evaluate one control intervention against its reference condition."""
    candidate = evaluate_next_token(
        intervened_logits,
        accepted_references,
        tokenizer,
        top_k=top_k,
    )
    return ControlConditionResult(
        evaluation=candidate,
        comparison=compare_token_ranks(clean, candidate),
    )


def require_exact_cases(
    results: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[str],
) -> None:
    """Require caller-owned case keys in their supplied order."""
    require_exact_case_keys(results, expected_keys=expected_keys)


def summarize_wrong_concept(
    matched_cases: Sequence[Mapping[str, object]],
    mismatched_cases: Sequence[Mapping[str, object]],
    *,
    expected_keys: Sequence[str],
    required_winning_case_count: int = WRONG_CONCEPT_REQUIRED_CASE_WINS,
) -> dict[str, object]:
    """Compare matched direction swaps against mismatched swaps."""
    require_exact_cases(matched_cases, expected_keys=expected_keys)
    require_exact_cases(mismatched_cases, expected_keys=expected_keys)
    cases = []
    for matched, mismatched in zip(matched_cases, mismatched_cases, strict=True):
        matched_gain = float(matched["log_rank_gain"])
        mismatched_gain = float(mismatched["log_rank_gain"])
        cases.append(
            {
                "key": matched["key"],
                "matched_log_rank_gain": matched_gain,
                "mismatched_log_rank_gain": mismatched_gain,
                "comparison": "matched_log_rank_gain > mismatched_log_rank_gain",
                "matched_wins": matched_gain > mismatched_gain,
            }
        )
    matched_mean = mean([float(case["log_rank_gain"]) for case in matched_cases])
    mismatched_mean = mean([float(case["log_rank_gain"]) for case in mismatched_cases])
    winning_count = sum(bool(case["matched_wins"]) for case in cases)
    aggregate_condition = matched_mean > mismatched_mean
    case_condition = winning_count >= required_winning_case_count
    return {
        "cases": cases,
        "matched_mean_log_rank_gain": matched_mean,
        "mismatched_mean_log_rank_gain": mismatched_mean,
        "aggregate_comparison": (
            "matched_mean_log_rank_gain > mismatched_mean_log_rank_gain"
        ),
        "aggregate_condition": aggregate_condition,
        "matched_winning_case_count": winning_count,
        "required_winning_case_count": required_winning_case_count,
        "case_condition": case_condition,
        "passed": aggregate_condition and case_condition,
    }


def controls_passed(controls: Mapping[str, Mapping[str, object]]) -> bool:
    """Return whether every configured negative control passed."""
    return all(bool(controls[name]["passed"]) for name, _ in CONTROL_CHECK_MAP)


def assemble_control_results(
    *,
    expected_keys: Sequence[str],
    identity: Mapping[str, object],
    matched_random_vector: Mapping[str, object],
    wrong_concept: Mapping[str, object],
    random_target: Mapping[str, object],
) -> dict[str, object]:
    """Assemble the serialized envelope for all negative controls."""
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
            "aggregate": f"arithmetic mean across exactly {len(expected_keys)} cases",
            "expected_case_keys": list(expected_keys),
            "percentile": (
                "sort ascending; linear interpolation at "
                f"(n - 1) * {PERCENTILE_QUANTILE:g} "
                f"({percentile_label(PERCENTILE_QUANTILE)})"
            ),
            "percentile_interpretation": PERCENTILE_INTERPRETATION,
            "comparison": "strictly greater than",
        },
        "thresholds": {
            "percentile_quantile": PERCENTILE_QUANTILE,
            "wrong_concept_required_case_wins": WRONG_CONCEPT_REQUIRED_CASE_WINS,
        },
        "tolerances": {
            "identity_logits": {"atol": IDENTITY_ATOL, "rtol": IDENTITY_RTOL},
            "random_vector_norm_float32": {"atol": NORM_ATOL, "rtol": NORM_RTOL},
            "random_vector_norm_low_precision": {
                "atol": LOW_PRECISION_NORM_ATOL,
                "rtol": LOW_PRECISION_NORM_RTOL,
            },
        },
        **control_results,
        "passed": controls_passed(control_results),
    }


def _control_failure(name: str, control: Mapping[str, object]) -> str:
    """Explain why one named negative control failed."""
    if name == "identity":
        return (
            "identity control failed: "
            f"{control.get('passed_case_count', 0)}/"
            f"{control.get('required_case_count', 5)} cases passed; "
            "required every identity comparison to pass"
        )
    if name in {"matched_random_vector", "random_target"}:
        label = name.replace("_", " ")
        return (
            f"{label} control failed: real mean log-rank gain="
            f"{control.get('real_mean_log_rank_gain')!r}; required strictly > "
            f"{percentile_label(PERCENTILE_QUANTILE)} sanity threshold="
            f"{control.get('percentile_95_threshold')!r}"
        )
    if name == "wrong_concept":
        return (
            "wrong concept control failed: required matched mean > mismatched mean "
            f"and at least {WRONG_CONCEPT_REQUIRED_CASE_WINS} strict case wins"
        )
    raise KeyError(f"Unknown control: {name}")


def aggregate_all_checks(
    existing_checks: Mapping[str, bool],
    existing_failures: Sequence[str],
    controls: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, bool], list[str], bool]:
    """Add negative-control gates to existing capability checks."""
    checks = dict(existing_checks)
    failures = list(existing_failures)
    for control_name, check_name in CONTROL_CHECK_MAP:
        if control_name not in controls:
            raise KeyError(f"Missing control payload: {control_name}")
        passed = bool(controls[control_name]["passed"])
        checks[check_name] = passed
        if not passed:
            failures.append(_control_failure(control_name, controls[control_name]))
    return checks, failures, all(checks.values())


def _condition_payload(
    prepared: PreparedIntervention,
    clean: NextTokenEvaluation,
    intervened_logits: torch.Tensor,
    runtime: ExperimentRuntime,
    *,
    top_k: int,
) -> dict[str, object]:
    spec = prepared.case.intervention
    if spec is None:
        raise ValueError(f"Case {prepared.case.key!r} has no intervention")
    condition = evaluate_control_condition(
        clean=clean,
        intervened_logits=intervened_logits,
        accepted_references=spec.target_answers,
        tokenizer=runtime.tokenizer,
        top_k=top_k,
    )
    return {
        "key": prepared.case.key,
        "intended_target_ids": list(clean.accepted_token_ids),
        "clean_rank": condition.comparison.baseline_rank,
        "intervened_rank": condition.comparison.candidate_rank,
        "intervened_top1_id": condition.evaluation.top1_id,
        "log_rank_gain": condition.comparison.log_rank_gain,
    }


def _real_rank_gain_cases(
    interventions: Sequence[InterventionResult],
    prepared: Sequence[PreparedIntervention],
) -> tuple[list[dict[str, object]], float]:
    """Extract real alpha-one rank gains and their arithmetic mean."""
    real_cases = []
    for result, context in zip(interventions, prepared, strict=True):
        matches = [
            condition
            for condition in result.conditions
            if float(condition.alpha) == float(CONTROL_ALPHA)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Case {context.case.key!r} needs exactly one alpha={CONTROL_ALPHA:g} condition"
            )
        condition = matches[0]
        real_cases.append(
            {
                "key": context.case.key,
                "clean_rank": condition.comparison.baseline_rank,
                "intervened_rank": condition.comparison.candidate_rank,
                "intervened_top1_id": condition.evaluation.top1_id,
                "log_rank_gain": condition.comparison.log_rank_gain,
            }
        )
    return real_cases, mean([float(case["log_rank_gain"]) for case in real_cases])


def _identity_control(
    prepared: Sequence[PreparedIntervention],
    interventions: Sequence[InterventionResult],
    runtime: ExperimentRuntime,
    layers: Sequence[int],
    *,
    top_k: int,
) -> dict[str, object]:
    """Verify source-to-self swaps preserve logits and evaluated ranks."""
    cases = []
    for context, real in zip(prepared, interventions, strict=True):
        identity_vectors = {
            layer: (source, source)
            for layer, (source, _) in sorted(context.vectors_by_layer.items())
        }
        logits = execute_intervention(
            model=runtime.model,
            forward_next_token=runtime.forward_next_token,
            scoring_input=context.scoring_input,
            vectors_by_layer=identity_vectors,
            alpha=CONTROL_ALPHA,
        )
        condition = evaluate_control_condition(
            clean=real.clean_target,
            intervened_logits=logits,
            accepted_references=real.clean_target.accepted_references,
            tokenizer=runtime.tokenizer,
            top_k=top_k,
        )
        clean_logits = context.clean_logits.detach().float().cpu()
        candidate_logits = logits.detach().float().cpu()
        maximum_difference = float((clean_logits - candidate_logits).abs().max().item())
        top1_unchanged = real.clean_target.top1_id == condition.evaluation.top1_id
        rank_unchanged = condition.comparison.rank_gain == 0
        logits_close = bool(
            torch.allclose(
                clean_logits,
                candidate_logits,
                atol=IDENTITY_ATOL,
                rtol=IDENTITY_RTOL,
            )
        )
        cases.append(
            {
                "key": context.case.key,
                "workspace_layers": sorted(identity_vectors),
                "alpha": CONTROL_ALPHA,
                "atol": IDENTITY_ATOL,
                "rtol": IDENTITY_RTOL,
                "clean_top1_id": real.clean_target.top1_id,
                "intervened_top1_id": condition.evaluation.top1_id,
                "top1_unchanged": top1_unchanged,
                "clean_target_rank": condition.comparison.baseline_rank,
                "intervened_target_rank": condition.comparison.candidate_rank,
                "target_rank_unchanged": rank_unchanged,
                "logits_close": logits_close,
                "maximum_absolute_logit_difference": maximum_difference,
                "passed": top1_unchanged and rank_unchanged and logits_close,
            }
        )
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
        "required_case_count": len(cases),
        "maximum_absolute_logit_difference": max(
            float(case["maximum_absolute_logit_difference"]) for case in cases
        ),
        "passed": passed_count == len(cases),
    }


def _matched_random_vector_control(
    prepared: Sequence[PreparedIntervention],
    interventions: Sequence[InterventionResult],
    runtime: ExperimentRuntime,
    layers: Sequence[int],
    real_cases: Sequence[Mapping[str, object]],
    real_mean: float,
    *,
    top_k: int,
) -> dict[str, object]:
    """Compare real swaps with deterministic matched-norm random vectors."""
    seed_results = []
    for seed in CONTROL_SEEDS:
        seed_cases = []
        norms_by_case = {}
        for context, real in zip(prepared, interventions, strict=True):
            vectors, norm_report = matched_random_vectors(
                context.vectors_by_layer,
                base_seed=seed,
                namespace=RANDOM_VECTOR_NAMESPACE,
                norm_atol=NORM_ATOL,
                norm_rtol=NORM_RTOL,
                low_precision_norm_atol=LOW_PRECISION_NORM_ATOL,
                low_precision_norm_rtol=LOW_PRECISION_NORM_RTOL,
                max_attempts=MAX_RANDOM_VECTOR_ATTEMPTS,
            )
            norms_by_case[context.case.key] = norm_report
            logits = execute_intervention(
                model=runtime.model,
                forward_next_token=runtime.forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=vectors,
                alpha=CONTROL_ALPHA,
            )
            seed_cases.append(
                _condition_payload(
                    context, real.clean_target, logits, runtime, top_k=top_k
                )
            )
        seed_results.append(
            {
                "seed": seed,
                "cases": seed_cases,
                "mean_log_rank_gain": mean(
                    [float(case["log_rank_gain"]) for case in seed_cases]
                ),
                "norms_by_case": norms_by_case,
            }
        )
    control_means = [float(result["mean_log_rank_gain"]) for result in seed_results]
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
        "real_cases": list(real_cases),
        "real_mean_log_rank_gain": real_mean,
        "seeds": seed_results,
        "control_mean_log_rank_gains": control_means,
        "percentile_95_threshold": gate["threshold"],
        "gate": gate,
        "passed": gate["passed"],
    }


def _wrong_concept_control(
    prepared: Sequence[PreparedIntervention],
    interventions: Sequence[InterventionResult],
    runtime: ExperimentRuntime,
    layers: Sequence[int],
    real_cases: Sequence[Mapping[str, object]],
    *,
    top_k: int,
) -> dict[str, object]:
    """Compare matched swaps with deliberately mismatched directions."""
    references = []
    for context in prepared:
        direction = (context.source.token_id, context.target.token_id)
        reference = next(
            (
                candidate
                for candidate in prepared
                if (candidate.source.token_id, candidate.target.token_id) != direction
            ),
            None,
        )
        if reference is None:
            raise ValueError(
                "Wrong-concept control requires at least two distinct swap directions"
            )
        references.append(reference)

    mismatched_cases = []
    mismatch_config = []
    for context, real, reference in zip(
        prepared,
        interventions,
        references,
        strict=True,
    ):
        logits = execute_intervention(
            model=runtime.model,
            forward_next_token=runtime.forward_next_token,
            scoring_input=context.scoring_input,
            vectors_by_layer=reference.vectors_by_layer,
            alpha=CONTROL_ALPHA,
        )
        mismatched_cases.append(
            _condition_payload(context, real.clean_target, logits, runtime, top_k=top_k)
        )
        mismatch_config.append(
            {
                "key": context.case.key,
                "source": {
                    "surface": reference.source.surface,
                    "token_id": reference.source.token_id,
                },
                "target": {
                    "surface": reference.target.surface,
                    "token_id": reference.target.token_id,
                },
            }
        )
    summary = summarize_wrong_concept(
        real_cases,
        mismatched_cases,
        expected_keys=tuple(context.case.key for context in prepared),
        required_winning_case_count=WRONG_CONCEPT_REQUIRED_CASE_WINS,
    )
    return {
        "configuration": {
            "alpha": CONTROL_ALPHA,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "mismatches": mismatch_config,
        },
        "matched_cases": list(real_cases),
        "mismatched_cases": mismatched_cases,
        **summary,
    }


def _random_target_control(
    prepared: Sequence[PreparedIntervention],
    interventions: Sequence[InterventionResult],
    runtime: ExperimentRuntime,
    layers: Sequence[int],
    real_cases: Sequence[Mapping[str, object]],
    real_mean: float,
    *,
    top_k: int,
) -> dict[str, object]:
    """Compare intended targets with deterministic unrelated token targets."""
    specs = [context.case.intervention for context in prepared]
    if any(spec is None for spec in specs):
        raise ValueError("Control cases must configure interventions")
    exclusions = build_random_target_exclusions(
        runtime.tokenizer,
        source_surfaces=tuple(
            surface
            for spec in specs
            if spec is not None
            for surface in concept_surfaces(spec.source_surface.strip())
        ),
        target_surfaces=tuple(
            surface
            for spec in specs
            if spec is not None
            for surface in concept_surfaces(spec.target_surface.strip())
        ),
        clean_answer_surfaces=tuple(
            surface
            for context in prepared
            for answer in context.case.expected_answers
            for surface in concept_surfaces(answer)
        ),
        intended_answer_surfaces=tuple(
            surface
            for spec in specs
            if spec is not None
            for answer in spec.target_answers
            for surface in concept_surfaces(answer)
        ),
        formatting_token_ids=tuple(
            int(item["token_id"])
            for context in prepared
            for item in context.formatting_prefix
        ),
    )
    output_vocab_size = int(runtime.unembedding_weight.shape[0])
    selected_targets = select_random_targets(
        runtime.tokenizer,
        excluded_ids=exclusions["all"],
        seeds=CONTROL_SEEDS,
        output_vocab_size=output_vocab_size,
        namespace=RANDOM_TARGET_NAMESPACE,
    )
    target_results = []
    for selected in selected_targets:
        target_vectors = single_token_vectors_by_layer(
            lens=runtime.lens,
            unembedding_weight=runtime.unembedding_weight,
            layers=layers,
            token_id=int(selected["token_id"]),
        )
        target_cases = []
        for context, real in zip(prepared, interventions, strict=True):
            vectors = {
                layer: (context.vectors_by_layer[layer][0], target_vectors[layer])
                for layer in layers
            }
            logits = execute_intervention(
                model=runtime.model,
                forward_next_token=runtime.forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=vectors,
                alpha=CONTROL_ALPHA,
            )
            target_cases.append(
                _condition_payload(
                    context, real.clean_target, logits, runtime, top_k=top_k
                )
            )
        target_results.append(
            {
                **selected,
                "cases": target_cases,
                "mean_log_rank_gain": mean(
                    [float(case["log_rank_gain"]) for case in target_cases]
                ),
            }
        )
    control_means = [float(result["mean_log_rank_gain"]) for result in target_results]
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
        "real_cases": list(real_cases),
        "real_mean_log_rank_gain": real_mean,
        "targets": target_results,
        "control_mean_log_rank_gains": control_means,
        "percentile_95_threshold": gate["threshold"],
        "gate": gate,
        "passed": gate["passed"],
    }


def run_control_suite(
    *,
    prepared: Sequence[PreparedIntervention],
    interventions: Sequence[InterventionResult],
    runtime: ExperimentRuntime,
    layers: Sequence[int],
    top_k: int,
) -> dict[str, object]:
    """Run deterministic negative controls and apply their aggregate gates."""
    if len(prepared) != CONTROL_REQUIRED_CASE_COUNT or len(interventions) != len(
        prepared
    ):
        raise ValueError(
            f"Negative controls require exactly {CONTROL_REQUIRED_CASE_COUNT} interventions"
        )
    expected_keys = tuple(context.case.key for context in prepared)
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("Control case keys must be unique")
    real_cases, real_mean = _real_rank_gain_cases(interventions, prepared)
    identity = _identity_control(
        prepared,
        interventions,
        runtime,
        layers,
        top_k=top_k,
    )
    matched_random = _matched_random_vector_control(
        prepared,
        interventions,
        runtime,
        layers,
        real_cases,
        real_mean,
        top_k=top_k,
    )
    wrong_concept = _wrong_concept_control(
        prepared,
        interventions,
        runtime,
        layers,
        real_cases,
        top_k=top_k,
    )
    random_target = _random_target_control(
        prepared,
        interventions,
        runtime,
        layers,
        real_cases,
        real_mean,
        top_k=top_k,
    )
    return assemble_control_results(
        expected_keys=expected_keys,
        identity=identity,
        matched_random_vector=matched_random,
        wrong_concept=wrong_concept,
        random_target=random_target,
    )
