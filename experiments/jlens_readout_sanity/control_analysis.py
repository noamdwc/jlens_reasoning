"""Pure preparation and result analysis for J-Lens negative controls."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    CONTROL_CHECK_MAP,
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
from experiments.jlens_readout_sanity.types import InterventionContext
from jlens_reasoning.experiments_utils.controls import (
    log_rank_gain,
    mean,
    percentile_label,
    require_exact_case_keys,
)
from jlens_reasoning.experiments_utils.tokens import concept_surfaces


def _intervention_payload_at_alpha(
    interventions: Mapping[str, Mapping[str, Any]],
    alpha: float,
) -> Mapping[str, Any]:
    matches = [
        payload for key, payload in interventions.items() if float(key) == float(alpha)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one intervention payload for alpha={alpha}")
    return matches[0]


def require_exact_cases(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> None:
    """Require caller-owned case keys in their supplied order."""
    require_exact_case_keys(results, expected_keys=expected_keys)


def real_rank_gain_cases(
    swap_results: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> tuple[list[dict[str, Any]], float]:
    """Extract the real control-alpha rank gains from swap results."""
    require_exact_cases(swap_results, expected_keys=expected_keys)
    real_cases = []
    for result in swap_results:
        alpha_one = _intervention_payload_at_alpha(
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
    require_exact_cases(real_cases, expected_keys=expected_keys)
    return real_cases, mean([case["log_rank_gain"] for case in real_cases])


@dataclass(frozen=True, slots=True)
class ControlMetadata:
    expected_keys: tuple[str, ...]
    wrong_references: tuple[InterventionContext, ...]
    source_surfaces: tuple[str, ...]
    target_surfaces: tuple[str, ...]
    clean_answer_surfaces: tuple[str, ...]
    intended_answer_surfaces: tuple[str, ...]
    formatting_token_ids: tuple[int, ...]


def _wrong_reference_contexts(
    contexts: Sequence[InterventionContext],
) -> tuple[InterventionContext, ...]:
    references = []
    for context in contexts:
        direction = (
            context.resolved.source.token_id,
            context.resolved.target.token_id,
        )
        reference = next(
            (
                candidate
                for candidate in contexts
                if (
                    candidate.resolved.source.token_id,
                    candidate.resolved.target.token_id,
                )
                != direction
            ),
            None,
        )
        if reference is None:
            raise ValueError(
                "Wrong-concept control requires at least two distinct swap directions"
            )
        references.append(reference)
    return tuple(references)


def _control_metadata(
    contexts: Sequence[InterventionContext],
) -> ControlMetadata:
    return ControlMetadata(
        expected_keys=tuple(context.resolved.case.key for context in contexts),
        wrong_references=_wrong_reference_contexts(contexts),
        source_surfaces=tuple(
            surface
            for context in contexts
            for surface in concept_surfaces(
                context.resolved.case.source_surface.strip()
            )
        ),
        target_surfaces=tuple(
            surface
            for context in contexts
            for surface in concept_surfaces(
                context.resolved.case.target_surface.strip()
            )
        ),
        clean_answer_surfaces=tuple(
            surface
            for context in contexts
            for answer in context.resolved.read_case.expected_answers
            for surface in concept_surfaces(answer)
        ),
        intended_answer_surfaces=tuple(
            surface
            for context in contexts
            for answer in context.resolved.case.target_answers
            for surface in concept_surfaces(answer)
        ),
        formatting_token_ids=tuple(
            item["token_id"]
            for context in contexts
            for item in context.formatting_prefix
        ),
    )


def summarize_wrong_concept(
    matched_cases: Sequence[Mapping[str, Any]],
    mismatched_cases: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
    required_winning_case_count: int = WRONG_CONCEPT_REQUIRED_CASE_WINS,
) -> dict[str, Any]:
    """Compare matched direction swaps against deliberately mismatched swaps."""
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
    winning_case_count = sum(bool(case["matched_wins"]) for case in cases)
    aggregate_condition = matched_mean > mismatched_mean
    case_condition = winning_case_count >= required_winning_case_count
    return {
        "cases": cases,
        "matched_mean_log_rank_gain": matched_mean,
        "mismatched_mean_log_rank_gain": mismatched_mean,
        "aggregate_comparison": (
            "matched_mean_log_rank_gain > mismatched_mean_log_rank_gain"
        ),
        "aggregate_condition": aggregate_condition,
        "matched_winning_case_count": winning_case_count,
        "required_winning_case_count": required_winning_case_count,
        "case_condition": case_condition,
        "passed": aggregate_condition and case_condition,
    }


def controls_passed(controls: Mapping[str, Mapping[str, Any]]) -> bool:
    """Return whether every configured negative control passed."""
    return all(bool(controls[name]["passed"]) for name, _ in CONTROL_CHECK_MAP)


def assemble_control_results(
    *,
    expected_keys: Sequence[str],
    identity: Mapping[str, Any],
    matched_random_vector: Mapping[str, Any],
    wrong_concept: Mapping[str, Any],
    random_target: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the stable serialized envelope for all negative controls."""
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
            "aggregate": (f"arithmetic mean across exactly {len(expected_keys)} cases"),
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


def _control_failure(name: str, control: Mapping[str, Any]) -> str:
    if name == "identity":
        return (
            "identity control failed: "
            f"{control.get('passed_case_count', 0)}/"
            f"{control.get('required_case_count', 5)} cases passed; "
            "required every identity comparison to pass; "
            "maximum absolute logit difference="
            f"{control.get('maximum_absolute_logit_difference')!r}"
        )
    if name == "matched_random_vector":
        return (
            "matched random vector control failed: real mean log-rank gain="
            f"{control.get('real_mean_log_rank_gain')!r}; required strictly > "
            f"{percentile_label(PERCENTILE_QUANTILE)} sanity threshold="
            f"{control.get('percentile_95_threshold')!r}"
        )
    if name == "wrong_concept":
        return (
            "wrong concept control failed: matched mean log-rank gain="
            f"{control.get('matched_mean_log_rank_gain')!r}, mismatched mean="
            f"{control.get('mismatched_mean_log_rank_gain')!r}; matched wins="
            f"{control.get('matched_winning_case_count')!r}; required matched "
            "mean > mismatched mean and at least "
            f"{control.get('required_winning_case_count', WRONG_CONCEPT_REQUIRED_CASE_WINS)} "
            "strict case wins"
        )
    if name == "random_target":
        return (
            "random target control failed: real mean log-rank gain="
            f"{control.get('real_mean_log_rank_gain')!r}; required strictly > "
            f"{percentile_label(PERCENTILE_QUANTILE)} sanity threshold="
            f"{control.get('percentile_95_threshold')!r}"
        )
    raise KeyError(f"Unknown control: {name}")


def aggregate_all_checks(
    existing_checks: Mapping[str, bool],
    existing_failures: Sequence[str],
    controls: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, bool], list[str], bool]:
    """Add all control gates to existing experiment capability gates."""
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
