import math
from types import SimpleNamespace

import pytest

from experiments.jlens_readout_sanity.constants import IDENTITY_ATOL, IDENTITY_RTOL
from experiments.jlens_readout_sanity.control_analysis import (
    _control_metadata,
    _wrong_reference_contexts,
    aggregate_all_checks,
    controls_passed,
    require_exact_cases,
    summarize_wrong_concept,
)
from jlens_reasoning.experiments_utils.tokens import concept_surfaces

EXPECTED_CASE_KEYS = (
    "spider",
    "france_capital",
    "france_language",
    "france_continent",
    "france_currency",
)


def _direction_context(key: str, source_id: int, target_id: int) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=SimpleNamespace(
            case=SimpleNamespace(key=key),
            source=SimpleNamespace(token_id=source_id),
            target=SimpleNamespace(token_id=target_id),
        )
    )


def _metadata_context(
    key: str,
    source_id: int,
    target_id: int,
    *,
    source_surface: str,
    target_surface: str,
    clean_answers: tuple[str, ...],
    target_answers: tuple[str, ...],
    formatting_token_id: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        resolved=SimpleNamespace(
            case=SimpleNamespace(
                key=key,
                source_surface=source_surface,
                target_surface=target_surface,
                target_answers=target_answers,
            ),
            read_case=SimpleNamespace(expected_answers=clean_answers),
            source=SimpleNamespace(token_id=source_id),
            target=SimpleNamespace(token_id=target_id),
        ),
        formatting_prefix=[{"token_id": formatting_token_id}],
    )


def test_control_metadata_derives_from_renamed_contexts() -> None:
    first = _metadata_context(
        "alpha",
        1,
        2,
        source_surface=" Wolf",
        target_surface=" Bear",
        clean_answers=("Pup",),
        target_answers=("Cub",),
        formatting_token_id=7,
    )
    second = _metadata_context(
        "beta",
        3,
        4,
        source_surface=" Oak",
        target_surface=" Pine",
        clean_answers=("Leaf",),
        target_answers=("Needle",),
        formatting_token_id=8,
    )

    metadata = _control_metadata((first, second))

    assert metadata.expected_keys == ("alpha", "beta")
    assert metadata.wrong_references == (second, first)
    assert metadata.source_surfaces == (
        *concept_surfaces("Wolf"),
        *concept_surfaces("Oak"),
    )
    assert metadata.target_surfaces == (
        *concept_surfaces("Bear"),
        *concept_surfaces("Pine"),
    )
    assert metadata.clean_answer_surfaces == (
        *concept_surfaces("Pup"),
        *concept_surfaces("Leaf"),
    )
    assert metadata.intended_answer_surfaces == (
        *concept_surfaces("Cub"),
        *concept_surfaces("Needle"),
    )
    assert metadata.formatting_token_ids == (7, 8)


def test_wrong_concept_references_derive_from_directions_not_case_names() -> None:
    first = _direction_context("alpha", 1, 2)
    second = _direction_context("beta", 3, 4)
    third = _direction_context("gamma", 3, 4)

    references = _wrong_reference_contexts((first, second, third))

    assert references == (second, first, first)


def test_wrong_concept_requires_two_distinct_directions() -> None:
    contexts = (
        _direction_context("alpha", 1, 2),
        _direction_context("beta", 1, 2),
    )

    with pytest.raises(ValueError, match="two distinct swap directions"):
        _wrong_reference_contexts(contexts)


def _gain_cases(gains: list[float]) -> list[dict[str, float | str]]:
    return [
        {"key": key, "log_rank_gain": gain}
        for key, gain in zip(EXPECTED_CASE_KEYS, gains, strict=True)
    ]


def _passing_controls() -> dict[str, dict[str, object]]:
    return {
        "identity": {
            "passed": True,
            "passed_case_count": 5,
            "required_case_count": 5,
            "maximum_absolute_logit_difference": 0.0,
        },
        "matched_random_vector": {
            "passed": True,
            "real_mean_log_rank_gain": 1.0,
            "percentile_95_threshold": 0.5,
        },
        "wrong_concept": {
            "passed": True,
            "matched_mean_log_rank_gain": 1.0,
            "mismatched_mean_log_rank_gain": 0.0,
            "matched_winning_case_count": 5,
            "required_winning_case_count": 4,
        },
        "random_target": {
            "passed": True,
            "real_mean_log_rank_gain": 1.0,
            "percentile_95_threshold": 0.5,
        },
    }


def test_real_rank_gain_cases_selects_numeric_control_alpha() -> None:
    from experiments.jlens_readout_sanity.control_analysis import (
        real_rank_gain_cases,
    )

    swaps = [
        {
            "key": "spider",
            "clean": {"target_rank": 20},
            "interventions": {
                "2.0": {"target_rank": 1, "top1_id": 9},
                "1": {"target_rank": 5, "top1_id": 8},
            },
        }
    ]

    cases, real_mean = real_rank_gain_cases(swaps, expected_keys=("spider",))

    assert cases == [
        {
            "key": "spider",
            "clean_rank": 20,
            "intervened_rank": 5,
            "intervened_top1_id": 8,
            "log_rank_gain": pytest.approx(math.log(4)),
        }
    ]
    assert real_mean == pytest.approx(math.log(4))


def test_assemble_control_results_preserves_the_serialized_envelope() -> None:
    from experiments.jlens_readout_sanity.control_analysis import (
        assemble_control_results,
    )

    passing = {"passed": True}
    result = assemble_control_results(
        expected_keys=EXPECTED_CASE_KEYS,
        identity=passing,
        matched_random_vector=passing,
        wrong_concept=passing,
        random_target=passing,
    )

    assert set(result) == {
        "seeds",
        "definitions",
        "thresholds",
        "tolerances",
        "identity",
        "matched_random_vector",
        "wrong_concept",
        "random_target",
        "passed",
    }
    assert result["definitions"]["expected_case_keys"] == list(EXPECTED_CASE_KEYS)
    assert result["thresholds"]["percentile_quantile"] == 0.95
    assert result["tolerances"]["identity_logits"] == {
        "atol": IDENTITY_ATOL,
        "rtol": IDENTITY_RTOL,
    }
    assert result["passed"] is True


def test_exact_case_validation_rejects_missing_duplicate_extra_and_order() -> None:
    complete = _gain_cases([1.0] * 5)
    require_exact_cases(complete, expected_keys=EXPECTED_CASE_KEYS)

    malformed = (
        complete[:-1],
        [*complete[:-1], complete[0]],
        [*complete, {"key": "extra", "log_rank_gain": 1.0}],
        list(reversed(complete)),
    )
    for cases in malformed:
        with pytest.raises(ValueError, match="exact case keys"):
            require_exact_cases(cases, expected_keys=EXPECTED_CASE_KEYS)


def test_wrong_concept_requires_aggregate_and_four_strict_case_wins() -> None:
    result = summarize_wrong_concept(
        _gain_cases([1.0, 1.0, 1.0, 1.0, 0.0]),
        _gain_cases([0.0, 0.0, 0.0, 0.0, 0.0]),
        expected_keys=EXPECTED_CASE_KEYS,
    )

    assert result["matched_mean_log_rank_gain"] == pytest.approx(0.8)
    assert result["mismatched_mean_log_rank_gain"] == 0.0
    assert result["matched_winning_case_count"] == 4
    assert result["aggregate_condition"] is True
    assert result["case_condition"] is True
    assert result["passed"] is True


def test_wrong_concept_ties_do_not_count_and_three_wins_fail() -> None:
    result = summarize_wrong_concept(
        _gain_cases([1.0, 1.0, 1.0, 0.0, 0.0]),
        _gain_cases([0.0, 0.0, 0.0, 0.0, -1.0]),
        expected_keys=EXPECTED_CASE_KEYS,
    )

    assert [case["matched_wins"] for case in result["cases"]] == [
        True,
        True,
        True,
        False,
        True,
    ]
    assert result["matched_winning_case_count"] == 4

    three_wins = summarize_wrong_concept(
        _gain_cases([1.0, 1.0, 1.0, 0.0, 0.0]),
        _gain_cases([0.0, 0.0, 0.0, 0.0, 0.0]),
        expected_keys=EXPECTED_CASE_KEYS,
    )
    assert three_wins["matched_winning_case_count"] == 3
    assert three_wins["passed"] is False


def test_all_four_controls_are_integrated_into_global_checks() -> None:
    controls = _passing_controls()

    checks, failures, passed = aggregate_all_checks(
        {"clean_baselines": True}, [], controls
    )

    assert checks == {
        "clean_baselines": True,
        "identity_control": True,
        "matched_random_vector_control": True,
        "wrong_concept_control": True,
        "random_target_control": True,
    }
    assert failures == []
    assert controls_passed(controls) is True
    assert passed is True


def test_control_and_global_pass_scopes_are_separate() -> None:
    controls = _passing_controls()

    _, failures, global_passed = aggregate_all_checks(
        {"clean_baselines": False}, ["existing failure"], controls
    )

    assert controls_passed(controls) is True
    assert global_passed is False
    assert failures == ["existing failure"]


@pytest.mark.parametrize(
    "control_name",
    ["identity", "matched_random_vector", "wrong_concept", "random_target"],
)
def test_each_failed_control_forces_global_failure_and_actionable_failure(
    control_name: str,
) -> None:
    controls = _passing_controls()
    controls[control_name]["passed"] = False

    checks, failures, passed = aggregate_all_checks(
        {"clean_baselines": True}, [], controls
    )

    assert checks[f"{control_name}_control"] is False
    assert controls_passed(controls) is False
    assert passed is False
    assert len(failures) == 1
    assert control_name.replace("_", " ") in failures[0]
    assert "required" in failures[0]


def test_missing_control_payload_is_rejected() -> None:
    controls = _passing_controls()
    del controls["identity"]

    with pytest.raises(KeyError, match="identity"):
        aggregate_all_checks({"clean_baselines": True}, [], controls)
