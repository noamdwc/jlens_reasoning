from copy import deepcopy

import pytest

from experiments.jlens_readout_sanity.reporting import (
    capability_rows,
    control_rows,
    render_sanity_report,
    swap_rows,
)


def sample_result() -> dict[str, object]:
    return {
        "passed": False,
        "failures": ["random target control failed"],
        "provenance": {
            "project_commit": "abc123",
            "jlens": "0.1.0",
            "torch": "2.7.0",
            "transformers": "5.5.0",
        },
        "intervention_strengths": [1.0, 2.0],
        "checks": {
            "clean_baselines": True,
            "spider_read": True,
            "swap_rank_improvements": True,
            "swap_target_top1": True,
            "identity_control": True,
            "matched_random_vector_control": True,
            "wrong_concept_control": True,
            "random_target_control": False,
        },
        "policy": {
            "clean_baselines": {"required_count": 2},
            "spider_read": {
                "maximum_rank": 5,
                "requires_better_than_logit_lens": True,
                "paper_target_rank": 1,
            },
            "swap_rank_improvements": {"required_count": 1, "case_count": 2},
            "swap_target_top1": {
                "required_count": 1,
                "case_count": 2,
                "paper_primary_alpha": 1.0,
                "paper_target_rank": 1,
            },
        },
        "cases": [
            {
                "key": "spider",
                "baseline": {"top1_token": " 8", "expected_top1": True},
                "summary": {
                    "jacobian_lens": {"best_rank": 3, "layer": 2, "position": 4},
                    "logit_lens": {"best_rank": 11, "layer": 3, "position": 4},
                },
            },
            {
                "key": "france_capital",
                "baseline": {"top1_token": " Paris", "expected_top1": True},
                "summary": {
                    "jacobian_lens": {"best_rank": 2, "layer": 2, "position": 5},
                    "logit_lens": {"best_rank": 7, "layer": 3, "position": 5},
                },
            },
        ],
        "swaps": [
            {
                "key": "spider",
                "clean": {"target_rank": 40},
                "interventions": {
                    "2.0": {"target_rank": 1},
                    "1.0": {"target_rank": 4},
                },
                "best_intervened_rank": 1,
                "improved": True,
                "target_top1": True,
                "workspace_loading": None,
            },
            {
                "key": "france_capital",
                "clean": {"target_rank": 80},
                "interventions": {
                    "2.0": {"target_rank": 20},
                    "1.0": {"target_rank": 10},
                },
                "best_intervened_rank": 10,
                "improved": True,
                "target_top1": False,
                "workspace_loading": 0.625,
            },
        ],
        "controls": {
            "tolerances": {
                "identity_logits": {"atol": 1e-6, "rtol": 1e-5},
            },
            "identity": {
                "passed_case_count": 2,
                "required_case_count": 2,
                "maximum_absolute_logit_difference": 1e-7,
                "passed": True,
            },
            "matched_random_vector": {
                "real_mean_log_rank_gain": 1.25,
                "percentile_95_threshold": 0.75,
                "passed": True,
            },
            "wrong_concept": {
                "matched_mean_log_rank_gain": 1.0,
                "mismatched_mean_log_rank_gain": 0.4,
                "matched_winning_case_count": 2,
                "required_winning_case_count": 2,
                "passed": True,
            },
            "random_target": {
                "real_mean_log_rank_gain": 0.5,
                "percentile_95_threshold": 0.6,
                "passed": False,
            },
        },
    }


def test_capability_rows_explain_sanity_and_paper_gaps() -> None:
    rows = {row.check: row for row in capability_rows(sample_result())}

    assert rows["clean_baselines"].status == "PASS"
    assert rows["clean_baselines"].paper_target == "N/A - no paper threshold"
    assert rows["spider_read"].observed == "J rank 3; logit rank 11"
    assert rows["spider_read"].sanity_threshold == (
        "J rank <= 5 and J rank < logit rank"
    )
    assert rows["spider_read"].sanity_margin == ("rank headroom +2; logit advantage +8")
    assert rows["spider_read"].paper_target == "J rank 1"
    assert rows["spider_read"].paper_gap == "+2 ranks"
    assert rows["swap_rank_improvements"].paper_gap == ("N/A - no paper threshold")
    assert rows["swap_target_top1"].status == "PASS"


def test_swap_rows_use_primary_alpha_for_the_paper_gap() -> None:
    rows = {row[0]: row for row in swap_rows(sample_result())}

    assert rows["spider"] == (
        "spider",
        "40",
        "4",
        "1",
        "1",
        "yes",
        "yes",
        "+3 ranks",
    )
    assert rows["france_capital"][-1] == "+9 ranks"


def test_control_rows_show_sanity_margins_without_inventing_paper_targets() -> None:
    rows = {row.check: row for row in control_rows(sample_result())}

    assert rows["identity_control"].status == "PASS"
    assert rows["matched_random_vector_control"].sanity_margin == "+0.5"
    assert rows["wrong_concept_control"].sanity_margin == (
        "mean advantage +0.6; win margin +0 cases"
    )
    assert rows["random_target_control"].status == "FAIL"
    assert rows["random_target_control"].sanity_margin == "-0.1"
    assert all(row.paper_target == "N/A - no paper threshold" for row in rows.values())


def test_complete_report_has_stable_section_order_and_failure_details() -> None:
    report = render_sanity_report(sample_result())

    assert report.index("OVERALL") < report.index("CAPABILITY CHECKS")
    assert report.index("CAPABILITY CHECKS") < report.index("READOUT DETAILS")
    assert report.index("READOUT DETAILS") < report.index("SWAP DETAILS")
    assert report.index("SWAP DETAILS") < report.index("NEGATIVE CONTROLS")
    assert "Overall status: FAIL" in report
    assert "project_commit=abc123" in report
    assert "spider_read" in report
    assert "+2 ranks" in report
    assert "N/A - no paper threshold" in report
    assert "france_capital" in report
    assert "0.625" in report
    assert "FAILURES\n- random target control failed" in report


def test_complete_report_omits_empty_failure_section() -> None:
    result = sample_result()
    result["passed"] = True
    result["failures"] = []

    report = render_sanity_report(result)

    assert "Overall status: PASS" in report
    assert "FAILURES" not in report


def test_complete_report_rejects_missing_required_data() -> None:
    result = deepcopy(sample_result())
    del result["cases"][0]["summary"]["jacobian_lens"]["best_rank"]

    with pytest.raises(KeyError, match="best_rank"):
        render_sanity_report(result)


def test_sample_result_is_deliberately_mixed() -> None:
    result = sample_result()

    assert result["passed"] is False
    assert result["checks"]["random_target_control"] is False
    assert result["checks"]["spider_read"] is True


def test_reporting_fixture_uses_numeric_intervention_keys() -> None:
    result = sample_result()
    interventions = result["swaps"][0]["interventions"]

    assert set(map(float, interventions)) == {1.0, 2.0}
    assert interventions["1.0"]["target_rank"] == 4
    assert interventions["2.0"]["target_rank"] == 1


@pytest.mark.parametrize("missing", ["checks", "policy", "cases", "swaps"])
def test_sample_result_contains_required_capability_sections(missing: str) -> None:
    result = sample_result()

    assert missing in result
