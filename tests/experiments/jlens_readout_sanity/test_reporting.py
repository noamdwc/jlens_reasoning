from experiments.jlens_readout_sanity.experiment import (
    Case,
    CaseResult,
    ExperimentResult,
    InterventionCondition,
    InterventionResult,
    InterventionSpec,
    LensLocation,
    ReadoutResult,
    ReadoutSpec,
)
from experiments.jlens_readout_sanity.reporting import (
    capability_rows,
    intervention_rows,
    readout_rows,
    render_sanity_report,
)
from jlens_reasoning.evaluation import (
    ModelOutput,
    NextTokenEvaluation,
    RankComparison,
    evaluate,
)
from jlens_reasoning.experiments_utils.tokens import TokenVariant


def next_token(rank: int) -> NextTokenEvaluation:
    return NextTokenEvaluation(("target",), (5,), 4, "answer", rank, ())


def condition(alpha: float, baseline: int, candidate: int) -> InterventionCondition:
    return InterventionCondition(
        alpha,
        next_token(candidate),
        RankComparison(
            baseline,
            candidate,
            baseline - candidate,
            0.0,
            candidate < baseline,
            candidate == 1,
        ),
    )


def sample_result() -> ExperimentResult:
    case = Case(
        "spider",
        "prompt",
        ("8", "eight"),
        ReadoutSpec(("spider",), require_capability_gate=True),
        InterventionSpec(" spider", " ant", ("6", "six")),
    )
    baseline = evaluate(ModelOutput("8"), case.expected_answers)
    readout = ReadoutResult(
        LensLocation(3, 2, 4),
        LensLocation(11, 3, 4),
        (2,),
        (4,),
        0.625,
        False,
        True,
        {},
    )
    intervention = InterventionResult(
        TokenVariant(2, " spider"),
        TokenVariant(3, " ant"),
        (),
        (2,),
        next_token(40),
        (condition(1.0, 40, 4), condition(2.0, 40, 1)),
    )
    controls = {
        "identity": {
            "passed_case_count": 1,
            "required_case_count": 1,
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
            "matched_winning_case_count": 1,
            "passed": True,
        },
        "random_target": {
            "real_mean_log_rank_gain": 0.5,
            "percentile_95_threshold": 0.6,
            "passed": False,
        },
    }
    return ExperimentResult(
        (CaseResult(case, baseline, readout, intervention),),
        controls,
        {
            "clean_baselines": True,
            "spider_read": True,
            "swap_rank_improvements": True,
            "swap_target_top1": True,
            "identity_control": True,
            "matched_random_vector_control": True,
            "wrong_concept_control": True,
            "random_target_control": False,
        },
        ("random target control failed",),
        {
            "spider_read": {"paper_target_rank": 1},
            "swap_target_top1": {
                "paper_primary_alpha": 1.0,
                "paper_target_rank": 1,
            },
        },
        {},
        {"project_commit": "abc123"},
    )


def test_typed_rows_keep_generated_answers_separate_from_token_ranks() -> None:
    result = sample_result()

    assert readout_rows(result)[0][1:3] == ("8", "correct")
    assert intervention_rows(result)[0][1:4] == (
        "40",
        "alpha=1: 4; alpha=2: 1",
        "1",
    )


def test_rows_report_spider_and_primary_intervention_paper_gaps() -> None:
    result = sample_result()

    spider = next(row for row in capability_rows(result) if row[0] == "spider_read")

    assert spider[-2:] == ("J rank 1", "+2 ranks")
    assert intervention_rows(result)[0][-1] == "+3 ranks"


def test_intervention_paper_gap_reports_an_exact_match() -> None:
    result = sample_result()
    intervention = result.cases[0].intervention
    assert intervention is not None
    intervention.conditions = (
        condition(1.0, 40, 1),
        condition(2.0, 40, 4),
    )

    assert intervention_rows(result)[0][-1] == "0 ranks"


def test_complete_report_has_stable_sections_and_failure_details() -> None:
    report = render_sanity_report(sample_result())

    assert report.index("OVERALL") < report.index("CAPABILITY CHECKS")
    assert report.index("CAPABILITY CHECKS") < report.index("READOUT DETAILS")
    assert report.index("READOUT DETAILS") < report.index("INTERVENTION DETAILS")
    assert report.index("INTERVENTION DETAILS") < report.index("NEGATIVE CONTROLS")
    assert "Overall status: FAIL" in report
    assert "project_commit=abc123" in report
    assert "clean answer" in report
    assert "alpha=1: 4; alpha=2: 1" in report
    assert "FAILURES\n- random target control failed" in report


def test_complete_report_omits_empty_failure_section() -> None:
    result = sample_result()
    result.checks["random_target_control"] = True
    result.failures = ()

    report = render_sanity_report(result)

    assert "Overall status: PASS" in report
    assert "FAILURES" not in report
