"""Compact text reporting for typed J-Lens sanity results."""

from __future__ import annotations

from experiments.jlens_readout_sanity.experiment import ExperimentResult


def _status(value: object) -> str:
    return "PASS" if bool(value) else "FAIL"


def _format_table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
    """Format equally sized string rows as a plain-text table."""
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("Every report row must match the header width")
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]

    def format_row(row: tuple[str, ...]) -> str:
        return "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))

    separator = tuple("-" * width for width in widths)
    return "\n".join(
        (format_row(headers), format_row(separator), *(format_row(row) for row in rows))
    )


def capability_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build compact rows for configured capability checks."""
    clean_count = sum(case.baseline.passed for case in result.cases)
    interventions = [case.intervention for case in result.cases if case.intervention]
    improved_count = sum(
        any(condition.comparison.improved for condition in intervention.conditions)
        for intervention in interventions
    )
    top1_count = sum(
        any(condition.comparison.reached_top1 for condition in intervention.conditions)
        for intervention in interventions
    )
    gated = next(
        (
            case.readout
            for case in result.cases
            if case.case.readout is not None
            and case.case.readout.require_capability_gate
        ),
        None,
    )
    readout_observed = (
        "not configured"
        if gated is None
        else (
            f"J rank {gated.jacobian_lens.best_rank}; "
            f"logit rank {gated.logit_lens.best_rank}"
        )
    )
    observations = {
        "clean_baselines": f"{clean_count}/{len(result.cases)} generated answers passed",
        "spider_read": readout_observed,
        "swap_rank_improvements": (
            f"{improved_count}/{len(interventions)} interventions improved"
        ),
        "swap_target_top1": f"{top1_count}/{len(interventions)} targets reached top-1",
    }
    return tuple(
        (name, observations.get(name, "see negative controls"), _status(passed))
        for name, passed in result.checks.items()
    )


def readout_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build per-case clean-answer and readout summary rows."""
    rows = []
    for case in result.cases:
        readout = case.readout
        if readout is None:
            continue
        rows.append(
            (
                case.case.key,
                case.baseline.extracted_answer or "N/A",
                case.baseline.answer_status.value,
                str(readout.jacobian_lens.best_rank),
                str(readout.jacobian_lens.layer),
                str(readout.jacobian_lens.position),
                str(readout.logit_lens.best_rank),
                (
                    "N/A"
                    if readout.workspace_loading is None
                    else f"{readout.workspace_loading:.6g}"
                ),
            )
        )
    return tuple(rows)


def intervention_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build per-case clean and intervened target-rank rows."""
    rows = []
    for case in result.cases:
        intervention = case.intervention
        if intervention is None:
            continue
        conditions = "; ".join(
            f"alpha={condition.alpha:g}: {condition.evaluation.target_rank}"
            for condition in intervention.conditions
        )
        best_rank = min(
            condition.evaluation.target_rank for condition in intervention.conditions
        )
        rows.append(
            (
                case.case.key,
                str(intervention.clean_target.target_rank),
                conditions,
                str(best_rank),
                (
                    "yes"
                    if any(
                        condition.comparison.improved
                        for condition in intervention.conditions
                    )
                    else "no"
                ),
                (
                    "yes"
                    if any(
                        condition.comparison.reached_top1
                        for condition in intervention.conditions
                    )
                    else "no"
                ),
            )
        )
    return tuple(rows)


def control_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build aggregate negative-control result rows."""
    controls = result.controls
    identity = controls["identity"]
    matched = controls["matched_random_vector"]
    wrong = controls["wrong_concept"]
    random_target = controls["random_target"]
    return (
        (
            "identity_control",
            f"{identity['passed_case_count']}/{identity['required_case_count']} cases",
            _status(identity["passed"]),
        ),
        (
            "matched_random_vector_control",
            (
                f"real {float(matched['real_mean_log_rank_gain']):.6g}; "
                f"p95 {float(matched['percentile_95_threshold']):.6g}"
            ),
            _status(matched["passed"]),
        ),
        (
            "wrong_concept_control",
            (
                f"matched {float(wrong['matched_mean_log_rank_gain']):.6g}; "
                f"mismatched {float(wrong['mismatched_mean_log_rank_gain']):.6g}; "
                f"wins {wrong['matched_winning_case_count']}"
            ),
            _status(wrong["passed"]),
        ),
        (
            "random_target_control",
            (
                f"real {float(random_target['real_mean_log_rank_gain']):.6g}; "
                f"p95 {float(random_target['percentile_95_threshold']):.6g}"
            ),
            _status(random_target["passed"]),
        ),
    )


def render_sanity_report(result: ExperimentResult) -> str:
    """Render the complete typed sanity result as stable plain text."""
    provenance = ", ".join(
        f"{name}={value}" for name, value in sorted(result.provenance.items())
    )
    sections = [
        "OVERALL\n"
        f"Overall status: {_status(result.passed)}\n"
        f"Provenance: {provenance or 'N/A'}",
        "CAPABILITY CHECKS\n"
        + _format_table(("check", "observed", "status"), capability_rows(result)),
        "READOUT DETAILS\n"
        + _format_table(
            (
                "case",
                "clean answer",
                "answer status",
                "J rank",
                "J layer",
                "J position",
                "logit rank",
                "workspace loading",
            ),
            readout_rows(result),
        ),
        "INTERVENTION DETAILS\n"
        + _format_table(
            ("case", "clean rank", "conditions", "best rank", "improved", "top-1"),
            intervention_rows(result),
        ),
        "NEGATIVE CONTROLS\n"
        + _format_table(("control", "observed", "status"), control_rows(result)),
    ]
    if result.failures:
        sections.append(
            "FAILURES\n" + "\n".join(f"- {item}" for item in result.failures)
        )
    return "\n\n".join(sections)
