"""Deterministic text reporting for the J-Lens sanity experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

NO_PAPER_THRESHOLD = "N/A - no paper threshold"


@dataclass(frozen=True, slots=True)
class ReportRow:
    check: str
    observed: str
    sanity_threshold: str
    sanity_margin: str
    status: str
    paper_target: str
    paper_gap: str


def _status(value: Any) -> str:
    return "PASS" if bool(value) else "FAIL"


def _signed(value: int | float) -> str:
    return f"{value:+g}"


def _rank_gap(rank: int, target_rank: int) -> str:
    gap = rank - target_rank
    return "0 ranks" if gap == 0 else f"+{gap} ranks"


def capability_rows(result: dict[str, Any]) -> tuple[ReportRow, ...]:
    """Build display rows for the four configured capability checks."""
    checks = result["checks"]
    policy = result["policy"]
    cases = result["cases"]
    swaps = result["swaps"]
    spider = next(case for case in cases if case["key"] == "spider")
    jacobian_rank = int(spider["summary"]["jacobian_lens"]["best_rank"])
    logit_rank = int(spider["summary"]["logit_lens"]["best_rank"])
    clean_count = sum(bool(case["baseline"]["expected_top1"]) for case in cases)
    improved_count = sum(bool(swap["improved"]) for swap in swaps)
    top1_count = sum(bool(swap["target_top1"]) for swap in swaps)
    clean_required = int(policy["clean_baselines"]["required_count"])
    spider_max = int(policy["spider_read"]["maximum_rank"])
    spider_paper_rank = int(policy["spider_read"]["paper_target_rank"])
    improved_required = int(policy["swap_rank_improvements"]["required_count"])
    top1_required = int(policy["swap_target_top1"]["required_count"])
    case_count = len(swaps)
    alphas = ", ".join(
        f"{float(alpha):g}" for alpha in result["intervention_strengths"]
    )
    return (
        ReportRow(
            "clean_baselines",
            f"{clean_count}/{len(cases)} expected answers at top-1",
            f"{clean_required}/{len(cases)}",
            f"{_signed(clean_count - clean_required)} cases",
            _status(checks["clean_baselines"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
        ReportRow(
            "spider_read",
            f"J rank {jacobian_rank}; logit rank {logit_rank}",
            f"J rank <= {spider_max} and J rank < logit rank",
            (
                f"rank headroom {_signed(spider_max - jacobian_rank)}; "
                f"logit advantage {_signed(logit_rank - jacobian_rank)}"
            ),
            _status(checks["spider_read"]),
            f"J rank {spider_paper_rank}",
            _rank_gap(jacobian_rank, spider_paper_rank),
        ),
        ReportRow(
            "swap_rank_improvements",
            f"{improved_count}/{case_count} swaps improved",
            f">= {improved_required}/{case_count}",
            f"{_signed(improved_count - improved_required)} cases",
            _status(checks["swap_rank_improvements"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
        ReportRow(
            "swap_target_top1",
            f"{top1_count}/{case_count} targets top-1 across alpha in {{{alphas}}}",
            f">= {top1_required}/{case_count}",
            f"{_signed(top1_count - top1_required)} cases",
            _status(checks["swap_target_top1"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
    )


def _intervention_at(swap: dict[str, Any], alpha: float) -> dict[str, Any]:
    matches = [
        payload
        for key, payload in swap["interventions"].items()
        if float(key) == float(alpha)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one intervention for alpha={alpha:g}")
    return matches[0]


def swap_rows(result: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Build per-case rank rows using alpha=1 for paper comparison."""
    swap_policy = result["policy"]["swap_target_top1"]
    primary_alpha = float(swap_policy["paper_primary_alpha"])
    paper_target_rank = int(swap_policy["paper_target_rank"])
    other_alphas = [
        float(alpha)
        for alpha in result["intervention_strengths"]
        if float(alpha) != primary_alpha
    ]
    if len(other_alphas) != 1:
        raise ValueError("Text report requires exactly one non-primary alpha")
    rescue_alpha = other_alphas[0]
    rows = []
    for swap in result["swaps"]:
        primary_rank = int(_intervention_at(swap, primary_alpha)["target_rank"])
        rescue_rank = int(_intervention_at(swap, rescue_alpha)["target_rank"])
        rows.append(
            (
                str(swap["key"]),
                str(int(swap["clean"]["target_rank"])),
                str(primary_rank),
                str(rescue_rank),
                str(int(swap["best_intervened_rank"])),
                "yes" if swap["improved"] else "no",
                "yes" if swap["target_top1"] else "no",
                _rank_gap(primary_rank, paper_target_rank),
            )
        )
    return tuple(rows)


def control_rows(result: dict[str, Any]) -> tuple[ReportRow, ...]:
    """Build display rows for the four configured negative-control checks."""
    checks = result["checks"]
    controls = result["controls"]
    identity = controls["identity"]
    matched_random = controls["matched_random_vector"]
    wrong = controls["wrong_concept"]
    random_target = controls["random_target"]
    identity_observed = int(identity["passed_case_count"])
    identity_required = int(identity["required_case_count"])
    identity_tolerances = controls["tolerances"]["identity_logits"]
    matched_real = float(matched_random["real_mean_log_rank_gain"])
    matched_threshold = float(matched_random["percentile_95_threshold"])
    wrong_matched = float(wrong["matched_mean_log_rank_gain"])
    wrong_mismatched = float(wrong["mismatched_mean_log_rank_gain"])
    wrong_wins = int(wrong["matched_winning_case_count"])
    wrong_required = int(wrong["required_winning_case_count"])
    target_real = float(random_target["real_mean_log_rank_gain"])
    target_threshold = float(random_target["percentile_95_threshold"])
    return (
        ReportRow(
            "identity_control",
            (
                f"{identity_observed}/{identity_required} cases; max |delta logit| "
                f"{float(identity['maximum_absolute_logit_difference']):.6g}"
            ),
            (
                f"{identity_required}/{identity_required} and logits close "
                f"(atol={identity_tolerances['atol']:.6g}, "
                f"rtol={identity_tolerances['rtol']:.6g})"
            ),
            f"{_signed(identity_observed - identity_required)} cases",
            _status(checks["identity_control"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
        ReportRow(
            "matched_random_vector_control",
            f"real mean {matched_real:.6g}; p95 {matched_threshold:.6g}",
            "real mean > p95 matched-random mean",
            _signed(matched_real - matched_threshold),
            _status(checks["matched_random_vector_control"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
        ReportRow(
            "wrong_concept_control",
            (
                f"matched mean {wrong_matched:.6g}; mismatched mean "
                f"{wrong_mismatched:.6g}; wins {wrong_wins}"
            ),
            f"matched mean > mismatched mean and wins >= {wrong_required}",
            (
                f"mean advantage {_signed(wrong_matched - wrong_mismatched)}; "
                f"win margin {_signed(wrong_wins - wrong_required)} cases"
            ),
            _status(checks["wrong_concept_control"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
        ReportRow(
            "random_target_control",
            f"real mean {target_real:.6g}; p95 {target_threshold:.6g}",
            "real mean > p95 random-target mean",
            _signed(target_real - target_threshold),
            _status(checks["random_target_control"]),
            NO_PAPER_THRESHOLD,
            NO_PAPER_THRESHOLD,
        ),
    )


def _format_table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> str:
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
        (
            format_row(headers),
            format_row(separator),
            *(format_row(row) for row in rows),
        )
    )


def _report_row_cells(rows: tuple[ReportRow, ...]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        (
            row.check,
            row.observed,
            row.sanity_threshold,
            row.sanity_margin,
            row.status,
            row.paper_target,
            row.paper_gap,
        )
        for row in rows
    )


def readout_rows(result: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Build per-case readout detail rows."""
    swaps = {swap["key"]: swap for swap in result["swaps"]}
    rows = []
    for case in result["cases"]:
        jacobian = case["summary"]["jacobian_lens"]
        logit = case["summary"]["logit_lens"]
        loading = swaps[case["key"]]["workspace_loading"]
        rows.append(
            (
                str(case["key"]),
                repr(case["baseline"]["top1_token"]),
                "yes" if case["baseline"]["expected_top1"] else "no",
                str(int(jacobian["best_rank"])),
                str(int(jacobian["layer"])),
                str(int(jacobian["position"])),
                str(int(logit["best_rank"])),
                "N/A" if loading is None else f"{float(loading):.6g}",
            )
        )
    return tuple(rows)


def render_sanity_report(result: dict[str, Any]) -> str:
    """Render the complete sanity result as stable, copyable plain text."""
    report_headers = (
        "check",
        "observed",
        "sanity threshold",
        "sanity margin",
        "status",
        "paper target",
        "paper gap",
    )
    provenance = result["provenance"]
    provenance_text = ", ".join(
        f"{name}={value}" for name, value in sorted(provenance.items())
    )
    sections = [
        "OVERALL\n"
        f"Overall status: {_status(result['passed'])}\n"
        f"Provenance: {provenance_text}",
        "CAPABILITY CHECKS\n"
        + _format_table(
            report_headers,
            _report_row_cells(capability_rows(result)),
        ),
        "READOUT DETAILS\n"
        + _format_table(
            (
                "case",
                "baseline token",
                "expected top-1",
                "J rank",
                "J layer",
                "J position",
                "logit rank",
                "workspace loading",
            ),
            readout_rows(result),
        ),
        "SWAP DETAILS\n"
        + _format_table(
            (
                "case",
                "clean rank",
                "alpha=1 rank",
                "alpha=2 rank",
                "best rank",
                "improved",
                "target top-1",
                "alpha=1 paper gap",
            ),
            swap_rows(result),
        ),
        "NEGATIVE CONTROLS\n"
        + _format_table(
            report_headers,
            _report_row_cells(control_rows(result)),
        ),
    ]
    if result["failures"]:
        sections.append(
            "FAILURES\n" + "\n".join(f"- {failure}" for failure in result["failures"])
        )
    return "\n\n".join(sections)
