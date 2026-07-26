# Text-Only Sanity Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove generated HTML visualizations and replace them with deterministic notebook text tables that show every check's observed value, Qwen sanity threshold and margin, binary status, and paper comparison where one exists.

**Architecture:** Preserve the runner's existing pass/fail booleans as the sole status authority, add the exact run policy to `result.json`, and introduce a pure experiment-local reporting module that formats those results without inference or gate recomputation. The notebook becomes a thin caller of that module and continues to save results before raising on failure.

**Tech Stack:** Python 3.11, dataclasses, nbformat notebook JSON, pytest, Ruff

---

## File Structure

- Modify `experiments/jlens_readout_sanity/constants.py`: name the existing one-swap top-1 requirement.
- Modify `experiments/jlens_readout_sanity/runner.py`: serialize the exact capability policy used by the run.
- Create `experiments/jlens_readout_sanity/reporting.py`: own report rows, margins, paper gaps, table formatting, and complete text rendering.
- Modify `experiments/jlens_readout_sanity/utils.py`: expose only the notebook-facing report entry point.
- Create `tests/experiments/jlens_readout_sanity/test_reporting.py`: test report semantics and formatting with plain dictionaries.
- Modify `tests/experiments/jlens_readout_sanity/test_constants.py`: pin the named top-1 count policy.
- Modify `tests/experiments/jlens_readout_sanity/test_runner.py`: verify serialized policy metadata.
- Modify `tests/experiments/jlens_readout_sanity/test_package.py`: verify the reporting facade boundary.
- Modify `tests/test_notebooks.py`: require text reporting and forbid HTML visualization code.
- Modify `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`: replace raw reporting plus the visualization cell with one text-report call.
- Modify `README.md`: list `result.json` as the only artifact and explain sanity thresholds versus paper gaps while preserving the pre-existing evaluation-policy edit.

### Task 1: Serialize the Exact Capability Policy

**Files:**
- Modify: `experiments/jlens_readout_sanity/constants.py`
- Modify: `experiments/jlens_readout_sanity/runner.py`
- Test: `tests/experiments/jlens_readout_sanity/test_constants.py`
- Test: `tests/experiments/jlens_readout_sanity/test_runner.py`

- [ ] **Step 1: Write the failing constant and result-schema tests**

Add `SWAP_TARGET_TOP1_REQUIRED_COUNT` to the constants import in
`tests/experiments/jlens_readout_sanity/test_constants.py` and assert:

```python
assert SWAP_TARGET_TOP1_REQUIRED_COUNT == 1
```

In the complete-run test in
`tests/experiments/jlens_readout_sanity/test_runner.py`, assert the returned
policy is JSON-compatible and records the values that actually governed the
run:

```python
assert result["policy"] == {
    "clean_baselines": {"required_count": 5},
    "spider_read": {
        "maximum_rank": 5,
        "requires_better_than_logit_lens": True,
        "paper_target_rank": 1,
    },
    "swap_rank_improvements": {
        "required_count": 3,
        "case_count": 5,
    },
    "swap_target_top1": {
        "required_count": 1,
        "case_count": 5,
        "paper_primary_alpha": 1.0,
        "paper_target_rank": 1,
    },
}
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_constants.py \
  tests/experiments/jlens_readout_sanity/test_runner.py -q
```

Expected: FAIL because `SWAP_TARGET_TOP1_REQUIRED_COUNT` and
`result["policy"]` do not exist.

- [ ] **Step 3: Add the named policy constant and use it in the gate**

Add to `experiments/jlens_readout_sanity/constants.py`:

```python
SWAP_TARGET_TOP1_REQUIRED_COUNT = 1
```

Import it in `runner.py` and replace the literal aggregate top-1 condition:

```python
"swap_target_top1": top1_count >= SWAP_TARGET_TOP1_REQUIRED_COUNT,
```

- [ ] **Step 4: Attach policy metadata to the assembled result**

Immediately before the existing `"cases"` field in `run_readout_sanity`, add:

```python
"policy": {
    "clean_baselines": {"required_count": len(read_results)},
    "spider_read": {
        "maximum_rank": SPIDER_READ_MAX_RANK,
        "requires_better_than_logit_lens": True,
        "paper_target_rank": 1,
    },
    "swap_rank_improvements": {
        "required_count": minimum_improvements,
        "case_count": len(swap_results),
    },
    "swap_target_top1": {
        "required_count": SWAP_TARGET_TOP1_REQUIRED_COUNT,
        "case_count": len(swap_results),
        "paper_primary_alpha": CONTROL_ALPHA,
        "paper_target_rank": 1,
    },
},
```

Do not derive any existing check boolean from this metadata after assembly.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run the Step 2 command again.

Expected: PASS.

- [ ] **Step 6: Commit the policy metadata**

```bash
git add \
  experiments/jlens_readout_sanity/constants.py \
  experiments/jlens_readout_sanity/runner.py \
  tests/experiments/jlens_readout_sanity/test_constants.py \
  tests/experiments/jlens_readout_sanity/test_runner.py
git commit -m "feat: record sanity reporting policy"
```

### Task 2: Build Pure Text Report Rows

**Files:**
- Create: `experiments/jlens_readout_sanity/reporting.py`
- Create: `tests/experiments/jlens_readout_sanity/test_reporting.py`
- Modify: `experiments/jlens_readout_sanity/utils.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_package.py`

- [ ] **Step 1: Write a reusable result fixture for reporting tests**

Create `tests/experiments/jlens_readout_sanity/test_reporting.py` with a
`sample_result()` function containing all four aggregate checks, one spider
case, one France case, two swaps, all four control summaries, policy metadata,
and provenance. Use these decisive values:

```python
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
```

Set the spider Jacobian rank to 3 and logit-lens rank to 11. Give the spider
swap clean/alpha-1/alpha-2 ranks 40/4/1 and the France swap ranks 80/10/20.
Set matched-random real/threshold values to 1.25/0.75, wrong-concept matched
and mismatched means to 1.0/0.4 with 2 required wins and 2 observed wins, and
random-target real/threshold values to 0.5/0.6.

- [ ] **Step 2: Write failing capability and paper-gap tests**

Import `capability_rows` and assert:

```python
rows = {row.check: row for row in capability_rows(sample_result())}

assert rows["clean_baselines"].status == "PASS"
assert rows["clean_baselines"].paper_target == "N/A - no paper threshold"
assert rows["spider_read"].observed == "J rank 3; logit rank 11"
assert rows["spider_read"].sanity_threshold == "J rank <= 5 and J rank < logit rank"
assert rows["spider_read"].sanity_margin == (
    "rank headroom +2; logit advantage +8"
)
assert rows["spider_read"].paper_target == "J rank 1"
assert rows["spider_read"].paper_gap == "+2 ranks"
assert rows["swap_rank_improvements"].paper_gap == "N/A - no paper threshold"
assert rows["swap_target_top1"].status == "PASS"
```

- [ ] **Step 3: Run the capability test and verify RED**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_reporting.py::test_capability_rows_explain_sanity_and_paper_gaps -q
```

Expected: FAIL because the reporting module does not exist.

- [ ] **Step 4: Implement the row contract and capability transformation**

Create `experiments/jlens_readout_sanity/reporting.py` with:

```python
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
    alphas = ", ".join(f"{float(alpha):g}" for alpha in result["intervention_strengths"])
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
```

- [ ] **Step 5: Run the capability test and verify GREEN**

Run the Step 3 command again.

Expected: PASS.

- [ ] **Step 6: Write failing per-swap and control-row tests**

Add assertions that `swap_rows()` uses the intervention whose numeric key is
equal to `paper_primary_alpha`, not dictionary order, and that it reports the
alpha-1 gap:

```python
rows = {row[0]: row for row in swap_rows(sample_result())}
assert rows["spider"] == ("spider", "40", "4", "1", "1", "yes", "yes", "+3 ranks")
assert rows["france_capital"][-1] == "+9 ranks"
```

Add control assertions:

```python
rows = {row.check: row for row in control_rows(sample_result())}
assert rows["identity_control"].status == "PASS"
assert rows["matched_random_vector_control"].sanity_margin == "+0.5"
assert rows["wrong_concept_control"].sanity_margin == (
    "mean advantage +0.6; win margin +0 cases"
)
assert rows["random_target_control"].status == "FAIL"
assert rows["random_target_control"].sanity_margin == "-0.1"
assert all(row.paper_target == "N/A - no paper threshold" for row in rows.values())
```

- [ ] **Step 7: Run the new tests and verify RED**

Run:

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_reporting.py -q
```

Expected: FAIL because `swap_rows` and `control_rows` do not exist.

- [ ] **Step 8: Implement swap and control transformations**

Add pure helpers to `reporting.py`:

```python
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
    swap_policy = result["policy"]["swap_target_top1"]
    alpha = float(swap_policy["paper_primary_alpha"])
    paper_target_rank = int(swap_policy["paper_target_rank"])
    other_alphas = [
        float(item) for item in result["intervention_strengths"] if float(item) != alpha
    ]
    if len(other_alphas) != 1:
        raise ValueError("Text report requires exactly one non-primary alpha")
    rescue_alpha = other_alphas[0]
    rows = []
    for swap in result["swaps"]:
        primary_rank = int(_intervention_at(swap, alpha)["target_rank"])
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
```

Implement `control_rows(result)` with one `ReportRow` per top-level control
check. Copy statuses from `result["checks"]` exactly:

```python
def control_rows(result: dict[str, Any]) -> tuple[ReportRow, ...]:
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
```

- [ ] **Step 9: Add deterministic table and complete-report tests**

Test `_format_table` through `render_sanity_report()` rather than testing
private padding mechanics. Assert the exact section order and stable text:

```python
report = render_sanity_report(sample_result())
assert report.index("OVERALL") < report.index("CAPABILITY CHECKS")
assert report.index("CAPABILITY CHECKS") < report.index("READOUT DETAILS")
assert report.index("READOUT DETAILS") < report.index("SWAP DETAILS")
assert report.index("SWAP DETAILS") < report.index("NEGATIVE CONTROLS")
assert "Overall status: FAIL" in report
assert "spider_read" in report
assert "+2 ranks" in report
assert "N/A - no paper threshold" in report
assert "FAILURES" in report
```

Also delete one required nested field from the fixture and assert
`render_sanity_report` raises `KeyError`; the renderer must not fabricate a
default value.

- [ ] **Step 10: Implement table formatting and the complete report**

Add deterministic table formatting, readout-detail rows, and the complete
report:

```python
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
    return "\n".join((format_row(headers), format_row(separator), *(format_row(row) for row in rows)))


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
        + _format_table(report_headers, _report_row_cells(capability_rows(result))),
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
        + _format_table(report_headers, _report_row_cells(control_rows(result))),
    ]
    if result["failures"]:
        sections.append(
            "FAILURES\n"
            + "\n".join(f"- {failure}" for failure in result["failures"])
        )
    return "\n\n".join(sections)
```

Do not use ANSI color so saved Colab output and copied logs remain stable.

- [ ] **Step 11: Expose the renderer through the notebook facade**

In `utils.py`, import `render_sanity_report`, add it to `__all__`, and extend
`test_package.py`:

```python
assert utils.render_sanity_report.__module__ == (
    "experiments.jlens_readout_sanity.reporting"
)
```

- [ ] **Step 12: Run reporting and package tests and verify GREEN**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_reporting.py \
  tests/experiments/jlens_readout_sanity/test_package.py -q
uv run ruff check \
  experiments/jlens_readout_sanity/reporting.py \
  experiments/jlens_readout_sanity/utils.py \
  tests/experiments/jlens_readout_sanity/test_reporting.py \
  tests/experiments/jlens_readout_sanity/test_package.py
```

Expected: all tests and lint checks PASS.

- [ ] **Step 13: Commit the reporting module**

```bash
git add \
  experiments/jlens_readout_sanity/reporting.py \
  experiments/jlens_readout_sanity/utils.py \
  tests/experiments/jlens_readout_sanity/test_reporting.py \
  tests/experiments/jlens_readout_sanity/test_package.py
git commit -m "feat: add text sanity report"
```

### Task 3: Replace Notebook HTML with the Text Report

**Files:**
- Modify: `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`
- Modify: `tests/test_notebooks.py`

- [ ] **Step 1: Replace the positive visualization assertions with failing removal assertions**

In `test_readout_sanity_notebook_has_pinned_gpu_workflow`, remove the positive
assertions for `compute_slice` and `mode="embed"`. Add:

```python
assert "render_sanity_report" in source
assert "jlens.vis" not in source
assert "compute_slice" not in source
assert "build_page" not in source
assert "notebook_iframe" not in source
assert ".html" not in source
assert "write_text(page" not in source
```

In `test_readout_execution_saving_and_reporting_are_separate_cells`, assert:

```python
assert "print(render_sanity_report(result))" in report_source
assert "for case in" not in report_source
assert "for swap in" not in report_source
assert "render-slices" not in cells_by_id
```

- [ ] **Step 2: Run notebook tests and verify RED**

Run:

```bash
uv run pytest tests/test_notebooks.py -q
```

Expected: FAIL because the visualization cell and imports still exist.

- [ ] **Step 3: Update the notebook facade import**

In the `load-model-and-lens` cell, add `render_sanity_report` to the existing
import from `experiments.jlens_readout_sanity.utils`.

- [ ] **Step 4: Replace both old output cells**

Set the `report-results` cell source to exactly:

```python
print(render_sanity_report(result))
```

Delete the complete `render-slices` cell. Keep the cell order otherwise
unchanged, especially `save-result` before `report-results` and `grade-run`
after it.

- [ ] **Step 5: Validate notebook JSON and verify GREEN**

Run:

```bash
uv run python -m json.tool \
  experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb >/dev/null
uv run pytest tests/test_notebooks.py -q
```

Expected: valid JSON and all notebook tests PASS.

- [ ] **Step 6: Commit the notebook change**

```bash
git add \
  experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb \
  tests/test_notebooks.py
git commit -m "refactor: replace sanity HTML with text report"
```

### Task 4: Update the User-Facing Artifact Contract

**Files:**
- Modify: `README.md`
- Test: `tests/test_notebooks.py`

- [ ] **Step 1: Write a failing README artifact test**

Add to `tests/test_notebooks.py`:

```python
def test_readout_sanity_documents_text_only_result_artifact() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "runs/jlens-readout-sanity/\n└── result.json" in readme
    assert "spider.html" not in readme
    assert "france_capital.html" not in readme
    assert "Qwen sanity threshold" in readme
    assert "paper gap" in readme
```

- [ ] **Step 2: Run the README test and verify RED**

Run:

```bash
uv run pytest \
  tests/test_notebooks.py::test_readout_sanity_documents_text_only_result_artifact -q
```

Expected: FAIL because README still lists the HTML files and does not explain
the comparison columns.

- [ ] **Step 3: Update README without overwriting the existing local edit**

Change only the read-and-change sanity section's artifact tree to:

```text
runs/jlens-readout-sanity/
└── result.json
```

Add this explanation after the experiment summary:

```text
The notebook prints a text-only report for every configured check. PASS or FAIL
always reflects the Qwen sanity threshold used by the run. Where the paper
provides a directly comparable target, the report also shows the paper gap as
diagnostic context; that gap does not change pass/fail.
```

Preserve the already-modified LLM answer-evaluation paragraph byte-for-byte.

- [ ] **Step 4: Run the documentation and notebook tests and verify GREEN**

Run:

```bash
uv run pytest tests/test_notebooks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit only the intended README and test changes**

Review `git diff -- README.md` and verify it contains both the pre-existing
evaluation-policy edit and the new artifact/reporting edit. Because the README
already had an uncommitted user change before this task, ask the user before
including that unrelated hunk in a commit; otherwise leave README unstaged and
report it as an uncommitted documentation change.

Stage and commit `tests/test_notebooks.py` only if README authorization is not
available:

```bash
git add tests/test_notebooks.py
git commit -m "test: require text-only sanity artifacts"
```

If the user authorizes the full README file, stage both files and use:

```bash
git add README.md tests/test_notebooks.py
git commit -m "docs: explain text-only sanity report"
```

### Task 5: Verify the Complete Change

**Files:**
- Verify all files changed by Tasks 1-4

- [ ] **Step 1: Confirm no active code or README creates or advertises HTML**

Run:

```bash
rg -n "jlens\.vis|compute_slice|build_page|notebook_iframe|spider\.html|france_capital\.html|write_text\(page" \
  experiments tests README.md
```

Expected: no matches. Historical design and plan documents are intentionally
excluded because they describe the state at the time they were written.

- [ ] **Step 2: Run focused experiment tests**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity \
  tests/test_notebooks.py -q
```

Expected: PASS.

- [ ] **Step 3: Run the full suite**

Run:

```bash
uv run pytest
```

Expected: PASS with no unexpected warnings or errors.

- [ ] **Step 4: Run formatting and lint checks**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
git diff --check
```

Expected: all checks PASS.

- [ ] **Step 5: Review the final branch state**

Run:

```bash
git status --short --branch
git log --oneline -6
```

Expected: all implementation commits are on `codex/jlens-readout-sanity`;
only explicitly preserved user-owned changes may remain unstaged.
