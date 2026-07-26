# Paper-Gap Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore diagnostic paper-rank gaps for the spider readout and each intervention's paper-primary alpha.

**Architecture:** Keep paper targets in the existing `ExperimentResult.policy` and calculate display-only gaps inside the reporting module. Extend the capability and intervention table rows without changing experiment checks or PASS/FAIL behavior.

**Tech Stack:** Python 3.11+, dataclasses, pytest, Ruff

---

### Task 1: Report configured paper-rank gaps

**Files:**
- Modify: `tests/experiments/jlens_readout_sanity/test_reporting.py`
- Modify: `experiments/jlens_readout_sanity/reporting.py`

- [x] **Step 1: Add policy data to the result fixture**

In `sample_result()`, replace the empty policy argument with the policy fields
already emitted by `run_experiment()`:

```python
        {
            "spider_read": {"paper_target_rank": 1},
            "swap_target_top1": {
                "paper_primary_alpha": 1.0,
                "paper_target_rank": 1,
            },
        },
```

- [x] **Step 2: Write failing tests for both paper comparisons**

Import `capability_rows`, then add:

```python
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
```

- [x] **Step 3: Run the focused tests and verify they fail**

Run:

```bash
uv run pytest -q \
  tests/experiments/jlens_readout_sanity/test_reporting.py::test_rows_report_spider_and_primary_intervention_paper_gaps \
  tests/experiments/jlens_readout_sanity/test_reporting.py::test_intervention_paper_gap_reports_an_exact_match
```

Expected: both tests fail because the current rows contain no paper-gap cells.

- [x] **Step 4: Add the minimal reporting implementation**

In `reporting.py`, add:

```python
NO_PAPER_COMPARISON = "N/A"


def _rank_gap(rank: int, target_rank: int) -> str:
    """Format the signed distance from a directly comparable paper rank."""
    gap = rank - target_rank
    return "0 ranks" if gap == 0 else f"{gap:+d} ranks"
```

In `capability_rows()`, read the spider paper target from
`result.policy["spider_read"]["paper_target_rank"]`. Append two cells to every
row: `("J rank 1", _rank_gap(...))` for `spider_read`, and
`(NO_PAPER_COMPARISON, NO_PAPER_COMPARISON)` otherwise.

In `intervention_rows()`, read:

```python
swap_policy = result.policy["swap_target_top1"]
primary_alpha = float(swap_policy["paper_primary_alpha"])
paper_target_rank = int(swap_policy["paper_target_rank"])
```

For each intervention, select the unique condition whose alpha equals
`primary_alpha`; raise
`ValueError(f"Expected one intervention condition for alpha={primary_alpha:g}")`
unless exactly one exists. Append
`_rank_gap(primary.evaluation.target_rank, paper_target_rank)` to its row.

Update `render_sanity_report()` headers:

```python
("check", "observed", "status", "paper target", "paper gap")
```

and:

```python
(
    "case",
    "clean rank",
    "conditions",
    "best rank",
    "improved",
    "top-1",
    "alpha=1 paper gap",
)
```

- [x] **Step 5: Run the focused reporting tests**

Run:

```bash
uv run pytest -q tests/experiments/jlens_readout_sanity/test_reporting.py
```

Expected: all reporting tests pass.

- [x] **Step 6: Run formatting and full verification**

Run:

```bash
uv run ruff check experiments/jlens_readout_sanity/reporting.py \
  tests/experiments/jlens_readout_sanity/test_reporting.py
uv run pytest -q
```

Expected: Ruff passes and all tests pass.

- [x] **Step 7: Commit only the reporting change**

```bash
git add experiments/jlens_readout_sanity/reporting.py \
  tests/experiments/jlens_readout_sanity/test_reporting.py \
  docs/superpowers/plans/2026-07-26-paper-gap-reporting.md
git commit -m "fix: restore paper-gap reporting"
```
