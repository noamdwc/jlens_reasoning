# Control Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the 617-line negative-control module into pure analysis, model-backed execution, and a short stable orchestration facade without changing experiment behavior or serialized results.

**Architecture:** Move deterministic metadata, validation, summaries, gates, failure formatting, and final envelope assembly to `control_analysis.py`. Move intervention/model work into four named functions in `control_execution.py`. Keep `controls.py` as the existing public import path and a thin `run_negative_controls()` coordinator.

**Tech Stack:** Python 3.11, PyTorch, pytest, Ruff

---

## File Structure

- Create `experiments/jlens_readout_sanity/control_analysis.py`: pure preparation, summaries, gates, and result assembly.
- Create `experiments/jlens_readout_sanity/control_execution.py`: identity, matched-random, wrong-concept, and random-target execution.
- Modify `experiments/jlens_readout_sanity/controls.py`: stable re-exports and short orchestration.
- Create `tests/experiments/jlens_readout_sanity/test_control_analysis.py`: analysis ownership and pure behavior.
- Create `tests/experiments/jlens_readout_sanity/test_control_execution.py`: execution ownership and identity behavior.
- Modify `tests/experiments/jlens_readout_sanity/test_controls.py`: retain public facade and orchestration coverage; remove tests moved to focused files.
- Modify `tests/experiments/jlens_readout_sanity/test_runner.py`: monkeypatch the execution module where intervention calls now occur.
- Modify `tests/experiments/jlens_readout_sanity/test_package.py`: enforce module boundaries and stable facade exports.

The line ranges below refer to `controls.py` at design commit `c530f40`.

### Task 1: Extract Pure Control Analysis

**Files:**
- Create: `experiments/jlens_readout_sanity/control_analysis.py`
- Modify: `experiments/jlens_readout_sanity/controls.py`
- Create: `tests/experiments/jlens_readout_sanity/test_control_analysis.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_package.py`

- [ ] **Step 1: Write failing module-ownership and facade tests**

Extend `tests/experiments/jlens_readout_sanity/test_package.py`:

```python
from experiments.jlens_readout_sanity import control_analysis


def test_control_analysis_owns_pure_control_logic() -> None:
    assert control_analysis._control_metadata.__module__ == (
        "experiments.jlens_readout_sanity.control_analysis"
    )
    assert control_analysis.summarize_wrong_concept.__module__ == (
        "experiments.jlens_readout_sanity.control_analysis"
    )
    assert control_analysis.aggregate_all_checks.__module__ == (
        "experiments.jlens_readout_sanity.control_analysis"
    )


def test_controls_facade_reexports_analysis_api() -> None:
    assert controls.summarize_wrong_concept is control_analysis.summarize_wrong_concept
    assert controls.require_exact_cases is control_analysis.require_exact_cases
    assert controls.controls_passed is control_analysis.controls_passed
    assert controls.aggregate_all_checks is control_analysis.aggregate_all_checks
```

- [ ] **Step 2: Run the ownership tests and verify RED**

Run:

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_package.py -q
```

Expected: collection fails because `control_analysis.py` does not exist.

- [ ] **Step 3: Create `control_analysis.py` by moving existing pure definitions**

Move these definitions from the original `controls.py` without changing their
bodies or returned keys:

- `_intervention_payload_at_alpha`, original lines 125-134;
- `require_exact_cases`, lines 137-143;
- `_ControlMetadata`, renamed to `ControlMetadata`, lines 146-154;
- `_wrong_reference_contexts`, lines 157-183;
- `_control_metadata`, lines 186-223, with its return annotation and constructor
  changed from `_ControlMetadata` to `ControlMetadata`;
- `summarize_wrong_concept`, lines 226-266;
- `controls_passed`, lines 269-271;
- `_control_failure`, lines 274-308;
- `aggregate_all_checks`, lines 311-326.

The new module imports only `Mapping`, `Sequence`, `dataclass`, `Any`,
`CONTROL_ALPHA`, `CONTROL_CHECK_MAP`, control reporting constants,
`InterventionContext`, `log_rank_gain`, `mean`, `percentile_label`,
`require_exact_case_keys`, and `concept_surfaces`. It must not import
`execute_intervention`, model objects, or vector-generation functions.

- [ ] **Step 4: Add failing tests for shared real-case preparation**

In `test_control_analysis.py`, test the new pure API:

```python
import math

import pytest

from experiments.jlens_readout_sanity.control_analysis import real_rank_gain_cases


def test_real_rank_gain_cases_selects_numeric_control_alpha() -> None:
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
```

- [ ] **Step 5: Run the preparation test and verify RED**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_control_analysis.py::test_real_rank_gain_cases_selects_numeric_control_alpha -q
```

Expected: FAIL because `real_rank_gain_cases` does not exist.

- [ ] **Step 6: Extract real-case preparation from the orchestrator**

Add this pure function to `control_analysis.py`, preserving the original
payload from lines 349-367:

```python
def real_rank_gain_cases(
    swap_results: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> tuple[list[dict[str, Any]], float]:
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
```

- [ ] **Step 7: Add final-envelope characterization tests**

Add this test using four minimal passing control payloads:

```python
from experiments.jlens_readout_sanity.control_analysis import (
    assemble_control_results,
)


def test_assemble_control_results_preserves_the_serialized_envelope() -> None:
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
```

- [ ] **Step 8: Run the envelope test and verify RED**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_control_analysis.py::test_assemble_control_results_preserves_the_serialized_envelope -q
```

Expected: FAIL because the assembler does not exist.

- [ ] **Step 9: Extract final-envelope assembly**

Add `assemble_control_results()` to `control_analysis.py` by moving the exact
dictionary construction from original lines 579-617. Its signature is:

```python
def assemble_control_results(
    *,
    expected_keys: Sequence[str],
    identity: Mapping[str, Any],
    matched_random_vector: Mapping[str, Any],
    wrong_concept: Mapping[str, Any],
    random_target: Mapping[str, Any],
) -> dict[str, Any]:
```

Inside, construct `control_results` in the existing order and return the
existing definitions, thresholds, tolerances, payloads, and `passed` field
unchanged.

- [ ] **Step 10: Turn `controls.py` imports into stable re-exports**

Import the moved names from `control_analysis.py` into `controls.py`. Preserve
the current import path for `runner.py` and tests:

```python
from experiments.jlens_readout_sanity.control_analysis import (
    _control_metadata,
    _wrong_reference_contexts,
    aggregate_all_checks,
    assemble_control_results,
    controls_passed,
    real_rank_gain_cases,
    require_exact_cases,
    summarize_wrong_concept,
)
```

Remove the moved definitions from `controls.py`. Replace its inline real-case
preparation and final dictionary construction with calls to the two new pure
functions. Do not move intervention execution yet.

- [ ] **Step 11: Run analysis, facade, and existing control tests**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_control_analysis.py \
  tests/experiments/jlens_readout_sanity/test_controls.py \
  tests/experiments/jlens_readout_sanity/test_package.py -q
uv run ruff check \
  experiments/jlens_readout_sanity/control_analysis.py \
  experiments/jlens_readout_sanity/controls.py \
  tests/experiments/jlens_readout_sanity/test_control_analysis.py \
  tests/experiments/jlens_readout_sanity/test_package.py
```

Expected: all tests and lint checks PASS.

- [ ] **Step 12: Commit the analysis split**

```bash
git add \
  experiments/jlens_readout_sanity/control_analysis.py \
  experiments/jlens_readout_sanity/controls.py \
  tests/experiments/jlens_readout_sanity/test_control_analysis.py \
  tests/experiments/jlens_readout_sanity/test_package.py
git commit -m "refactor: extract pure control analysis"
```

### Task 2: Extract Model-Backed Control Execution

**Files:**
- Create: `experiments/jlens_readout_sanity/control_execution.py`
- Modify: `experiments/jlens_readout_sanity/controls.py`
- Create: `tests/experiments/jlens_readout_sanity/test_control_execution.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_runner.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_package.py`

- [ ] **Step 1: Write failing execution ownership tests**

Add to `test_package.py`:

```python
from experiments.jlens_readout_sanity import control_execution


def test_control_execution_owns_model_backed_logic() -> None:
    assert control_execution.analyze_identity_case.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_identity_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_matched_random_vector_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_wrong_concept_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )
    assert control_execution.run_random_target_control.__module__ == (
        "experiments.jlens_readout_sanity.control_execution"
    )


def test_controls_facade_reexports_identity_analyzer() -> None:
    assert controls.analyze_identity_case is control_execution.analyze_identity_case
```

- [ ] **Step 2: Run ownership tests and verify RED**

Run:

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_package.py -q
```

Expected: collection fails because `control_execution.py` does not exist.

- [ ] **Step 3: Move the identity analyzer and rank payload helper**

Create `control_execution.py`. Move `analyze_identity_case` from original
lines 50-104 and `_rank_gain_payload` from lines 107-122 without changing
their behavior. Import `execute_intervention` in this new module.

- [ ] **Step 4: Extract the four execution functions**

Move each cohesive block from the original orchestrator into these functions:

```python
def run_identity_control(
    *, contexts, expected_keys, model, forward_next_token, layers
) -> dict[str, Any]:
    """Run and aggregate source-to-self identity interventions."""


def run_matched_random_vector_control(
    *, contexts, expected_keys, real_cases, real_mean,
    model, forward_next_token, layers
) -> dict[str, Any]:
    """Compare real swaps with deterministic matched-norm random vectors."""


def run_wrong_concept_control(
    *, contexts, wrong_references, expected_keys, real_cases,
    model, forward_next_token, layers
) -> dict[str, Any]:
    """Compare matched directions with deliberately mismatched directions."""


def run_random_target_control(
    *, contexts, metadata, expected_keys, real_cases, real_mean,
    model, lens, tokenizer, unembedding_weight,
    forward_next_token, layers
) -> dict[str, Any]:
    """Compare real targets with deterministic unrelated vocabulary targets."""
```

Use precise annotations matching the existing `run_negative_controls`
parameters. Function bodies are exact moves of original lines 369-397,
399-460, 462-498, and 500-577 respectively. Preserve loop nesting, `del`
statements, dictionary insertion order, seed order, target selection, and
execution order.

- [ ] **Step 5: Rewrite `run_negative_controls()` as orchestration**

After validation and `metadata = _control_metadata(contexts)`, the facade calls:

```python
real_cases, real_mean = real_rank_gain_cases(
    swap_results,
    expected_keys=metadata.expected_keys,
)
identity = run_identity_control(
    contexts=contexts,
    expected_keys=metadata.expected_keys,
    model=model,
    forward_next_token=forward_next_token,
    layers=layers,
)
matched_random_vector = run_matched_random_vector_control(
    contexts=contexts,
    expected_keys=metadata.expected_keys,
    real_cases=real_cases,
    real_mean=real_mean,
    model=model,
    forward_next_token=forward_next_token,
    layers=layers,
)
wrong_concept = run_wrong_concept_control(
    contexts=contexts,
    wrong_references=metadata.wrong_references,
    expected_keys=metadata.expected_keys,
    real_cases=real_cases,
    model=model,
    forward_next_token=forward_next_token,
    layers=layers,
)
random_target = run_random_target_control(
    contexts=contexts,
    metadata=metadata,
    expected_keys=metadata.expected_keys,
    real_cases=real_cases,
    real_mean=real_mean,
    model=model,
    lens=lens,
    tokenizer=tokenizer,
    unembedding_weight=unembedding_weight,
    forward_next_token=forward_next_token,
    layers=layers,
)
return assemble_control_results(
    expected_keys=metadata.expected_keys,
    identity=identity,
    matched_random_vector=matched_random_vector,
    wrong_concept=wrong_concept,
    random_target=random_target,
)
```

Import and re-export `analyze_identity_case` from the new module. Keep
`aggregate_all_checks` imported from analysis so `runner.py` remains unchanged.

- [ ] **Step 6: Update the integration monkeypatch location**

In `test_runner.py`, replace:

```python
import experiments.jlens_readout_sanity.controls as controls_module
```

with:

```python
import experiments.jlens_readout_sanity.control_execution as control_execution_module
```

Replace the existing `monkeypatch.setattr(controls_module,
"execute_intervention", ...)` call with the same patch against
`control_execution_module`. Keep the patches against the shared intervention
utility and runner because those symbols are independently used.

- [ ] **Step 7: Add a focused identity execution test**

Move the existing identity behavior tests from `test_runner.py` into
`test_control_execution.py`, keeping their tensors and expected payloads
unchanged. Import `analyze_identity_case` directly from `control_execution`.
Run the moved tests once before deleting the originals, then delete the
original copies and run again so no coverage is lost.

- [ ] **Step 8: Verify execution extraction and deterministic integration**

Run:

```bash
uv run pytest \
  tests/experiments/jlens_readout_sanity/test_control_execution.py \
  tests/experiments/jlens_readout_sanity/test_controls.py \
  tests/experiments/jlens_readout_sanity/test_runner.py \
  tests/experiments/jlens_readout_sanity/test_package.py -q
uv run ruff check \
  experiments/jlens_readout_sanity/control_execution.py \
  experiments/jlens_readout_sanity/controls.py \
  tests/experiments/jlens_readout_sanity/test_control_execution.py \
  tests/experiments/jlens_readout_sanity/test_runner.py
```

Expected: all tests PASS, including the integration assertions for 180
intervention calls, deterministic targets, vector construction counts, result
schema, and tensor-free serialization.

- [ ] **Step 9: Check the resulting responsibility sizes without enforcing caps**

Run:

```bash
wc -l \
  experiments/jlens_readout_sanity/controls.py \
  experiments/jlens_readout_sanity/control_analysis.py \
  experiments/jlens_readout_sanity/control_execution.py
rg -n '^def |^class |^@dataclass' \
  experiments/jlens_readout_sanity/controls.py \
  experiments/jlens_readout_sanity/control_analysis.py \
  experiments/jlens_readout_sanity/control_execution.py
```

Review for cohesion rather than exact limits. `controls.py` should contain only
the facade imports and orchestrator. `control_analysis.py` must have no model
execution import. `control_execution.py` may be around 300-350 lines if its
four functions are individually readable.

- [ ] **Step 10: Commit execution extraction**

```bash
git add \
  experiments/jlens_readout_sanity/control_execution.py \
  experiments/jlens_readout_sanity/controls.py \
  tests/experiments/jlens_readout_sanity/test_control_execution.py \
  tests/experiments/jlens_readout_sanity/test_runner.py \
  tests/experiments/jlens_readout_sanity/test_package.py
git commit -m "refactor: extract control execution"
```

### Task 3: Align the Remaining Tests and Verify the Refactor

**Files:**
- Modify: `tests/experiments/jlens_readout_sanity/test_controls.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_control_analysis.py`
- Verify: all production and test files changed in Tasks 1-2

- [ ] **Step 1: Move pure experiment tests to the analysis test module**

Move these existing tests and their local fixtures from `test_controls.py` to
`test_control_analysis.py` without changing assertions:

- metadata derivation and wrong-reference tests, current lines 95-194;
- exact case validation, wrong-concept summary, gate aggregation, failure
  messages, and serialization tests, current lines 457-641.

Keep shared utility tests for percentile math, subseeds, random-vector norm
matching, exclusions, and random-target selection in `test_controls.py` because
they exercise `jlens_reasoning.experiments_utils.controls`, not the new
experiment-local analysis module.

- [ ] **Step 2: Add a facade surface test**

In `test_controls.py`, assert the stable surface explicitly:

```python
def test_controls_public_surface_stays_stable() -> None:
    assert controls.run_negative_controls.__module__ == (
        "experiments.jlens_readout_sanity.controls"
    )
    assert controls.analyze_identity_case is control_execution.analyze_identity_case
    assert controls.aggregate_all_checks is control_analysis.aggregate_all_checks
```

- [ ] **Step 3: Run all focused control tests**

Run:

```bash
uv run pytest tests/experiments/jlens_readout_sanity -q
```

Expected: PASS with no duplicated test names or lost coverage.

- [ ] **Step 4: Run the full project suite**

Run:

```bash
uv run pytest
```

Expected: all tests PASS.

- [ ] **Step 5: Run formatting, lint, and diff checks**

Run:

```bash
uv run ruff format --check .
uv run ruff check .
git diff --check
```

Expected: all checks PASS. If formatting is required, run `uv run ruff format`
only on the changed Python files, then repeat all three commands.

- [ ] **Step 6: Review behavior-preservation evidence**

Confirm from the passing integration test that:

- all four control payloads retain their existing fields;
- `controls["passed"]` and global checks are unchanged;
- the execution count and order remain unchanged;
- random seeds and selected target IDs remain unchanged;
- no tensors enter the serialized result.

Also run:

```bash
git status --short --branch
git diff --stat HEAD~2..HEAD
```

Expected: the refactor is isolated to the planned control modules and tests;
the pre-existing unstaged README edit remains untouched.

- [ ] **Step 7: Commit the test organization if it is not already included**

```bash
git add \
  tests/experiments/jlens_readout_sanity/test_controls.py \
  tests/experiments/jlens_readout_sanity/test_control_analysis.py
git commit -m "test: align control coverage with module boundaries"
```
