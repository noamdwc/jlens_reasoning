# Readout Sanity Module Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exclude every Qwen-style added control token from random-target selection and replace the monolithic `readout_sanity.py` with a stable facade over focused modules.

**Architecture:** Move immutable case/token definitions into `readout_cases.py`, stateless shared helpers into `readout_utils.py`, intervention mechanics into `intervention_utils.py`, and negative-control execution into `readout_controls.py`. Keep readout analysis and top-level orchestration in `readout_sanity.py`, with explicit compatibility re-exports for every name currently imported by tests and the notebook.

**Tech Stack:** Python 3.11, PyTorch, jlens, pytest, Ruff.

**Design reference:** `docs/superpowers/specs/2026-07-21-readout-sanity-module-split-design.md`

---

## File Structure

- Modify `src/jlens_reasoning/experiments/sanity_controls.py`: exclude every `added_tokens_decoder` ID, including `special=False` control tokens.
- Modify `tests/test_sanity_controls.py`: add a Qwen-style non-special control-token regression.
- Create `src/jlens_reasoning/experiments/readout_cases.py`: case dataclasses, fixed cases, token variants, and case resolution.
- Create `src/jlens_reasoning/experiments/readout_utils.py`: ranking, formatting, workspace, validation, checks, and serialization helpers.
- Create `src/jlens_reasoning/experiments/intervention_utils.py`: J-Lens vectors, hooks, contexts, identity/swap analysis, and intervention helpers.
- Create `src/jlens_reasoning/experiments/readout_controls.py`: negative-control orchestration and result assembly.
- Modify `src/jlens_reasoning/experiments/readout_sanity.py`: retain readout analysis and top-level orchestration; re-export the prior import surface.
- Create `tests/test_readout_module_structure.py`: architecture and facade-compatibility tests.
- Modify `tests/test_readout_sanity.py`: patch the intervention implementation module rather than the facade after extraction.

## Task 1: Fix non-special added control-token exclusions

**Files:**

- Modify: `tests/test_sanity_controls.py`
- Modify: `src/jlens_reasoning/experiments/sanity_controls.py:225-266`

- [ ] **Step 1: Write the failing regression test**

Extend `VocabularyTokenizer.added_tokens_decoder` with a Qwen-style control
token whose `special` flag is false, then require it in `reserved_special` and
the full exclusion union:

```python
self.added_tokens_decoder = {
    15: SimpleNamespace(special=True),
    16: SimpleNamespace(special=False),
}

# In test_random_target_exclusions_are_token_id_based_and_complete:
assert exclusions["reserved_special"] == [0, 13, 14, 15, 16]
assert 16 in exclusions["all"]
```

Update the fixture vocabulary/expected union so ID 16 is otherwise eligible;
the assertion must prove exclusion comes from `added_tokens_decoder` rather
than another category.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py::test_random_target_exclusions_are_token_id_based_and_complete -q
```

Expected: FAIL because ID 16 is absent from `reserved_special` and `all`.

- [ ] **Step 3: Implement the minimal exclusion fix**

Replace the filtered set with all added-token IDs:

```python
added_control_ids = {
    int(token_id)
    for token_id in getattr(tokenizer, "added_tokens_decoder", {})
}
```

Use `added_control_ids` in the `reserved_special` category together with
`tokenizer.all_special_ids`. Do not alter selection, seeds, or the P2
random-target vector path.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: all sanity-control tests pass.

- [ ] **Step 5: Commit the behavioral fix**

```bash
git add src/jlens_reasoning/experiments/sanity_controls.py tests/test_sanity_controls.py
git commit -m "fix: exclude added control tokens from sanity targets"
```

## Task 2: Extract case and token definitions

**Files:**

- Create: `src/jlens_reasoning/experiments/readout_cases.py`
- Create: `tests/test_readout_module_structure.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py:44-201`

- [ ] **Step 1: Write a failing module-boundary test**

Create `tests/test_readout_module_structure.py`:

```python
from jlens_reasoning.experiments import readout_cases
from jlens_reasoning.experiments import readout_sanity


def test_case_definitions_live_in_focused_module_and_are_reexported() -> None:
    assert readout_sanity.ReadoutCase is readout_cases.ReadoutCase
    assert readout_sanity.SwapCase is readout_cases.SwapCase
    assert readout_sanity.TokenVariant is readout_cases.TokenVariant
    assert readout_sanity.READOUT_CASES is readout_cases.READOUT_CASES
    assert readout_sanity.SWAP_CASES is readout_cases.SWAP_CASES
    assert readout_sanity.single_token_surface is readout_cases.single_token_surface
    assert readout_sanity.concept_token_variants is readout_cases.concept_token_variants
```

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py -q
```

Expected: collection fails because `readout_cases` does not exist.

- [ ] **Step 3: Move the cohesive case/token block**

Create `readout_cases.py` with these existing definitions, unchanged:

```text
ReadoutCase
SwapCase
READOUT_CASES
SWAP_CASES
TokenVariant
ResolvedSwapCase
single_token_surface
resolve_swap_cases
_concept_surfaces
concept_token_variants
```

Import them explicitly in `readout_sanity.py` so the old module continues to
expose the same objects. Do not duplicate definitions in the facade.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py tests/test_readout_sanity.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the extraction**

```bash
git add src/jlens_reasoning/experiments/readout_cases.py src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_module_structure.py
git commit -m "refactor: extract readout case definitions"
```

## Task 3: Extract stateless readout utilities

**Files:**

- Create: `src/jlens_reasoning/experiments/readout_utils.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_module_structure.py`

- [ ] **Step 1: Extend the boundary test for utilities**

Add:

```python
from jlens_reasoning.experiments import readout_utils


def test_stateless_utilities_are_reexported_from_facade() -> None:
    exported_names = (
        "find_last_subsequence",
        "positions_after_literal",
        "best_target_rank",
        "top_tokens",
        "prepare_scoring_input",
        "aggregate_capability_checks",
        "workspace_loading",
        "workspace_layers",
        "write_results",
        "validate_model_lens",
    )
    for name in exported_names:
        assert getattr(readout_sanity, name) is getattr(readout_utils, name)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py::test_stateless_utilities_are_reexported_from_facade -q
```

Expected: collection fails because `readout_utils` does not exist.

- [ ] **Step 3: Move stateless helpers without behavior changes**

Move these existing functions to `readout_utils.py`:

```text
find_last_subsequence
positions_after_literal
positions_from_literal
best_target_rank
top_tokens
prepare_scoring_input
aggregate_capability_checks
workspace_loading
workspace_layers
_jsonable
write_results
validate_model_lens
```

Import `_concept_surfaces` from `readout_cases.py`. Move `TOP_K = 25` with
`top_tokens`, then import and re-export `TOP_K` and the public helpers in
`readout_sanity.py`. This keeps one authoritative default value.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py tests/test_readout_sanity.py tests/test_sanity_controls.py -q
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit the extraction**

```bash
git add src/jlens_reasoning/experiments/readout_utils.py src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_module_structure.py
git commit -m "refactor: extract readout utilities"
```

## Task 4: Extract intervention mechanics

**Files:**

- Create: `src/jlens_reasoning/experiments/intervention_utils.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_module_structure.py`
- Modify: `tests/test_readout_sanity.py:1-35, 813-834`

- [ ] **Step 1: Add a failing intervention-module test**

Add:

```python
from jlens_reasoning.experiments import intervention_utils


def test_intervention_mechanics_are_reexported_from_facade() -> None:
    exported_names = (
        "LensCoordinateSwapper",
        "jlens_vector",
        "coordinate_swap",
        "execute_intervention",
        "analyze_identity_case",
        "summarize_swap_logits",
        "analyze_swap_case",
        "_token_vectors_by_layer",
    )
    for name in exported_names:
        assert getattr(readout_sanity, name) is getattr(intervention_utils, name)
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py::test_intervention_mechanics_are_reexported_from_facade -q
```

Expected: collection fails because `intervention_utils` does not exist.

- [ ] **Step 3: Move intervention definitions**

Move these definitions unchanged, updating imports only:

```text
InterventionContext
jlens_vector
coordinate_swap
LensCoordinateSwapper
execute_intervention
_next_token_payload
analyze_identity_case
summarize_swap_logits
_token_vectors_by_layer
_prepare_intervention_context
_rank_gain_payload
_intervention_payload_at_alpha
analyze_swap_case
```

The module imports case types/token helpers from `readout_cases.py` and rank,
workspace, formatting, and position helpers from `readout_utils.py`.
`readout_sanity.py` explicitly imports the public compatibility names plus the
private context helpers used by orchestration.

- [ ] **Step 4: Update the integration monkeypatch target**

In `tests/test_readout_sanity.py`, import:

```python
import jlens_reasoning.experiments.intervention_utils as intervention_utils_module
```

Use `intervention_utils_module.execute_intervention` as the real function. In
this intermediate task, patch both module attributes:

```python
monkeypatch.setattr(
    intervention_utils_module,
    "execute_intervention",
    recording_execute_intervention,
)
monkeypatch.setattr(
    readout_sanity_module,
    "execute_intervention",
    recording_execute_intervention,
)
```

The first patch records swap execution after extraction; the second records
controls that still reside in the facade until Task 5.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py tests/test_readout_sanity.py -q
```

Expected: all selected tests pass, including hook cleanup and the 180-call
integration invariant.

- [ ] **Step 6: Commit the extraction**

```bash
git add src/jlens_reasoning/experiments/intervention_utils.py src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_module_structure.py tests/test_readout_sanity.py
git commit -m "refactor: extract intervention utilities"
```

## Task 5: Extract negative-control orchestration

**Files:**

- Create: `src/jlens_reasoning/experiments/readout_controls.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py:913-1200`
- Modify: `tests/test_readout_module_structure.py`

- [ ] **Step 1: Add a failing controls-module boundary test**

Add:

```python
from jlens_reasoning.experiments import readout_controls


def test_negative_control_orchestration_has_a_focused_module() -> None:
    assert callable(readout_controls.run_negative_controls)
    assert readout_sanity._run_negative_controls is readout_controls.run_negative_controls
```

- [ ] **Step 2: Run the boundary test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py::test_negative_control_orchestration_has_a_focused_module -q
```

Expected: collection fails because `readout_controls` does not exist.

- [ ] **Step 3: Move control orchestration verbatim**

Move `_run_negative_controls` to `readout_controls.py` and rename it
`run_negative_controls`. Import intervention operations through the module:

```python
from jlens_reasoning.experiments import intervention_utils as interventions
```

Call `interventions.execute_intervention`,
`interventions._rank_gain_payload`, and
`interventions._token_vectors_by_layer` so one monkeypatch records all shared
execution calls. Preserve the current random-target loop structure exactly:
`_token_vectors_by_layer` remains inside the inner context loop, leaving P2
unmodified.

In `readout_sanity.py`, import:

```python
from jlens_reasoning.experiments.readout_controls import (
    run_negative_controls as _run_negative_controls,
)
```

After this move, remove the now-ineffective `readout_sanity_module` monkeypatch
from the integration test and keep only the `intervention_utils_module` patch.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py tests/test_readout_sanity.py tests/test_sanity_controls.py -q
```

Expected: all selected tests pass, with the integration test still observing
180 interventions and unchanged control schemas.

- [ ] **Step 5: Commit the extraction**

```bash
git add src/jlens_reasoning/experiments/readout_controls.py src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_module_structure.py
git commit -m "refactor: extract readout control orchestration"
```

## Task 6: Final facade cleanup and verification

**Files:**

- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_module_structure.py`

- [ ] **Step 1: Add final facade-size and import assertions**

Add:

```python
from pathlib import Path


def test_readout_sanity_is_a_small_stable_facade() -> None:
    facade = Path(readout_sanity.__file__)
    assert len(facade.read_text(encoding="utf-8").splitlines()) < 500
    assert readout_sanity.run_readout_sanity.__module__ == (
        "jlens_reasoning.experiments.readout_sanity"
    )
```

The line cap guards against recreating the monolith while leaving room for
readout analysis and orchestration.

- [ ] **Step 2: Run the new assertion and verify current state**

Run:

```bash
.venv/bin/pytest tests/test_readout_module_structure.py -q
```

Expected: PASS only after the four module extractions; if it fails, remove
remaining misplaced helpers from the facade without changing behavior.

- [ ] **Step 3: Normalize imports and module documentation**

Remove unused imports from `readout_sanity.py`, add concise module docstrings
to every extracted file, and keep explicit facade imports for all names listed
in `tests/test_readout_sanity.py`. Do not introduce wildcard imports or change
the random-target vector loop.

- [ ] **Step 4: Run focused verification**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_readout_sanity.py tests/test_readout_module_structure.py tests/test_notebooks.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Run full repository verification**

Run:

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
```

Expected: 0 test failures, no Ruff diagnostics, all files formatted, and no
whitespace errors.

- [ ] **Step 6: Inspect scope and preserve the user change**

Run:

```bash
git status --short
git diff --stat f6216a6308d38b5ad0e733a830ee677c5b7c5131..HEAD
git diff f6216a6308d38b5ad0e733a830ee677c5b7c5131..HEAD -- src/jlens_reasoning/experiments tests
```

Confirm `README.md` remains modified but unstaged, P1 is fixed, P2 is unchanged,
and only the approved modules/tests/docs changed.

- [ ] **Step 7: Commit final cleanup if needed**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_module_structure.py
git commit -m "refactor: finalize readout sanity facade"
```
