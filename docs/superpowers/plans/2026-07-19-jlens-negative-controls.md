# J-Lens Sanity Negative Controls Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add identity, matched random-vector, wrong-concept, and random-target hard gates to the existing five-case J-Lens read-and-change sanity experiment while preserving every existing prompt, metric, threshold, check, and gate.

**Architecture:** Put deterministic control math, random generation, token filtering, and gate aggregation in a new pure `sanity_controls.py` module. Refactor the existing experiment runner around one reusable intervention context and one hook-backed execution helper so the real `alpha=1` swap and every control share scoring inputs, workspace layers, vector construction/preprocessing, hook locations, activation positions, and execution. Keep the notebook as orchestration and concise reporting only.

**Tech Stack:** Python 3.11, PyTorch, `jlens`, Transformers tokenizer interfaces, pytest, nbformat, Ruff.

**Design reference:** `docs/superpowers/specs/2026-07-17-jlens-negative-controls-design.md`

---

## File Structure

- Create `src/jlens_reasoning/experiments/sanity_controls.py`: fixed seeds, tolerances, rank gain, percentile, stable sub-seeds, matched-norm random vectors, random-target exclusions/selection, wrong-concept summaries, and the single global check/failure aggregator.
- Create `tests/test_sanity_controls.py`: CPU-only unit tests for all deterministic control calculations and schemas.
- Modify `src/jlens_reasoning/experiments/readout_sanity.py`: share one prepared per-case runtime and one hook-backed intervention executor across real and control conditions; assemble all four control payloads.
- Modify `tests/test_readout_sanity.py`: verify identity execution and full runner/check/failure integration using tiny real tensor hooks and tokenizer/model stubs.
- Modify `notebooks/01_jlens_readout_sanity.ipynb`: print the required concise control summary without changing model loading, cases, saved artifact path, or final failure behavior.
- Modify `tests/test_notebooks.py`: assert the notebook reports all four controls and overall control status.

The pre-existing uncommitted `README.md` change is outside scope and must remain untouched.

## Task 1: Deterministic scores, percentile, and seed derivation

**Files:**

- Create: `src/jlens_reasoning/experiments/sanity_controls.py`
- Create: `tests/test_sanity_controls.py`

- [ ] **Step 1: Write failing tests for shared definitions**

Create `tests/test_sanity_controls.py` with imports and tests equivalent to:

```python
import math

import pytest

from jlens_reasoning.experiments.sanity_controls import (
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    derive_subseed,
    log_rank_gain,
    percentile,
    strict_percentile_gate,
)


def test_shared_control_definitions_are_fixed() -> None:
    assert len(CONTROL_SEEDS) == 16
    assert len(set(CONTROL_SEEDS)) == 16
    assert all(isinstance(seed, int) for seed in CONTROL_SEEDS)
    assert IDENTITY_ATOL == 1e-6
    assert IDENTITY_RTOL == 1e-5


def test_log_rank_gain_uses_natural_logarithms() -> None:
    assert log_rank_gain(100, 10) == pytest.approx(math.log(10.0))
    assert log_rank_gain(10, 100) == pytest.approx(-math.log(10.0))
    assert log_rank_gain(7, 7) == 0.0


def test_percentile_uses_documented_linear_interpolation() -> None:
    values = [float(value) for value in range(16)]
    assert percentile(values, 0.95) == pytest.approx(14.25)


def test_percentile_gate_is_strict() -> None:
    values = [0.0] * 15 + [4.0]
    threshold = percentile(values, 0.95)
    assert strict_percentile_gate(threshold, values, quantile=0.95)["passed"] is False
    assert strict_percentile_gate(threshold + 1e-12, values, quantile=0.95)[
        "passed"
    ] is True


def test_subseeds_are_stable_and_role_specific() -> None:
    assert derive_subseed(11, 7, "source") == derive_subseed(11, 7, "source")
    assert derive_subseed(11, 7, "source") != derive_subseed(11, 7, "target")
    assert derive_subseed(11, 7, "source") != derive_subseed(11, 8, "source")
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: collection fails because `sanity_controls` does not exist.

- [ ] **Step 3: Implement the shared deterministic definitions**

Create `sanity_controls.py` with:

```python
"""Deterministic negative-control calculations for J-Lens sanity runs."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Any

CONTROL_SEEDS = (
    11,
    29,
    47,
    71,
    101,
    131,
    167,
    199,
    239,
    281,
    331,
    379,
    431,
    487,
    547,
    607,
)
IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5
NORM_ATOL = 1e-6
NORM_RTOL = 1e-5
PERCENTILE_QUANTILE = 0.95


def log_rank_gain(clean_rank: int, intervened_rank: int) -> float:
    if clean_rank < 1 or intervened_rank < 1:
        raise ValueError("Ranks must be positive one-based integers")
    return math.log(clean_rank) - math.log(intervened_rank)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return math.fsum(values) / len(values)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must lie in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def strict_percentile_gate(
    real_score: float,
    control_scores: Sequence[float],
    *,
    quantile: float = PERCENTILE_QUANTILE,
) -> dict[str, Any]:
    threshold = percentile(control_scores, quantile)
    return {
        "real_score": float(real_score),
        "percentile": quantile,
        "threshold": threshold,
        "comparison": "real_score > threshold",
        "passed": float(real_score) > threshold,
    }


def derive_subseed(base_seed: int, layer_index: int, role: str) -> int:
    if role not in {"source", "target"}:
        raise ValueError("Role must be 'source' or 'target'")
    payload = f"jlens-control-v1:{base_seed}:{layer_index}:{role}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add src/jlens_reasoning/experiments/sanity_controls.py tests/test_sanity_controls.py
git commit -m "feat: add deterministic sanity control definitions"
```

## Task 2: Deterministic norm-matched random vectors

**Files:**

- Modify: `src/jlens_reasoning/experiments/sanity_controls.py`
- Modify: `tests/test_sanity_controls.py`

- [ ] **Step 1: Add failing vector tests**

Append tests that use real tensors rather than mocked vector functions:

```python
import torch

from jlens_reasoning.experiments.sanity_controls import matched_random_vectors


def test_random_vectors_are_deterministic_for_fixed_seed() -> None:
    real = {
        7: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        9: (torch.tensor([1.0, 0.0]), torch.tensor([5.0, 12.0])),
    }
    first, _ = matched_random_vectors(real, base_seed=CONTROL_SEEDS[0])
    second, _ = matched_random_vectors(dict(reversed(list(real.items()))), base_seed=CONTROL_SEEDS[0])
    assert all(torch.equal(first[layer][0], second[layer][0]) for layer in real)
    assert all(torch.equal(first[layer][1], second[layer][1]) for layer in real)


def test_random_vectors_match_each_per_layer_role_norm() -> None:
    real = {
        2: (torch.tensor([3.0, 4.0]), torch.tensor([0.0, 2.0])),
        3: (torch.zeros(2), torch.tensor([5.0, 12.0])),
    }
    generated, norms = matched_random_vectors(real, base_seed=CONTROL_SEEDS[1])

    for layer in sorted(real):
        for role_index, role in enumerate(("source", "target")):
            assert torch.linalg.vector_norm(generated[layer][role_index]) == pytest.approx(
                torch.linalg.vector_norm(real[layer][role_index]).item(),
                abs=1e-6,
                rel=1e-5,
            )
            assert norms[str(layer)][role]["matched"] is True
    assert torch.equal(generated[3][0], torch.zeros(2))
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: import fails for `matched_random_vectors`.

- [ ] **Step 3: Implement CPU-float32 random generation and norm recording**

Add `torch`, `Mapping`, and functions that:

```python
def _matched_random_vector(
    real_vector: torch.Tensor,
    *,
    base_seed: int,
    layer_index: int,
    role: str,
) -> torch.Tensor:
    real_cpu = real_vector.detach().to(device="cpu", dtype=torch.float32)
    real_norm = torch.linalg.vector_norm(real_cpu)
    if real_norm.item() == 0.0:
        return torch.zeros_like(real_cpu)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(derive_subseed(base_seed, layer_index, role))
    random_vector = torch.randn(real_cpu.shape, generator=generator, dtype=torch.float32)
    random_norm = torch.linalg.vector_norm(random_vector)
    if random_norm.item() == 0.0:
        random_vector.zero_()
        random_vector[0] = 1.0
        random_norm = torch.linalg.vector_norm(random_vector)
    return random_vector * (real_norm / random_norm)
```

`matched_random_vectors` must iterate `sorted(real_vectors)`, generate source
and target roles separately, move each generated tensor to the corresponding
real tensor's device, and return both the vector mapping and a JSON-ready norm
report using `torch.isclose(..., atol=NORM_ATOL, rtol=NORM_RTOL)`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: all Task 1-2 tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/jlens_reasoning/experiments/sanity_controls.py tests/test_sanity_controls.py
git commit -m "feat: generate norm-matched control vectors"
```

## Task 3: Random-target exclusions and deterministic selection

**Files:**

- Modify: `src/jlens_reasoning/experiments/sanity_controls.py`
- Modify: `tests/test_sanity_controls.py`

- [ ] **Step 1: Add a lightweight vocabulary tokenizer and failing tests**

Add a tokenizer stub with `encode`, `decode`, `get_vocab`, and
`all_special_ids`. Its vocabulary must include ordinary tokens, source and
target concepts, a multi-token intended answer, whitespace-only formatting,
reserved/special IDs, and an explicitly prefiltered ID. Add tests asserting:

```python
exclusions = build_random_target_exclusions(
    tokenizer,
    source_surfaces=("spider", "France"),
    target_surfaces=("ant", "China"),
    intended_answer_surfaces=("six", "New Yuan"),
    formatting_token_ids=(8,),
    existing_excluded_ids=(9,),
)
excluded = set(exclusions["all"])
assert {source_id, france_id, ant_id, china_id} <= excluded
assert {new_id, yuan_id} <= excluded
assert {formatting_id, whitespace_id, special_id, reserved_id, 9} <= excluded
```

Also select twice with `CONTROL_SEEDS` and assert identical ordered results,
16 results, uniqueness when at least 16 eligible IDs exist, and every result
outside the exclusion set.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: imports fail for the exclusion and selection helpers.

- [ ] **Step 3: Implement ID-based exclusions**

Add:

```python
def _encoded_ids(tokenizer: Any, surfaces: Sequence[str]) -> set[int]:
    return {
        int(token_id)
        for surface in surfaces
        for token_id in tokenizer.encode(surface, add_special_tokens=False)
    }
```

`build_random_target_exclusions` must return sorted per-category lists for
`sources`, `targets`, `intended_answers`, `formatting`, `reserved_special`,
`decoded_formatting`, `existing_filter`, and their sorted union `all`.
Reserved IDs come from `tokenizer.all_special_ids` plus special entries exposed
by `added_tokens_decoder`. Decoded formatting IDs are vocabulary IDs whose
single-token decoded surface is empty or whitespace-only. Multi-token inputs
contribute every encoded ID.

- [ ] **Step 4: Implement deterministic unique selection**

`select_random_targets` must build the ascending list of unique IDs from
`tokenizer.get_vocab().values()`, subtract exclusions, and for each seed use:

```python
digest = hashlib.sha256(f"jlens-random-target-v1:{seed}".encode()).digest()
index = int.from_bytes(digest[:8], "big") % len(remaining)
token_id = remaining.pop(index)
```

If the remaining list is exhausted before 16 selections, reset it to the full
eligible list. Return `[{"seed": seed, "token_id": id, "token": decoded}, ...]`.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: all Task 1-3 tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/jlens_reasoning/experiments/sanity_controls.py tests/test_sanity_controls.py
git commit -m "feat: select deterministic random target controls"
```

## Task 4: Control summaries and one global aggregation path

**Files:**

- Modify: `src/jlens_reasoning/experiments/sanity_controls.py`
- Modify: `tests/test_sanity_controls.py`

- [ ] **Step 1: Write failing wrong-concept and aggregation tests**

Add tests for `summarize_wrong_concept` using five matched and five mismatched
case dictionaries. Cover four strict wins passing, three wins failing, and a
tie not counting. Assert both aggregates are arithmetic means.

Add tests for `aggregate_all_checks` with four control payloads:

```python
controls = {
    "identity": {"passed": True},
    "matched_random_vector": {"passed": True},
    "wrong_concept": {"passed": True},
    "random_target": {"passed": True},
}
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
assert passed is True
```

Parameterize each control as failed and assert global `passed` is false and a
failure containing the control name plus observed/required values is appended.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: imports fail for the summary and aggregation helpers.

- [ ] **Step 3: Implement strict wrong-concept comparison**

`summarize_wrong_concept` must pair cases by key, compute arithmetic means,
mark `matched_wins = matched_gain > mismatched_gain`, count wins, and set:

```python
aggregate_condition = matched_mean > mismatched_mean
case_condition = winning_case_count >= 4
passed = aggregate_condition and case_condition
```

Return both conditions, both means, all paired cases, required wins, observed
wins, and the combined gate.

- [ ] **Step 4: Implement the authoritative gate map and aggregator**

Define one ordered tuple:

```python
CONTROL_CHECK_MAP = (
    ("identity", "identity_control"),
    ("matched_random_vector", "matched_random_vector_control"),
    ("wrong_concept", "wrong_concept_control"),
    ("random_target", "random_target_control"),
)
```

`aggregate_all_checks` must copy existing checks/failures, iterate this tuple,
add every control result, call one control-specific failure formatter for each
false result, and return `(checks, failures, all(checks.values()))`. Missing
control payloads must raise rather than silently omit a gate.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py -q
```

Expected: all pure control tests pass.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/jlens_reasoning/experiments/sanity_controls.py tests/test_sanity_controls.py
git commit -m "feat: aggregate negative control gates"
```

## Task 5: Share the real alpha=1 execution path and add identity

**Files:**

- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_sanity.py`

- [ ] **Step 1: Write failing shared-path and identity tests**

Extend the existing tiny model/block test infrastructure. Add a test that
prepares one case runtime, runs real and identity conditions through the same
hook executor, and asserts hook calls cover the same workspace layers and all
activation positions. Assert identity reports unchanged top-1, unchanged
intended-target rank, `logits_close=True`, maximum difference, exact
tolerances, and `passed=True`.

Add a failure test where the tiny forward deliberately perturbs identity
logits beyond `atol=1e-6`, `rtol=1e-5`, and assert the identity gate fails.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_readout_sanity.py -q
```

Expected: imports or assertions fail because the shared runtime and identity
runner do not exist.

- [ ] **Step 3: Introduce a per-case intervention context**

Add an internal dataclass containing the resolved cases/tokens, raw and
formatting-adjusted inputs, formatting prefix, clean logits, intended-target
IDs, workspace layers, and real per-layer source/target J-Lens vectors.

Build it once per case in ascending case order. Source/target token vectors
must be constructed only with the existing `jlens_vector` function. Preserve
the existing France workspace-loading calculation and every existing JSON
field.

- [ ] **Step 4: Extract the single hook-backed executor**

Add one function with the effective behavior:

```python
def execute_intervention(
    context: InterventionContext,
    *,
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
) -> torch.Tensor:
    with (
        torch.inference_mode(),
        LensCoordinateSwapper(model.layers, vectors_by_layer, alpha=alpha),
    ):
        return forward_next_token(context.scoring_input)
```

Route the existing real `alpha=1` and `alpha=2` runs through this helper before
adding controls. Re-run all existing readout tests to prove behavior and schema
are unchanged.

- [ ] **Step 5: Implement identity through that executor**

For spider use the real source vector as both roles; for all France cases do
the same. Execute all workspace hooks together at `alpha=1`. Compare detached
float logits with `torch.allclose(atol=IDENTITY_ATOL, rtol=IDENTITY_RTOL)`, the
existing top-1 calculation, and `best_target_rank`. Return per-case results,
active layers, tolerances, global maximum absolute difference, and the all-case
gate.

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_readout_sanity.py -q
```

Expected: all existing tests plus the shared-path identity tests pass.

- [ ] **Step 7: Commit Task 5**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py
git commit -m "refactor: share sanity intervention execution path"
```

## Task 6: Execute all randomized and wrong-concept controls

**Files:**

- Modify: `src/jlens_reasoning/experiments/readout_sanity.py`
- Modify: `tests/test_readout_sanity.py`
- Modify: `tests/test_sanity_controls.py`

- [ ] **Step 1: Write failing integration tests for all controls**

Extend the tiny deterministic setup to five cases. Make its forward function
produce controlled ranks for real, random-vector, wrong-concept, and
random-target vector pairs while still passing through real tensor hooks.
Assert:

- every randomized intervention uses `alpha=1`;
- every result scores the original real target IDs;
- matched-random contains exactly 16 seed entries and five cases per entry;
- wrong-concept uses `France→China` for spider and `spider→ant` for France;
- random-target contains exactly 16 selected target entries and five cases per
  entry;
- sampled target vectors equal `jlens_vector(lens, unembedding_weight, layer,
  sampled_token_id)` before entering the common executor;
- real aggregates use the five existing real `alpha=1` ranks;
- both percentile comparisons are strict.

- [ ] **Step 2: Run integration tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_readout_sanity.py -q
```

Expected: failures because the three controls are not orchestrated.

- [ ] **Step 3: Add a reusable case score payload**

For every condition record `key`, clean rank, intervened rank, and
`log_rank_gain`. Compute the real per-case payload once from the existing
`swaps[*].clean.target_rank` and `swaps[*].interventions["1.0"].target_rank`.
Its aggregate is `mean` across exactly the five cases.

- [ ] **Step 4: Add matched random-vector execution**

For each `CONTROL_SEEDS` entry and each context, call
`matched_random_vectors(context.real_vectors, base_seed=seed)`, then call the
shared executor at `alpha=1`. Record per-layer role norms, per-case gains, and
seed means. Pass the 16 means and real mean to `strict_percentile_gate`.

- [ ] **Step 5: Add wrong-concept execution**

Resolve the configured `France`, `China`, `spider`, and `ant` token IDs through
the same strict token-surface path as the real swaps. Build every mismatched
per-layer vector with `jlens_vector`, call the shared executor at `alpha=1`,
score each context's original target IDs, and pass matched/mismatched case
payloads to `summarize_wrong_concept`.

- [ ] **Step 6: Add random-target execution**

Build categorized exclusions from all configured source/target surfaces,
every accepted target-answer surface variant, observed formatting-prefix IDs,
tokenizer reserved/special IDs, whitespace-only IDs, and existing filter IDs.
Select exactly 16 targets with `CONTROL_SEEDS`. For each target and context,
construct the target vector with `jlens_vector` at every workspace layer, pair
it with the context's real source vector, and call the shared executor at
`alpha=1`. Record token IDs/strings, five per-case gains, means, exclusions,
and the strict percentile gate.

- [ ] **Step 7: Assemble the complete controls schema**

Add top-level `controls` with shared seeds, definitions, thresholds,
tolerances, four payloads, and `passed`. Call `aggregate_all_checks` once with
the unchanged output from `aggregate_capability_checks`; use its returned
checks, failures, and passed values as the only global aggregation path.

- [ ] **Step 8: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_readout_sanity.py -q
```

Expected: all focused tests pass without a model download, GPU, or network.

- [ ] **Step 9: Commit Task 6**

```bash
git add src/jlens_reasoning/experiments/readout_sanity.py tests/test_readout_sanity.py tests/test_sanity_controls.py
git commit -m "feat: gate J-Lens sanity with negative controls"
```

## Task 7: Stable serialization and notebook summary

**Files:**

- Modify: `tests/test_sanity_controls.py`
- Modify: `tests/test_notebooks.py`
- Modify: `notebooks/01_jlens_readout_sanity.ipynb`

- [ ] **Step 1: Write failing serialization and notebook tests**

Serialize a complete small control result twice with `write_results`; assert
byte-for-byte equality after stable JSON writing and verify `controls` contains
the seed list, definitions, thresholds, tolerances, per-case results, repeated
results, individual gates, and overall gate.

In `tests/test_notebooks.py`, assert the notebook source contains labels for:

```text
identity_control
matched_random_vector_control
wrong_concept_control
random_target_control
overall_controls
```

and still contains the existing result path, read/swap output, HTML rendering,
and final global failure check.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_notebooks.py -q
```

Expected: notebook reporting assertions fail.

- [ ] **Step 3: Add the concise notebook control summary**

After saving `result.json`, print one line per requested control with observed
values and pass/fail, plus `overall_controls`. Do not print the 16-entry arrays
on success. Preserve all existing reporting and the final `result["passed"]`
failure behavior.

Use `nbformat` or a JSON-aware formatter to edit the notebook, then run Ruff's
notebook formatting so source cells remain canonical.

- [ ] **Step 4: Run serialization and notebook tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_notebooks.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add notebooks/01_jlens_readout_sanity.ipynb tests/test_notebooks.py tests/test_sanity_controls.py
git commit -m "feat: report J-Lens sanity controls"
```

## Task 8: Final verification and review

**Files:**

- Verify all changed files; make only fixes caused by this implementation.

- [ ] **Step 1: Run focused control tests**

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_readout_sanity.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full test suite**

```bash
.venv/bin/pytest -q
```

Expected: all tests pass. Record any unrelated pre-existing failure separately
after confirming it reproduces without these changes.

- [ ] **Step 3: Run repository Ruff checks**

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Expected: both commands exit zero.

- [ ] **Step 4: Check patch whitespace and scope**

```bash
git diff --check
git status --short
git diff --stat origin/codex/jlens-readout-sanity...HEAD
```

Expected: `git diff --check` exits zero; only planned implementation files,
design/plan documents, commits, and the pre-existing unstaged README change
appear.

- [ ] **Step 5: Inspect the final diff against every requirement**

Confirm exact prompts/cases, existing alphas/gates, all four controls, 16
recorded seeds, strict percentile behavior, exact J-Lens token-vector path,
five-case arithmetic means, strict wrong-concept wins, one gate aggregation
path, actionable failures, stable JSON, and concise notebook output.

- [ ] **Step 6: Request code review**

Invoke `superpowers:requesting-code-review`, address any verified issues, and
rerun Steps 1-4 before claiming completion.
