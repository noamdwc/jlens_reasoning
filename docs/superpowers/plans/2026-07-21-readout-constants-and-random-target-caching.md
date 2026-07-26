# Readout Constants and Random-Target Caching Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove redundant random-target J-Lens vector construction and give experiment policy constants focused ownership without breaking existing imports or changing experiment results.

**Architecture:** Add a private single-token vector helper and cache each selected target's per-layer vectors outside the context loop, reusing source vectors already stored in each `InterventionContext`. Add `readout_constants.py` and `sanity_constants.py` as focused owners while retaining compatibility aliases in `readout_sanity.py`, `readout_utils.py`, and `sanity_controls.py`.

**Tech Stack:** Python 3.11, PyTorch, pytest, Ruff.

**Design reference:** `docs/superpowers/specs/2026-07-21-readout-constants-and-random-target-caching-design.md`

---

## File Structure

- Create `src/jlens_reasoning/experiments/readout_constants.py`: model/lens artifact coordinates and readout policy defaults.
- Create `src/jlens_reasoning/experiments/sanity_constants.py`: negative-control seeds, keys, thresholds, tolerances, and execution limits.
- Modify `src/jlens_reasoning/experiments/intervention_utils.py`: add one private single-token vector helper and compose the existing pair helper from it.
- Modify `src/jlens_reasoning/experiments/readout_controls.py`: cache random-target vectors outside the context loop and consume constants from their owner.
- Modify `src/jlens_reasoning/experiments/readout_sanity.py`: consume and re-export readout constants.
- Modify `src/jlens_reasoning/experiments/readout_utils.py`: consume workspace/default constants and re-export `TOP_K`.
- Modify `src/jlens_reasoning/experiments/sanity_controls.py`: consume and re-export sanity constants.
- Modify `tests/test_readout_sanity.py`: prove the P2 optimization removes redundant vector construction without changing control execution.
- Create `tests/test_experiment_constants.py`: enforce focused constants ownership and legacy compatibility aliases without broadening the unrelated facade-export table.

## Task 1: Cache Random-Target Vectors Once Per Target and Layer

**Files:**

- Modify: `tests/test_readout_sanity.py:790-980`
- Modify: `src/jlens_reasoning/experiments/intervention_utils.py:230-265`
- Modify: `src/jlens_reasoning/experiments/readout_controls.py:242-266`

- [ ] **Step 1: Add the failing vector-construction count assertion**

In `test_run_readout_sanity_integrates_all_controls_without_storing_logits`, add a recorder beside the existing intervention recorder, before calling `run_readout_sanity`:

```python
    vector_calls: list[dict[str, int]] = []
    real_jlens_vector = intervention_utils_module.jlens_vector

    def recording_jlens_vector(*args, **kwargs):
        vector_calls.append(
            {
                "layer": kwargs["layer"],
                "token_id": kwargs["token_id"],
            }
        )
        return real_jlens_vector(*args, **kwargs)

    monkeypatch.setattr(
        intervention_utils_module,
        "jlens_vector",
        recording_jlens_vector,
    )
```

Immediately after `controls = result["controls"]`, add:

```python
    selected_target_ids = {
        target["token_id"] for target in controls["random_target"]["targets"]
    }
    assert len(selected_target_ids) == len(CONTROL_SEEDS)
    assert len(vector_calls) == 10 + len(CONTROL_SEEDS)
    assert all(
        sum(call["token_id"] == token_id for call in vector_calls) == 1
        for token_id in selected_target_ids
    )
```

The fixed `10` is the expected five contexts times one workspace layer times one source and one real target vector. Each of the 16 selected random targets must add only one construction for that layer.

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_readout_sanity.py::test_run_readout_sanity_integrates_all_controls_without_storing_logits -q
```

Expected: both parametrizations fail because the current implementation records `170` vector constructions instead of `26`.

- [ ] **Step 3: Add the private single-token vector helper**

In `intervention_utils.py`, add this helper immediately before `_token_vectors_by_layer`:

```python
def _single_token_vectors_by_layer(
    *,
    lens: Any,
    unembedding_weight: torch.Tensor,
    layers: Sequence[int],
    token_id: int,
) -> dict[int, torch.Tensor]:
    return {
        layer: jlens_vector(
            lens,
            unembedding_weight,
            layer=layer,
            token_id=token_id,
        )
        for layer in layers
    }
```

Replace `_token_vectors_by_layer` with:

```python
def _token_vectors_by_layer(
    *,
    lens: Any,
    unembedding_weight: torch.Tensor,
    layers: Sequence[int],
    source_token_id: int,
    target_token_id: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    source_vectors = _single_token_vectors_by_layer(
        lens=lens,
        unembedding_weight=unembedding_weight,
        layers=layers,
        token_id=source_token_id,
    )
    target_vectors = _single_token_vectors_by_layer(
        lens=lens,
        unembedding_weight=unembedding_weight,
        layers=layers,
        token_id=target_token_id,
    )
    return {
        layer: (source_vectors[layer], target_vectors[layer]) for layer in layers
    }
```

- [ ] **Step 4: Cache target vectors in the random-target loop**

Replace the inner vector construction in `readout_controls.py` with:

```python
    random_target_results = []
    for selected in selected_targets:
        target_vectors_by_layer = interventions._single_token_vectors_by_layer(
            lens=lens,
            unembedding_weight=unembedding_weight,
            layers=layers,
            token_id=selected["token_id"],
        )
        target_cases = []
        for context in contexts:
            vectors_by_layer = {
                layer: (
                    context.real_vectors_by_layer[layer][0],
                    target_vectors_by_layer[layer],
                )
                for layer in layers
            }
            intervened_logits = interventions.execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=vectors_by_layer,
                alpha=1.0,
            )
            target_cases.append(
                interventions._rank_gain_payload(context, intervened_logits)
            )
            del intervened_logits, vectors_by_layer
        require_exact_cases(target_cases)
        random_target_results.append(
            {
                **selected,
                "cases": target_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in target_cases]
                ),
            }
        )
```

Do not change loop ordering, selected targets, forward calls, rank scoring, or output assembly.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_readout_sanity.py tests/test_sanity_controls.py -q
```

Expected: all tests pass, including both integration-test parametrizations with exactly 26 vector constructions.

- [ ] **Step 6: Commit the P2 fix**

```bash
git add src/jlens_reasoning/experiments/intervention_utils.py src/jlens_reasoning/experiments/readout_controls.py tests/test_readout_sanity.py
git commit -m "perf: cache random target lens vectors"
```

## Task 2: Extract Readout Constants with Compatibility Re-exports

**Files:**

- Create: `tests/test_experiment_constants.py`
- Create: `src/jlens_reasoning/experiments/readout_constants.py`
- Modify: `src/jlens_reasoning/experiments/readout_sanity.py:1-100,175-185,225-240`
- Modify: `src/jlens_reasoning/experiments/readout_utils.py:1-20,91-120,150-160`

- [ ] **Step 1: Add failing readout constants ownership tests**

Create `tests/test_experiment_constants.py` with:

```python
import inspect

from jlens_reasoning.experiments import (
    readout_constants,
    readout_sanity,
    readout_utils,
)


def test_readout_constants_have_focused_ownership_and_legacy_aliases() -> None:
    expected = {
        "MODEL_NAME": "Qwen/Qwen3.5-4B",
        "LENS_REPO": "neuronpedia/jacobian-lens",
        "LENS_REVISION": "qwen-n1000",
        "LENS_FILE": (
            "qwen3.5-4b/jlens/Salesforce-wikitext/"
            "Qwen3.5-4B_jacobian_lens_n1000.pt"
        ),
        "TOP_K": 25,
        "WORKSPACE_LAYER_LOWER_FRACTION": 0.35,
        "WORKSPACE_LAYER_UPPER_FRACTION": 0.80,
        "DEFAULT_INTERVENTION_STRENGTHS": (1.0, 2.0),
        "DEFAULT_MINIMUM_IMPROVEMENTS": 3,
        "DEFAULT_MAX_FORMATTING_TOKENS": 2,
        "SPIDER_READ_MAX_RANK": 5,
    }
    for name, value in expected.items():
        assert getattr(readout_constants, name) == value

    for name in ("MODEL_NAME", "LENS_REPO", "LENS_REVISION", "LENS_FILE", "TOP_K"):
        assert getattr(readout_sanity, name) is getattr(readout_constants, name)
    assert readout_utils.TOP_K is readout_constants.TOP_K

    run_defaults = inspect.signature(readout_sanity.run_readout_sanity).parameters
    assert (
        run_defaults["alphas"].default
        is readout_constants.DEFAULT_INTERVENTION_STRENGTHS
    )
    assert (
        run_defaults["minimum_improvements"].default
        is readout_constants.DEFAULT_MINIMUM_IMPROVEMENTS
    )
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_experiment_constants.py -q
```

Expected: test collection fails because `readout_constants` does not exist.

- [ ] **Step 3: Create the readout constants module**

Create `src/jlens_reasoning/experiments/readout_constants.py`:

```python
"""Artifact coordinates and policy constants for J-Lens readout experiments."""

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)

TOP_K = 25
WORKSPACE_LAYER_LOWER_FRACTION = 0.35
WORKSPACE_LAYER_UPPER_FRACTION = 0.80
DEFAULT_INTERVENTION_STRENGTHS = (1.0, 2.0)
DEFAULT_MINIMUM_IMPROVEMENTS = 3
DEFAULT_MAX_FORMATTING_TOKENS = 2
SPIDER_READ_MAX_RANK = 5
```

- [ ] **Step 4: Consume readout constants from their owner**

In `readout_utils.py`, remove the local `TOP_K = 25` assignment and import:

```python
from jlens_reasoning.experiments.readout_constants import (
    DEFAULT_MAX_FORMATTING_TOKENS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    TOP_K,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
)
```

Use `DEFAULT_MAX_FORMATTING_TOKENS` as the default for `prepare_scoring_input`,
`DEFAULT_MINIMUM_IMPROVEMENTS` as the default for
`aggregate_capability_checks`, and the two named workspace fractions in
`workspace_layers`:

```python
def workspace_layers(n_layers: int, source_layers: Iterable[int]) -> list[int]:
    lower = math.ceil(WORKSPACE_LAYER_LOWER_FRACTION * n_layers)
    upper = math.floor(WORKSPACE_LAYER_UPPER_FRACTION * n_layers)
    return [layer for layer in source_layers if lower <= layer <= upper]
```

In `readout_sanity.py`, remove the four local artifact assignments and import:

```python
from jlens_reasoning.experiments.readout_constants import (
    DEFAULT_INTERVENTION_STRENGTHS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    SPIDER_READ_MAX_RANK,
    TOP_K,
)
```

Remove `TOP_K` from the `readout_utils` import block. Replace the spider rank
literal with `SPIDER_READ_MAX_RANK`, and update the runner defaults:

```python
    alphas: Sequence[float] = DEFAULT_INTERVENTION_STRENGTHS,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
```

These direct imports preserve `readout_sanity.MODEL_NAME`, `LENS_REPO`,
`LENS_REVISION`, `LENS_FILE`, and `TOP_K`. `readout_utils.TOP_K` also remains a
binding to the owner module's value.

- [ ] **Step 5: Run readout constants and focused behavior tests**

Run:

```bash
.venv/bin/pytest tests/test_experiment_constants.py tests/test_readout_sanity.py tests/test_notebooks.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit readout constants extraction**

```bash
git add src/jlens_reasoning/experiments/readout_constants.py src/jlens_reasoning/experiments/readout_sanity.py src/jlens_reasoning/experiments/readout_utils.py tests/test_experiment_constants.py
git commit -m "refactor: centralize readout constants"
```

## Task 3: Extract Sanity-Control Constants with Compatibility Re-exports

**Files:**

- Modify: `tests/test_experiment_constants.py`
- Create: `src/jlens_reasoning/experiments/sanity_constants.py`
- Modify: `src/jlens_reasoning/experiments/sanity_controls.py:1-55,90-155,300-325`
- Modify: `src/jlens_reasoning/experiments/intervention_utils.py:20-35`
- Modify: `src/jlens_reasoning/experiments/readout_controls.py:15-50,45-333`

- [ ] **Step 1: Add failing sanity constants ownership tests**

Append to `tests/test_experiment_constants.py`:

```python
from jlens_reasoning.experiments import sanity_constants, sanity_controls


def test_sanity_constants_have_focused_ownership_and_legacy_aliases() -> None:
    assert len(sanity_constants.CONTROL_SEEDS) == 16
    assert sanity_constants.CONTROL_CASE_KEYS == (
        "spider",
        "france_capital",
        "france_language",
        "france_continent",
        "france_currency",
    )
    assert sanity_constants.IDENTITY_ATOL == 1e-6
    assert sanity_constants.IDENTITY_RTOL == 1e-5
    assert sanity_constants.NORM_ATOL == 1e-6
    assert sanity_constants.NORM_RTOL == 1e-5
    assert sanity_constants.LOW_PRECISION_NORM_ATOL == 1e-2
    assert sanity_constants.LOW_PRECISION_NORM_RTOL == 1e-2
    assert sanity_constants.PERCENTILE_QUANTILE == 0.95
    assert sanity_constants.CONTROL_ALPHA == 1.0
    assert sanity_constants.WRONG_CONCEPT_REQUIRED_CASE_WINS == 4
    assert sanity_constants.MAX_RANDOM_VECTOR_ATTEMPTS == 1024

    legacy_names = (
        "CONTROL_SEEDS",
        "CONTROL_CASE_KEYS",
        "IDENTITY_ATOL",
        "IDENTITY_RTOL",
        "NORM_ATOL",
        "NORM_RTOL",
        "PERCENTILE_QUANTILE",
        "PERCENTILE_INTERPRETATION",
        "CONTROL_CHECK_MAP",
    )
    for name in legacy_names:
        assert getattr(sanity_controls, name) is getattr(sanity_constants, name)
```

Place the new import with the existing import group at the top of the test
file; do not leave a mid-file import after Ruff formatting.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_experiment_constants.py -q
```

Expected: test collection fails because `sanity_constants` does not exist.

- [ ] **Step 3: Create the sanity constants module**

Create `src/jlens_reasoning/experiments/sanity_constants.py`:

```python
"""Fixed policy constants for deterministic J-Lens sanity controls."""

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
CONTROL_CASE_KEYS = (
    "spider",
    "france_capital",
    "france_language",
    "france_continent",
    "france_currency",
)

CONTROL_ALPHA = 1.0
IDENTITY_ATOL = 1e-6
IDENTITY_RTOL = 1e-5
NORM_ATOL = 1e-6
NORM_RTOL = 1e-5
LOW_PRECISION_NORM_ATOL = 1e-2
LOW_PRECISION_NORM_RTOL = 1e-2
PERCENTILE_QUANTILE = 0.95
PERCENTILE_INTERPRETATION = (
    "deterministic sanity check; not statistical significance"
)
WRONG_CONCEPT_REQUIRED_CASE_WINS = 4
MAX_RANDOM_VECTOR_ATTEMPTS = 1024

CONTROL_CHECK_MAP = (
    ("identity", "identity_control"),
    ("matched_random_vector", "matched_random_vector_control"),
    ("wrong_concept", "wrong_concept_control"),
    ("random_target", "random_target_control"),
)
```

- [ ] **Step 4: Consume and re-export sanity constants**

Delete the local uppercase assignments from `sanity_controls.py`. Import the
constants it uses normally, and explicitly alias compatibility-only exports:

```python
from jlens_reasoning.experiments.sanity_constants import (
    CONTROL_CASE_KEYS,
    CONTROL_CHECK_MAP,
    LOW_PRECISION_NORM_ATOL,
    LOW_PRECISION_NORM_RTOL,
    MAX_RANDOM_VECTOR_ATTEMPTS,
    NORM_ATOL,
    NORM_RTOL,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
)
from jlens_reasoning.experiments.sanity_constants import (
    CONTROL_SEEDS as CONTROL_SEEDS,
)
from jlens_reasoning.experiments.sanity_constants import (
    IDENTITY_ATOL as IDENTITY_ATOL,
)
from jlens_reasoning.experiments.sanity_constants import (
    IDENTITY_RTOL as IDENTITY_RTOL,
)
```

Update `_norm_tolerances`:

```python
def _norm_tolerances(dtype: torch.dtype) -> tuple[float, float]:
    if dtype in {torch.float16, torch.bfloat16}:
        return LOW_PRECISION_NORM_ATOL, LOW_PRECISION_NORM_RTOL
    return NORM_ATOL, NORM_RTOL
```

Replace `range(1024)` with `range(MAX_RANDOM_VECTOR_ATTEMPTS)` and include the
constant in the error text with an f-string.

In `intervention_utils.py`, import `IDENTITY_ATOL` and `IDENTITY_RTOL` from
`sanity_constants`; continue importing `log_rank_gain` from `sanity_controls`.

In `readout_controls.py`, import these policy constants from
`sanity_constants`:

```python
from jlens_reasoning.experiments.sanity_constants import (
    CONTROL_ALPHA,
    CONTROL_CASE_KEYS,
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    LOW_PRECISION_NORM_ATOL,
    LOW_PRECISION_NORM_RTOL,
    NORM_ATOL,
    NORM_RTOL,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
    WRONG_CONCEPT_REQUIRED_CASE_WINS,
)
```

Keep only functions imported from `sanity_controls`. Replace all control
execution/configuration `1.0` literals with `CONTROL_ALPHA`, pass
`required_winning_case_count=WRONG_CONCEPT_REQUIRED_CASE_WINS` to
`summarize_wrong_concept`, report that same value in `thresholds`, and use the
named low-precision tolerance constants in the result payload.

- [ ] **Step 5: Run constants and control tests**

Run:

```bash
.venv/bin/pytest tests/test_experiment_constants.py tests/test_sanity_controls.py tests/test_readout_sanity.py -q
```

Expected: all tests pass with unchanged result schemas and compatibility
imports.

- [ ] **Step 6: Commit sanity constants extraction**

```bash
git add src/jlens_reasoning/experiments/sanity_constants.py src/jlens_reasoning/experiments/sanity_controls.py src/jlens_reasoning/experiments/intervention_utils.py src/jlens_reasoning/experiments/readout_controls.py tests/test_experiment_constants.py
git commit -m "refactor: centralize sanity control constants"
```

## Task 4: Full Verification and Scope Audit

**Files:**

- Verify only; modify change-caused failures if any.

- [ ] **Step 1: Run focused experiment tests**

```bash
.venv/bin/pytest tests/test_experiment_constants.py tests/test_sanity_controls.py tests/test_readout_sanity.py tests/test_readout_module_structure.py tests/test_notebooks.py -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full suite and static checks**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
git diff --check c5b5116296210cf4b304531838614c7e66665437..HEAD
```

Expected: all tests pass, Ruff reports no lint or formatting issues, and Git
reports no whitespace errors.

- [ ] **Step 3: Audit compatibility and requested scope**

Run:

```bash
.venv/bin/python - <<'PY'
from jlens_reasoning.experiments import (
    readout_constants,
    readout_sanity,
    readout_utils,
    sanity_constants,
    sanity_controls,
)

assert readout_sanity.MODEL_NAME is readout_constants.MODEL_NAME
assert readout_sanity.TOP_K is readout_constants.TOP_K
assert readout_utils.TOP_K is readout_constants.TOP_K
assert sanity_controls.CONTROL_SEEDS is sanity_constants.CONTROL_SEEDS
assert sanity_controls.CONTROL_CASE_KEYS is sanity_constants.CONTROL_CASE_KEYS
print("compatibility aliases verified")
PY
git status --short
git diff 9ead1db..HEAD -- tests/test_readout_module_structure.py
```

Expected: the alias script succeeds; `README.md` remains the only unrelated
unstaged change; the constants work does not implement the separate P3 facade
test expansion.

- [ ] **Step 4: Review the final diff**

```bash
git diff --stat c5b5116296210cf4b304531838614c7e66665437..HEAD
git log --oneline -8
```

Confirm the final commits contain only the P2 optimization, focused constants
ownership, compatibility re-exports, tests, design, and plan documentation.
