# Notebook-Owned Experiment Cases Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the J-Lens notebook the only production source of its readout and swap cases while deriving control metadata from the cases passed into the runner.

**Architecture:** Add a typed, data-only `define-cases` notebook cell and require the notebook to pass both tuples into `run_readout_sanity`. Remove case objects and fixed case keys from `constants.py`; validate the five-case contract in the runner and derive ordered keys, exclusion surfaces, and wrong-concept references from resolved contexts inside experiment-local controls.

**Tech Stack:** Python 3.11, Jupyter/nbformat, PyTorch, pytest, Ruff, setuptools, uv.

**Design reference:** `docs/superpowers/specs/2026-07-22-notebook-owned-experiment-cases-design.md`

---

## File Structure

### Create

- `tests/experiments/jlens_readout_sanity/case_fixtures.py`: loads and executes only the notebook's data-only `define-cases` cell so Python tests consume the same authoritative tuples without duplicating them.

### Modify

- `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`: add the case-definition cell, remove the case import from constants, and pass both tuples to the runner.
- `experiments/jlens_readout_sanity/constants.py`: remove case definitions and fixed case keys; add the fixed required case count.
- `experiments/jlens_readout_sanity/runner.py`: require explicit cases and validate count, uniqueness, and ordered key equality before resolving token surfaces.
- `experiments/jlens_readout_sanity/controls.py`: derive keys, exclusions, and wrong-concept references from resolved contexts.
- `tests/test_notebooks.py`: execute and validate the notebook case cell and its explicit runner call.
- `tests/experiments/jlens_readout_sanity/test_constants.py`: assert policy only, including absence of notebook-owned cases.
- `tests/experiments/jlens_readout_sanity/test_runner.py`: consume notebook cases and cover the explicit runner contract.
- `tests/experiments/jlens_readout_sanity/test_controls.py`: remove fixed-key aliases and prove derivation with synthetic keys/directions.

No registry, compatibility module, or new production cases module is created.

---

## Task 1: Establish the notebook as the tested case source

**Files:**

- Modify: `tests/test_notebooks.py`
- Modify: `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`

- [ ] **Step 1: Write the failing notebook case-cell test**

Add these helpers and the exact case assertion to `tests/test_notebooks.py`:

```python
def notebook_cells_by_id() -> dict[str, str]:
    notebook = load_notebook(
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")
    )
    return {cell.id: cell.source for cell in notebook.cells}


def execute_notebook_case_cell() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(notebook_cells_by_id()["define-cases"], namespace)
    return namespace


def test_readout_cases_are_defined_visibly_in_the_notebook() -> None:
    namespace = execute_notebook_case_cell()
    readout_cases = namespace["READOUT_CASES"]
    swap_cases = namespace["SWAP_CASES"]

    assert [(case.key, case.prompt) for case in readout_cases] == [
        ("spider", "The number of legs on the animal that spins webs is"),
        ("france_capital", "The capital of France is the city of"),
        ("france_language", "Most people in France speak"),
        ("france_continent", "France is a country on the continent of"),
        (
            "france_currency",
            "The single-word name for the currency now used in France is the",
        ),
    ]
    assert [
        (case.key, case.source_surface, case.target_surface, case.target_answers)
        for case in swap_cases
    ] == [
        ("spider", " spider", " ant", ("6", "six")),
        ("france_capital", " France", " China", ("Beijing",)),
        ("france_language", " France", " China", ("Chinese",)),
        ("france_continent", " France", " China", ("Asia",)),
        ("france_currency", " France", " China", ("Yuan",)),
    ]
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_notebooks.py::test_readout_cases_are_defined_visibly_in_the_notebook -q
```

Expected: failure with `KeyError: 'define-cases'` because the cell does not exist.

- [ ] **Step 3: Add the data-only notebook cell**

Insert a code cell with ID `define-cases` after `initialize-colab` and before `load-model-and-lens`. Its complete source is:

```python
from experiments.jlens_readout_sanity.types import ReadoutCase, SwapCase

READOUT_CASES = (
    ReadoutCase(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    ),
    ReadoutCase(
        key="france_capital",
        prompt="The capital of France is the city of",
        expected_answers=("Paris",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_language",
        prompt="Most people in France speak",
        expected_answers=("French",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_continent",
        prompt="France is a country on the continent of",
        expected_answers=("Europe",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_currency",
        prompt="The single-word name for the currency now used in France is the",
        expected_answers=("Euro",),
        target_concepts=("France",),
        literal_argument="France",
    ),
)

SWAP_CASES = (
    SwapCase("spider", " spider", " ant", ("6", "six")),
    SwapCase("france_capital", " France", " China", ("Beijing",)),
    SwapCase("france_language", " France", " China", ("Chinese",)),
    SwapCase("france_continent", " France", " China", ("Asia",)),
    SwapCase("france_currency", " France", " China", ("Yuan",)),
)
```

Keep `execution_count` null and `outputs` empty. At this intermediate checkpoint the model-loading cell still imports the same names from constants; Task 4 removes that import after all Python consumers are migrated.

- [ ] **Step 4: Run notebook tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_notebooks.py -q
.venv/bin/ruff check tests/test_notebooks.py experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb
.venv/bin/ruff format --check tests/test_notebooks.py experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb
```

Expected: all notebook tests pass and Ruff reports both files formatted.

- [ ] **Step 5: Commit Task 1**

```bash
git add tests/test_notebooks.py experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb
git commit -m "docs: define J-Lens experiment cases in notebook"
```

## Task 2: Require and validate notebook-supplied cases

**Files:**

- Modify: `experiments/jlens_readout_sanity/constants.py`
- Modify: `experiments/jlens_readout_sanity/runner.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_constants.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_runner.py`

- [ ] **Step 1: Write failing policy and runner-contract tests**

Add `CONTROL_REQUIRED_CASE_COUNT` to the constants imports in `test_constants.py` and assert:

```python
assert CONTROL_REQUIRED_CASE_COUNT == 5
```

Add `import inspect` and these tests to `test_runner.py`:

```python
def test_run_requires_explicit_case_tuples() -> None:
    parameters = inspect.signature(run_readout_sanity).parameters
    assert parameters["cases"].default is inspect.Parameter.empty
    assert parameters["swap_cases"].default is inspect.Parameter.empty


def test_case_configuration_requires_five_ordered_matching_keys() -> None:
    valid_reads = tuple(READOUT_CASES)
    valid_swaps = tuple(SWAP_CASES)

    _validate_case_configuration(valid_reads, valid_swaps)

    malformed = (
        (valid_reads[:-1], valid_swaps[:-1]),
        (valid_reads, tuple(reversed(valid_swaps))),
        (valid_reads, (*valid_swaps[:-1], valid_swaps[0])),
    )
    for read_cases, swap_cases in malformed:
        with pytest.raises(ValueError, match="five|unique|same keys in the same order"):
            _validate_case_configuration(read_cases, swap_cases)
```

Import `_validate_case_configuration` from `runner.py` for this focused contract test.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_constants.py tests/experiments/jlens_readout_sanity/test_runner.py::test_run_requires_explicit_case_tuples tests/experiments/jlens_readout_sanity/test_runner.py::test_case_configuration_requires_five_ordered_matching_keys -q
```

Expected: collection fails because `CONTROL_REQUIRED_CASE_COUNT` and `_validate_case_configuration` do not exist, and runner cases still have defaults.

- [ ] **Step 3: Add the fixed count and runner validation**

Add to `constants.py` beside the other control policy:

```python
CONTROL_REQUIRED_CASE_COUNT = 5
```

Import it in `runner.py` and add:

```python
def _validate_case_configuration(
    cases: Sequence[ReadoutCase],
    swap_cases: Sequence[SwapCase],
) -> None:
    read_keys = [case.key for case in cases]
    swap_keys = [case.key for case in swap_cases]
    if len(read_keys) != CONTROL_REQUIRED_CASE_COUNT or len(swap_keys) != (
        CONTROL_REQUIRED_CASE_COUNT
    ):
        raise ValueError(
            "Negative controls require exactly "
            f"{CONTROL_REQUIRED_CASE_COUNT} readout and swap cases"
        )
    if len(set(read_keys)) != len(read_keys):
        raise ValueError("Readout case keys must be unique")
    if len(set(swap_keys)) != len(swap_keys):
        raise ValueError("Swap case keys must be unique")
    if read_keys != swap_keys:
        raise ValueError("Readout and swap cases must have the same keys in the same order")
```

Change the runner signature to:

```python
def run_readout_sanity(
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    cases: Sequence[ReadoutCase],
    swap_cases: Sequence[SwapCase],
    alphas: Sequence[float] = DEFAULT_INTERVENTION_STRENGTHS,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
    top_k: int = TOP_K,
) -> dict[str, Any]:
```

At the start of the body, after `validate_model_lens`, call:

```python
_validate_case_configuration(cases, swap_cases)
```

Remove the tuple-equality check against module constants. Keep `resolve_swap_cases` immediately after structural validation so strict token surfaces still fail before lens forwards.

- [ ] **Step 4: Update existing runner calls with explicit tuples**

Every test invocation intended to reach experiment execution must pass:

```python
cases=READOUT_CASES,
swap_cases=SWAP_CASES,
```

Tests intentionally supplying malformed tuples keep their explicit custom arguments.

- [ ] **Step 5: Run runner and constants tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_constants.py tests/experiments/jlens_readout_sanity/test_runner.py -q
```

Expected: all focused tests pass; malformed case sets fail before forwards.

- [ ] **Step 6: Commit Task 2**

```bash
git add experiments/jlens_readout_sanity/constants.py experiments/jlens_readout_sanity/runner.py tests/experiments/jlens_readout_sanity/test_constants.py tests/experiments/jlens_readout_sanity/test_runner.py
git commit -m "refactor: require explicit experiment cases"
```

## Task 3: Derive control metadata and mismatch references

**Files:**

- Modify: `experiments/jlens_readout_sanity/controls.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_controls.py`

- [ ] **Step 1: Write failing derivation tests with arbitrary keys**

Import `_wrong_reference_contexts` and add a lightweight context factory to `test_controls.py`:

```python
def _direction_context(key: str, source_id: int, target_id: int):
    return SimpleNamespace(
        resolved=SimpleNamespace(
            case=SimpleNamespace(key=key),
            source=SimpleNamespace(token_id=source_id),
            target=SimpleNamespace(token_id=target_id),
        )
    )


def test_wrong_concept_references_derive_from_directions_not_case_names() -> None:
    first = _direction_context("alpha", 1, 2)
    second = _direction_context("beta", 3, 4)
    third = _direction_context("gamma", 3, 4)

    references = _wrong_reference_contexts((first, second, third))

    assert references == (second, first, first)


def test_wrong_concept_requires_two_distinct_directions() -> None:
    contexts = (
        _direction_context("alpha", 1, 2),
        _direction_context("beta", 1, 2),
    )
    with pytest.raises(ValueError, match="two distinct swap directions"):
        _wrong_reference_contexts(contexts)
```

Change the exact-key test to call the local wrapper with explicit caller-owned keys:

```python
require_exact_cases(results, expected_keys=("a", "b"))
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_controls.py::test_wrong_concept_references_derive_from_directions_not_case_names tests/experiments/jlens_readout_sanity/test_controls.py::test_wrong_concept_requires_two_distinct_directions -q
```

Expected: collection fails because `_wrong_reference_contexts` does not exist.

- [ ] **Step 3: Make exact-key validation caller-owned**

Replace the fixed wrapper with:

```python
def require_exact_cases(
    results: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
) -> None:
    require_exact_case_keys(results, expected_keys=expected_keys)
```

Update `summarize_wrong_concept` to accept `expected_keys` as a required keyword argument and pass it to both validation calls:

```python
def summarize_wrong_concept(
    matched_cases: Sequence[Mapping[str, Any]],
    mismatched_cases: Sequence[Mapping[str, Any]],
    *,
    expected_keys: Sequence[str],
    required_winning_case_count: int = WRONG_CONCEPT_REQUIRED_CASE_WINS,
) -> dict[str, Any]:
```

Update its existing tests to pass `expected_keys=EXPECTED_CASE_KEYS`.

- [ ] **Step 4: Implement derived mismatch references**

Add:

```python
def _wrong_reference_contexts(
    contexts: Sequence[InterventionContext],
) -> tuple[InterventionContext, ...]:
    references = []
    for context in contexts:
        direction = (
            context.resolved.source.token_id,
            context.resolved.target.token_id,
        )
        reference = next(
            (
                candidate
                for candidate in contexts
                if (
                    candidate.resolved.source.token_id,
                    candidate.resolved.target.token_id,
                )
                != direction
            ),
            None,
        )
        if reference is None:
            raise ValueError(
                "Wrong-concept control requires at least two distinct swap directions"
            )
        references.append(reference)
    return tuple(references)
```

- [ ] **Step 5: Derive all control data at the top of `run_negative_controls`**

Import `CONTROL_REQUIRED_CASE_COUNT` and remove imports of `CONTROL_CASE_KEYS`, `READOUT_CASES`, and `SWAP_CASES`.

Start the function with:

```python
if len(contexts) != CONTROL_REQUIRED_CASE_COUNT:
    raise ValueError(
        f"Negative controls require exactly {CONTROL_REQUIRED_CASE_COUNT} cases"
    )
expected_keys = tuple(context.resolved.case.key for context in contexts)
wrong_references = _wrong_reference_contexts(contexts)
require_exact_cases(swap_results, expected_keys=expected_keys)
```

Pass `expected_keys` to every `require_exact_cases` and
`summarize_wrong_concept` call. Replace the hardcoded spider/France lookup and
loop with:

```python
for context, wrong_reference in zip(contexts, wrong_references, strict=True):
```

Derive exclusion surfaces from contexts:

```python
source_surfaces = tuple(
    surface
    for context in contexts
    for surface in concept_surfaces(
        context.resolved.case.source_surface.strip()
    )
)
target_surfaces = tuple(
    surface
    for context in contexts
    for surface in concept_surfaces(
        context.resolved.case.target_surface.strip()
    )
)
clean_answer_surfaces = tuple(
    surface
    for context in contexts
    for answer in context.resolved.read_case.expected_answers
    for surface in concept_surfaces(answer)
)
intended_answer_surfaces = tuple(
    surface
    for context in contexts
    for answer in context.resolved.case.target_answers
    for surface in concept_surfaces(answer)
)
```

Change the result definition from the fixed wording `exactly five cases` to:

```python
"aggregate": f"arithmetic mean across exactly {len(expected_keys)} cases",
```

- [ ] **Step 6: Run control and integration tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_controls.py tests/experiments/jlens_readout_sanity/test_runner.py -q
```

Expected: all tests pass, including the complete integration's deterministic IDs, 180 interventions, result schema, and four global control gates.

- [ ] **Step 7: Commit Task 3**

```bash
git add experiments/jlens_readout_sanity/controls.py tests/experiments/jlens_readout_sanity/test_controls.py
git commit -m "refactor: derive controls from supplied cases"
```

## Task 4: Remove Python case definitions and wire the notebook explicitly

**Files:**

- Create: `tests/experiments/jlens_readout_sanity/case_fixtures.py`
- Modify: `experiments/jlens_readout_sanity/constants.py`
- Modify: `experiments/jlens_readout_sanity/runner.py`
- Modify: `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`
- Modify: `tests/test_notebooks.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_constants.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_runner.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_controls.py`

- [ ] **Step 1: Write failing ownership assertions**

Import the constants module in `test_constants.py`:

```python
import experiments.jlens_readout_sanity.constants as constants_module
```

Remove direct case imports and replace their tests with:

```python
def test_case_definitions_are_not_owned_by_constants() -> None:
    assert not hasattr(constants_module, "READOUT_CASES")
    assert not hasattr(constants_module, "SWAP_CASES")
    assert not hasattr(constants_module, "CONTROL_CASE_KEYS")
```

Extend the notebook workflow test:

```python
cells_by_id = notebook_cells_by_id()
assert "READOUT_CASES" not in cells_by_id["load-model-and-lens"]
assert "cases=READOUT_CASES" in cells_by_id["run-experiment"]
assert "swap_cases=SWAP_CASES" in cells_by_id["run-experiment"]
```

- [ ] **Step 2: Run ownership tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/experiments/jlens_readout_sanity/test_constants.py::test_case_definitions_are_not_owned_by_constants tests/test_notebooks.py::test_readout_sanity_notebook_has_pinned_gpu_workflow -q
```

Expected: constants still exposes all three fixed case names and the run cell does not pass the tuples.

- [ ] **Step 3: Create the test fixture loader without duplicating cases**

Create `case_fixtures.py`:

```python
from pathlib import Path
from typing import Any

import nbformat

NOTEBOOK = Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb")


def _load_notebook_cases() -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    case_cell = next(cell for cell in notebook.cells if cell.id == "define-cases")
    namespace: dict[str, Any] = {}
    exec(case_cell.source, namespace)
    return namespace["READOUT_CASES"], namespace["SWAP_CASES"]


READOUT_CASES, SWAP_CASES = _load_notebook_cases()
```

In `test_runner.py`, remove `READOUT_CASES` and `SWAP_CASES` from constants imports and import them from this fixture module:

```python
from tests.experiments.jlens_readout_sanity.case_fixtures import (
    READOUT_CASES,
    SWAP_CASES,
)
```

In `test_controls.py`, remove `CONTROL_CASE_KEYS` from constants imports and compare against its local `EXPECTED_CASE_KEYS` wherever fixed output ordering is asserted.

- [ ] **Step 4: Remove case ownership from constants and runner imports**

Delete from `constants.py`:

```python
from experiments.jlens_readout_sanity.types import ReadoutCase, SwapCase
```

Delete the complete `READOUT_CASES` and `SWAP_CASES` tuple definitions and:

```python
CONTROL_CASE_KEYS = tuple(case.key for case in SWAP_CASES)
```

Remove `READOUT_CASES` and `SWAP_CASES` from the constants import block in `runner.py`. The required runner parameters added in Task 2 remain unchanged.

- [ ] **Step 5: Update notebook imports and execution**

Remove `READOUT_CASES` from the constants import in `load-model-and-lens`.

Add to the `run_readout_sanity` call in `run-experiment`:

```python
cases=READOUT_CASES,
swap_cases=SWAP_CASES,
```

Do not move the case definitions into the run cell; `define-cases` remains independently rerunnable.

- [ ] **Step 6: Run focused ownership and behavior tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_notebooks.py tests/experiments/jlens_readout_sanity/test_constants.py tests/experiments/jlens_readout_sanity/test_runner.py tests/experiments/jlens_readout_sanity/test_controls.py -q
.venv/bin/ruff check experiments/jlens_readout_sanity tests/test_notebooks.py tests/experiments/jlens_readout_sanity
.venv/bin/ruff format --check experiments/jlens_readout_sanity tests/test_notebooks.py tests/experiments/jlens_readout_sanity
```

Expected: all focused tests pass and Ruff is clean.

- [ ] **Step 7: Verify there is one production definition source**

Run:

```bash
rg -n '^(\s*"?READOUT_CASES|\s*"?SWAP_CASES)\s*=' experiments src
```

Expected: the only matches are the two source lines serialized inside
`experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`; no `.py` file
defines either tuple.

- [ ] **Step 8: Commit Task 4**

```bash
git add experiments/jlens_readout_sanity/constants.py experiments/jlens_readout_sanity/runner.py experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb tests/test_notebooks.py tests/experiments/jlens_readout_sanity/case_fixtures.py tests/experiments/jlens_readout_sanity/test_constants.py tests/experiments/jlens_readout_sanity/test_runner.py tests/experiments/jlens_readout_sanity/test_controls.py
git commit -m "refactor: make notebook the experiment case source"
```

## Task 5: Complete repository and installed-wheel verification

**Files:**

- Verify all changed files.
- Modify only files required to fix migration-caused failures.

- [ ] **Step 1: Run focused tests**

```bash
.venv/bin/pytest tests/test_notebooks.py tests/test_package_discovery.py tests/test_imports.py tests/experiments_utils tests/experiments/jlens_readout_sanity -q
```

Expected: all focused tests pass.

- [ ] **Step 2: Run the full suite and Ruff**

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
```

Expected: pytest, lint, formatting, and whitespace checks all exit zero.

- [ ] **Step 3: Build and verify the wheel outside the checkout**

Run in one shell from the worktree:

```bash
JLENS_CASES_ROOT="$PWD"
JLENS_CASES_WHEELS="$(mktemp -d)"
JLENS_CASES_TARGET="$(mktemp -d)"
uv build --quiet --wheel --clear --out-dir "$JLENS_CASES_WHEELS" .
JLENS_CASES_WHEEL="$(find "$JLENS_CASES_WHEELS" -maxdepth 1 -type f -name '*.whl' -print -quit)"
test -n "$JLENS_CASES_WHEEL"
uv pip install --quiet --python "$JLENS_CASES_ROOT/.venv/bin/python" --no-deps --target "$JLENS_CASES_TARGET" "$JLENS_CASES_WHEEL"
cd /private/tmp
JLENS_CASES_TARGET="$JLENS_CASES_TARGET" "$JLENS_CASES_ROOT/.venv/bin/python" -c "import os,sys; target=os.environ['JLENS_CASES_TARGET']; sys.path.insert(0,target); import experiments.jlens_readout_sanity.utils as e; import jlens_reasoning.experiments_utils.controls as c; assert e.__file__.startswith(target); assert c.__file__.startswith(target)"
cd "$JLENS_CASES_ROOT"
```

Expected: both package roots import from the temporary target. The notebook is a repository-owned Colab artifact and is not required as wheel package data.

- [ ] **Step 4: Inspect final scope and preserve the user README edit**

```bash
git status --short
git diff -- README.md
rg -n 'CONTROL_CASE_KEYS|from .*constants import .*READOUT_CASES|from .*constants import .*SWAP_CASES' experiments src tests
```

Expected: the only worktree change outside committed implementation is the pre-existing unstaged README evaluation-policy paragraph; ripgrep finds no fixed control key or constants-owned case imports.

- [ ] **Step 5: Request code review**

Invoke `superpowers:requesting-code-review` for the range from the design commit to the final implementation head. Address every Critical or Important finding, then repeat Steps 1-4.
