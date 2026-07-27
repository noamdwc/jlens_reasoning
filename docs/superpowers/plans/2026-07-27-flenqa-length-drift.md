# FLenQA Length-Drift Readout (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every unique FLenQA prompt through the Jacobian lens on one A100, and persist a labelled, deduplicated readout that supports offline concept-drift analysis without re-running the GPU job.

**Architecture:** A CPU build phase turns 12,000 dataset rows into ~9,862 unique prompts with conditions derived from prompt content, then a GPU run phase reads the lens at labelled anchor positions and writes atomic Parquet shards with resume. Every module is import-clean on CPU so the whole pipeline is testable without a model.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace `transformers` ≥5.5 and `datasets`, the `jlens` package (pinned git rev), PyArrow, pytest, ruff.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-27-flenqa-length-drift-design.md`. Read it before starting.
- Branch: `feature/flenqa-length-drift`. Do not commit to `main`.
- Python imports go at the **top of the file**, never inline or mid-file.
- ruff: `line-length = 88`, lint rules `["B", "E", "F", "I", "UP", "W"]`. Run `uv run ruff format .` and `uv run ruff check .` before every commit.
- All new modules start with `from __future__ import annotations`.
- Tests are CPU-only and model-free. No repository, HuggingFace, W&B, or Google Drive credentials. Follow the existing fake-tokenizer pattern in `tests/experiments_utils/test_tokens.py`.
- Run tests with `uv run pytest`.
- `max_seq_len` is **always passed explicitly** as `MAX_SEQ_LEN = 4096`. Never rely on the `jlens` default of 512.
- `positions` passed to `lens.apply()` is **never `None`**.
- The unit of analysis is `prompt_id`. `source_row_id` is provenance only — never a key, seed, or unit of analysis.

**One deliberate deviation from the spec:** placement fractions are computed from **character** offsets in the CPU build phase, not token offsets. The spec says token positions. Characters keep the build phase tokenizer-free and CI-testable, and the values are fractions, so the classification is equivalent. This is recorded in `run_meta` as `placement_basis="characters"`.

---

## File Structure

**Create — shared library (`src/jlens_reasoning/`):**

| File | Responsibility |
| --- | --- |
| `benchmarks/__init__.py` | package marker |
| `benchmarks/flenqa.py` | `FlenqaRow`, schema verification, row loading |
| `benchmarks/flenqa_prompts.py` | per-task prompt templates; `FlenqaPrompt`; dedup to `prompt_id` |
| `benchmarks/flenqa_conditions.py` | content-derived `padding_type_effective`, placement fractions, `dispersion_effective` |
| `experiments_utils/spans.py` | find-all span location; char→token conversion via offset mapping |
| `experiments_utils/storage.py` | atomic Parquet shard writer/reader; completed-id scan |

**Create — experiment (`experiments/flenqa_length_drift/`):**

| File | Responsibility |
| --- | --- |
| `__init__.py` | package marker |
| `constants.py` | model/lens coordinates, budgets, expected schema values |
| `bridges.py` | task-specific bridge-entity extraction + 300-problem gate |
| `anchors.py` | labelled anchor selection; summary position selection |
| `readout.py` | one lens pass → `topk` / `bridge` / `summary` records |
| `scoring.py` | deterministic logit score + generation extraction |
| `preflight.py` | 3000-token lens-validity gate |
| `experiment.py` | run loop, config hash, checkpoint/resume, Drive sync |
| `flenqa_smoke.ipynb` | L4 driver, ~33 prompts, measures wall-clock |
| `flenqa_length_drift.ipynb` | A100 driver, full run |

**Modify:** `pyproject.toml` (add `pyarrow`), `README.md` (document the experiment).

**Create — tests:** one file per module under `tests/benchmarks/`, `tests/experiments_utils/`, `tests/experiments/flenqa_length_drift/`.

---

### Task 1: FLenQA row model and schema verification

**Files:**
- Create: `src/jlens_reasoning/benchmarks/__init__.py`
- Create: `src/jlens_reasoning/benchmarks/flenqa.py`
- Create: `experiments/flenqa_length_drift/__init__.py`
- Create: `experiments/flenqa_length_drift/constants.py`
- Test: `tests/benchmarks/__init__.py`, `tests/benchmarks/test_flenqa.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FlenqaRow` frozen dataclass with fields `source_row_id: int`, `problem_id: int`, `sample_id: int`, `task: str`, `label: bool`, `key_texts: tuple[str, ...]`, `rule: str | None`, `question: str`, `mixin: str`, `ctx_size_declared: int`, `padding_type_declared: str`, `dispersion_declared: str`
  - `verify_schema(records: Sequence[Mapping[str, Any]]) -> None` — raises `ValueError`
  - `normalize_rows(records: Sequence[Mapping[str, Any]]) -> tuple[FlenqaRow, ...]`
  - Constants in `constants.py`: `TASKS`, `CTX_SIZES`, `PADDING_TYPES`, `DISPERSIONS`, `EXPECTED_ROW_COUNT`, `EXPECTED_PROBLEM_COUNT`, `EXPECTED_ROWS_PER_PROBLEM`, `RULETAKER_TASK`

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/__init__.py` (empty) and `tests/benchmarks/test_flenqa.py`:

```python
import pytest

from jlens_reasoning.benchmarks.flenqa import (
    FlenqaRow,
    normalize_rows,
    verify_schema,
)


def _record(**overrides: object) -> dict[str, object]:
    record = {
        "global_sample_id": 0,
        "sample_id": 0,
        "dataset": "PIR",
        "label": "True",
        "facts": ["Fact A text.", "Fact B text."],
        "rule": None,
        "statement": None,
        "assertion/question": "Is it so?",
        "mixin": "Fact A text.\nFact B text.",
        "ctx_size": 250,
        "padding_type": "books",
        "dispersion": "first",
    }
    record.update(overrides)
    return record


def test_verify_schema_rejects_unknown_padding_type() -> None:
    with pytest.raises(ValueError, match="padding_type"):
        verify_schema([_record(padding_type="duplicate")])


def test_verify_schema_rejects_unknown_dispersion() -> None:
    with pytest.raises(ValueError, match="dispersion"):
        verify_schema([_record(dispersion="scattered")])


def test_verify_schema_rejects_unknown_ctx_size() -> None:
    with pytest.raises(ValueError, match="ctx_size"):
        verify_schema([_record(ctx_size=750)])


def test_verify_schema_accepts_valid_records() -> None:
    verify_schema([_record(), _record(padding_type="same", label="False")])


def test_normalize_reads_facts_for_pir() -> None:
    (row,) = normalize_rows([_record()])

    assert row == FlenqaRow(
        source_row_id=0,
        problem_id=0,
        sample_id=0,
        task="PIR",
        label=True,
        key_texts=("Fact A text.", "Fact B text."),
        rule=None,
        question="Is it so?",
        mixin="Fact A text.\nFact B text.",
        ctx_size_declared=250,
        padding_type_declared="books",
        dispersion_declared="first",
    )


def test_normalize_reads_statement_and_rule_for_ruletaker() -> None:
    (row,) = normalize_rows(
        [
            _record(
                dataset="Simplified RuleTaker",
                facts=None,
                statement=["Dave is small.", "Dave is good."],
                rule="If X is good and X is small then X is loud.",
            )
        ]
    )

    assert row.task == "Simplified RuleTaker"
    assert row.key_texts == ("Dave is small.", "Dave is good.")
    assert row.rule == "If X is good and X is small then X is loud."


def test_normalize_assigns_sequential_source_row_ids() -> None:
    rows = normalize_rows([_record(), _record(), _record()])

    assert [row.source_row_id for row in rows] == [0, 1, 2]


def test_normalize_rejects_missing_key_texts() -> None:
    with pytest.raises(ValueError, match="key texts"):
        normalize_rows([_record(facts=None)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_flenqa.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jlens_reasoning.benchmarks'`

- [ ] **Step 3: Write the constants**

Create `experiments/flenqa_length_drift/__init__.py` (empty) and `experiments/flenqa_length_drift/constants.py`:

```python
"""Fixed policy and artifact coordinates for the FLenQA length-drift readout."""

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
MODEL_PATH = "/content/drive/MyDrive/data/jlens-reasoning/assets/models/qwen3.5-4b"
LENS_PATH = (
    "/content/drive/MyDrive/data/jlens-reasoning/assets/lenses/"
    "qwen3.5-4b/Qwen3.5-4B_jacobian_lens_n1000.pt"
)

DATASET_REPO = "alonj/FLenQA"
DATASET_SPLIT = "eval"

PIR_TASK = "PIR"
MONOREL_TASK = "MonoRel"
RULETAKER_TASK = "Simplified RuleTaker"
TASKS = (PIR_TASK, MONOREL_TASK, RULETAKER_TASK)
CTX_SIZES = (250, 500, 1000, 2000, 3000)
PADDING_TYPES = ("books", "same")
DISPERSIONS = ("first", "middle", "last", "random")
LABELS = ("True", "False")

EXPECTED_ROW_COUNT = 12000
EXPECTED_PROBLEM_COUNT = 300
EXPECTED_ROWS_PER_PROBLEM = 40

MAX_SEQ_LEN = 4096
TOP_K = 25
ANCHOR_PADDING_COUNT = 4
ANCHOR_BUDGET = 12
SUMMARY_POSITION_BUDGET = 48
KEY_SPAN_SUMMARY_CAP = 12
FINAL_POSITION_COUNT = 4
PLACEMENT_EPSILON = 0.02
SHARD_SIZE = 500
PLACEMENT_BASIS = "characters"
```

- [ ] **Step 4: Write the row model**

Create `src/jlens_reasoning/benchmarks/__init__.py` (empty) and `src/jlens_reasoning/benchmarks/flenqa.py`:

```python
"""FLenQA row loading, schema verification, and normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from experiments.flenqa_length_drift.constants import (
    CTX_SIZES,
    DISPERSIONS,
    LABELS,
    PADDING_TYPES,
    RULETAKER_TASK,
    TASKS,
)


@dataclass(frozen=True, slots=True)
class FlenqaRow:
    source_row_id: int
    problem_id: int
    sample_id: int
    task: str
    label: bool
    key_texts: tuple[str, ...]
    rule: str | None
    question: str
    mixin: str
    ctx_size_declared: int
    padding_type_declared: str
    dispersion_declared: str


_CATEGORICAL = (
    ("dataset", TASKS),
    ("ctx_size", CTX_SIZES),
    ("padding_type", PADDING_TYPES),
    ("dispersion", DISPERSIONS),
    ("label", LABELS),
)


def verify_schema(records: Sequence[Mapping[str, Any]]) -> None:
    """Assert the released FLenQA value sets, which contradict the paper."""
    for field, allowed in _CATEGORICAL:
        observed = {record[field] for record in records}
        unexpected = observed - set(allowed)
        if unexpected:
            raise ValueError(
                f"Unexpected {field} values {sorted(map(str, unexpected))}; "
                f"expected a subset of {list(allowed)}"
            )


def _key_texts(record: Mapping[str, Any]) -> tuple[str, ...]:
    source = record["statement"] if record["dataset"] == RULETAKER_TASK else record["facts"]
    if not source:
        raise ValueError(
            f"Row for task {record['dataset']!r} has no key texts; "
            "PIR/MonoRel use 'facts' and RuleTaker uses 'statement'"
        )
    return tuple(str(text) for text in source)


def normalize_rows(records: Sequence[Mapping[str, Any]]) -> tuple[FlenqaRow, ...]:
    """Convert raw dataset records into typed rows with stable provenance ids."""
    return tuple(
        FlenqaRow(
            source_row_id=index,
            problem_id=int(record["global_sample_id"]),
            sample_id=int(record["sample_id"]),
            task=str(record["dataset"]),
            label=record["label"] == "True",
            key_texts=_key_texts(record),
            rule=record["rule"] if record["rule"] else None,
            question=str(record["assertion/question"]),
            mixin=str(record["mixin"]),
            ctx_size_declared=int(record["ctx_size"]),
            padding_type_declared=str(record["padding_type"]),
            dispersion_declared=str(record["dispersion"]),
        )
        for index, record in enumerate(records)
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_flenqa.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/benchmarks experiments/flenqa_length_drift tests/benchmarks
git commit -m "feat: add FLenQA row model and schema verification"
```

---

### Task 2: Prompt construction and deduplication to prompt_id

**Files:**
- Create: `src/jlens_reasoning/benchmarks/flenqa_prompts.py`
- Test: `tests/benchmarks/test_flenqa_prompts.py`

**Interfaces:**
- Consumes: `FlenqaRow` from Task 1; `RULETAKER_TASK` from `constants`.
- Produces:
  - `build_prompt_text(row: FlenqaRow) -> str`
  - `compute_prompt_id(text: str) -> str` — 16-hex-char SHA-256 prefix
  - `FlenqaPrompt` frozen dataclass: `prompt_id: str`, `problem_id: int`, `task: str`, `text: str`, `question: str`, `key_texts: tuple[str, ...]`, `rule: str | None`, `label: bool`, `mixin: str`, `ctx_size_declared: int`, `source_row_ids: tuple[int, ...]`, `padding_type_declared: tuple[str, ...]`, `dispersion_declared: tuple[str, ...]`
  - `deduplicate(rows: Sequence[FlenqaRow]) -> tuple[FlenqaPrompt, ...]`

The RuleTaker `rule` never appears in `mixin`, so the template must inject it or the task is unanswerable.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_flenqa_prompts.py`:

```python
import pytest

from jlens_reasoning.benchmarks.flenqa import FlenqaRow
from jlens_reasoning.benchmarks.flenqa_prompts import (
    build_prompt_text,
    compute_prompt_id,
    deduplicate,
)


def _row(**overrides: object) -> FlenqaRow:
    fields = {
        "source_row_id": 0,
        "problem_id": 0,
        "sample_id": 0,
        "task": "PIR",
        "label": True,
        "key_texts": ("Fact A.", "Fact B."),
        "rule": None,
        "question": "Is it so?",
        "mixin": "Fact A.\nFact B.",
        "ctx_size_declared": 250,
        "padding_type_declared": "books",
        "dispersion_declared": "first",
    }
    fields.update(overrides)
    return FlenqaRow(**fields)


def test_pir_prompt_contains_context_and_question() -> None:
    text = build_prompt_text(_row())

    assert "Fact A.\nFact B." in text
    assert "Is it so?" in text
    assert "True or False" in text


def test_ruletaker_prompt_injects_the_rule() -> None:
    text = build_prompt_text(
        _row(
            task="Simplified RuleTaker",
            rule="If X is good then X is loud.",
            question="Dave is loud.",
        )
    )

    assert "If X is good then X is loud." in text


def test_prompt_id_is_content_determined_and_stable() -> None:
    assert compute_prompt_id("abc") == compute_prompt_id("abc")
    assert compute_prompt_id("abc") != compute_prompt_id("abd")
    assert len(compute_prompt_id("abc")) == 16


def test_identical_prompts_collapse_to_one_observation() -> None:
    rows = [
        _row(source_row_id=0, padding_type_declared="books", dispersion_declared="first"),
        _row(source_row_id=1, padding_type_declared="same", dispersion_declared="random"),
    ]

    (prompt,) = deduplicate(rows)

    assert prompt.source_row_ids == (0, 1)
    assert prompt.padding_type_declared == ("books", "same")
    assert prompt.dispersion_declared == ("first", "random")


def test_distinct_prompts_stay_separate() -> None:
    rows = [_row(source_row_id=0), _row(source_row_id=1, mixin="Other.")]

    assert len(deduplicate(rows)) == 2


def test_deduplicate_rejects_groups_mixing_label() -> None:
    rows = [_row(source_row_id=0, label=True), _row(source_row_id=1, label=False)]

    with pytest.raises(ValueError, match="label"):
        deduplicate(rows)


def test_deduplicate_rejects_groups_mixing_problem() -> None:
    rows = [_row(source_row_id=0, problem_id=0), _row(source_row_id=1, problem_id=1)]

    with pytest.raises(ValueError, match="problem_id"):
        deduplicate(rows)


def test_deduplicate_rejects_groups_mixing_ctx_size() -> None:
    rows = [
        _row(source_row_id=0, ctx_size_declared=250),
        _row(source_row_id=1, ctx_size_declared=500),
    ]

    with pytest.raises(ValueError, match="ctx_size"):
        deduplicate(rows)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_flenqa_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jlens_reasoning.benchmarks.flenqa_prompts'`

- [ ] **Step 3: Write the implementation**

Create `src/jlens_reasoning/benchmarks/flenqa_prompts.py`:

```python
"""FLenQA prompt construction and deduplication to unique observations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

from experiments.flenqa_length_drift.constants import RULETAKER_TASK
from jlens_reasoning.benchmarks.flenqa import FlenqaRow

PROMPT_ID_WIDTH = 16

_ASSERTION_TEMPLATE = (
    "{context}\n\n"
    "Rule: {rule}\n\n"
    "Based only on the text above, is the following statement True or False?\n"
    "Statement: {question}\n"
    "Answer:"
)
_QUESTION_TEMPLATE = (
    "{context}\n\n"
    "Based only on the text above, answer True or False.\n"
    "Question: {question}\n"
    "Answer:"
)


def build_prompt_text(row: FlenqaRow) -> str:
    """Render the exact prompt string sent to the model."""
    if row.task == RULETAKER_TASK:
        if not row.rule:
            raise ValueError(
                f"RuleTaker row {row.source_row_id} has no rule; the rule never "
                "appears in mixin, so the task would be unanswerable"
            )
        return _ASSERTION_TEMPLATE.format(
            context=row.mixin,
            rule=row.rule,
            question=row.question,
        )
    return _QUESTION_TEMPLATE.format(context=row.mixin, question=row.question)


def compute_prompt_id(text: str) -> str:
    """Hash the exact prompt text; identical prompts share an identifier."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:PROMPT_ID_WIDTH]


@dataclass(frozen=True, slots=True)
class FlenqaPrompt:
    prompt_id: str
    problem_id: int
    task: str
    text: str
    question: str
    key_texts: tuple[str, ...]
    rule: str | None
    label: bool
    mixin: str
    ctx_size_declared: int
    source_row_ids: tuple[int, ...]
    padding_type_declared: tuple[str, ...]
    dispersion_declared: tuple[str, ...]


def _assert_constant(group: Sequence[FlenqaRow], field: str) -> None:
    observed = {getattr(row, field) for row in group}
    if len(observed) > 1:
        raise ValueError(
            f"Prompt group mixes {field}: {sorted(map(str, observed))}; "
            "identical prompts must agree on it"
        )


def deduplicate(rows: Sequence[FlenqaRow]) -> tuple[FlenqaPrompt, ...]:
    """Collapse rows that produce byte-identical prompts into one observation."""
    grouped: dict[str, list[FlenqaRow]] = {}
    texts: dict[str, str] = {}
    for row in rows:
        text = build_prompt_text(row)
        prompt_id = compute_prompt_id(text)
        grouped.setdefault(prompt_id, []).append(row)
        texts[prompt_id] = text

    prompts: list[FlenqaPrompt] = []
    for prompt_id, group in grouped.items():
        for field in ("problem_id", "label", "ctx_size_declared", "task"):
            _assert_constant(group, field)
        first = group[0]
        prompts.append(
            FlenqaPrompt(
                prompt_id=prompt_id,
                problem_id=first.problem_id,
                task=first.task,
                text=texts[prompt_id],
                question=first.question,
                key_texts=first.key_texts,
                rule=first.rule,
                label=first.label,
                mixin=first.mixin,
                ctx_size_declared=first.ctx_size_declared,
                source_row_ids=tuple(row.source_row_id for row in group),
                padding_type_declared=tuple(
                    sorted({row.padding_type_declared for row in group})
                ),
                dispersion_declared=tuple(
                    sorted({row.dispersion_declared for row in group})
                ),
            )
        )
    return tuple(prompts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_flenqa_prompts.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/benchmarks/flenqa_prompts.py tests/benchmarks/test_flenqa_prompts.py
git commit -m "feat: build FLenQA prompts and deduplicate to prompt_id"
```

---

### Task 3: Content-derived condition variables

**Files:**
- Create: `src/jlens_reasoning/benchmarks/flenqa_conditions.py`
- Test: `tests/benchmarks/test_flenqa_conditions.py`

**Interfaces:**
- Consumes: `FlenqaPrompt` from Task 2; `PLACEMENT_EPSILON` from `constants`.
- Produces:
  - `PromptConditions` frozen dataclass: `padding_type_effective: str`, `dispersion_effective: str`, `frac_padding_before: float`, `frac_padding_between: float`, `frac_padding_after: float`, `n_padding_chars: int`
  - `derive_conditions(prompt: FlenqaPrompt) -> PromptConditions`

`padding_type_effective = "none"` is assigned **only** when the mixin contains the key texts and nothing else — never inferred from `ctx_size`. `random` is not a possible output: it is a generation procedure, not a property of an input.

- [ ] **Step 1: Write the failing test**

Create `tests/benchmarks/test_flenqa_conditions.py`:

```python
from jlens_reasoning.benchmarks.flenqa_conditions import derive_conditions
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt


def _prompt(mixin: str, **overrides: object) -> FlenqaPrompt:
    fields = {
        "prompt_id": "0" * 16,
        "problem_id": 0,
        "task": "PIR",
        "text": "unused",
        "question": "Is it so?",
        "key_texts": ("AAAA", "BBBB"),
        "rule": None,
        "label": True,
        "mixin": mixin,
        "ctx_size_declared": 250,
        "source_row_ids": (0,),
        "padding_type_declared": ("books",),
        "dispersion_declared": ("first",),
    }
    fields.update(overrides)
    return FlenqaPrompt(**fields)


def test_unpadded_prompt_is_none_and_not_applicable() -> None:
    conditions = derive_conditions(_prompt("AAAA\nBBBB"))

    assert conditions.padding_type_effective == "none"
    assert conditions.dispersion_effective == "not_applicable"
    assert conditions.n_padding_chars == 0


def test_padding_type_none_is_never_assigned_when_padding_exists() -> None:
    conditions = derive_conditions(_prompt("AAAA\nBBBB pppppppppp"))

    assert conditions.padding_type_effective == "books"


def test_declared_padding_type_is_used_when_padding_exists() -> None:
    conditions = derive_conditions(
        _prompt("AAAA\nBBBB pppppppppp", padding_type_declared=("same",))
    )

    assert conditions.padding_type_effective == "same"


def test_padding_only_after_keys_is_first() -> None:
    conditions = derive_conditions(_prompt("AAAABBBB" + "p" * 40))

    assert conditions.dispersion_effective == "first"
    assert conditions.frac_padding_after == 1.0


def test_padding_only_before_keys_is_last() -> None:
    conditions = derive_conditions(_prompt("p" * 40 + "AAAABBBB"))

    assert conditions.dispersion_effective == "last"
    assert conditions.frac_padding_before == 1.0


def test_padding_split_evenly_around_keys_is_middle() -> None:
    conditions = derive_conditions(_prompt("p" * 20 + "AAAABBBB" + "p" * 20))

    assert conditions.dispersion_effective == "middle"


def test_padding_between_keys_is_scattered() -> None:
    conditions = derive_conditions(_prompt("p" * 10 + "AAAA" + "p" * 20 + "BBBB"))

    assert conditions.dispersion_effective == "scattered"
    assert conditions.frac_padding_between > 0.0


def test_fractions_sum_to_one_when_padding_exists() -> None:
    conditions = derive_conditions(_prompt("p" * 10 + "AAAA" + "p" * 20 + "BBBB" + "p" * 10))

    total = (
        conditions.frac_padding_before
        + conditions.frac_padding_between
        + conditions.frac_padding_after
    )
    assert abs(total - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/benchmarks/test_flenqa_conditions.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/jlens_reasoning/benchmarks/flenqa_conditions.py`:

```python
"""Condition variables derived from prompt content, not row metadata."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.flenqa_length_drift.constants import PLACEMENT_EPSILON
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt

PADDING_NONE = "none"
DISPERSION_NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class PromptConditions:
    padding_type_effective: str
    dispersion_effective: str
    frac_padding_before: float
    frac_padding_between: float
    frac_padding_after: float
    n_padding_chars: int


def _key_bounds(mixin: str, key_texts: tuple[str, ...]) -> tuple[int, int, int]:
    """Return (first key start, last key end, total key characters)."""
    starts: list[int] = []
    ends: list[int] = []
    total = 0
    for key in key_texts:
        start = mixin.find(key)
        if start < 0:
            raise ValueError(f"Key text not found in mixin: {key[:40]!r}")
        starts.append(start)
        ends.append(start + len(key))
        total += len(key)
    return min(starts), max(ends), total


def _classify(before: float, between: float, after: float) -> str:
    if before <= PLACEMENT_EPSILON and between <= PLACEMENT_EPSILON:
        return "first"
    if after <= PLACEMENT_EPSILON and between <= PLACEMENT_EPSILON:
        return "last"
    if between <= PLACEMENT_EPSILON and abs(before - after) <= PLACEMENT_EPSILON:
        return "middle"
    return "scattered"


def derive_conditions(prompt: FlenqaPrompt) -> PromptConditions:
    """Classify padding and placement from the prompt itself."""
    mixin = prompt.mixin
    first_start, last_end, key_chars = _key_bounds(mixin, prompt.key_texts)

    non_key = len(mixin.strip()) - key_chars
    separators = len(prompt.key_texts) - 1
    n_padding = max(non_key - separators, 0)
    if n_padding == 0:
        return PromptConditions(
            padding_type_effective=PADDING_NONE,
            dispersion_effective=DISPERSION_NOT_APPLICABLE,
            frac_padding_before=0.0,
            frac_padding_between=0.0,
            frac_padding_after=0.0,
            n_padding_chars=0,
        )

    before_chars = first_start
    after_chars = len(mixin) - last_end
    between_chars = max(len(mixin) - before_chars - after_chars - key_chars - separators, 0)
    total = before_chars + between_chars + after_chars
    total = total if total else 1

    before = before_chars / total
    between = between_chars / total
    after = after_chars / total

    if len(prompt.padding_type_declared) != 1:
        raise ValueError(
            f"Prompt {prompt.prompt_id} has padding but declares "
            f"{prompt.padding_type_declared}; padded prompts must be unambiguous"
        )

    return PromptConditions(
        padding_type_effective=prompt.padding_type_declared[0],
        dispersion_effective=_classify(before, between, after),
        frac_padding_before=before,
        frac_padding_between=between,
        frac_padding_after=after,
        n_padding_chars=n_padding,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/benchmarks/test_flenqa_conditions.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/benchmarks/flenqa_conditions.py tests/benchmarks/test_flenqa_conditions.py
git commit -m "feat: derive FLenQA conditions from prompt content"
```

---

### Task 4: Span location with offset mapping

**Files:**
- Create: `src/jlens_reasoning/experiments_utils/spans.py`
- Test: `tests/experiments_utils/test_spans.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `SPAN_OK = "ok"`, `SPAN_AMBIGUOUS = "ambiguous"`, `SPAN_UNRESOLVED = "unresolved"`
  - `CharSpan` frozen dataclass: `start: int`, `end: int`
  - `find_all_spans(text: str, needle: str) -> tuple[CharSpan, ...]`
  - `locate_unique_span(text: str, needle: str) -> tuple[CharSpan | None, str, int]` returning `(span, status, match_count)`
  - `char_span_to_token_span(offsets: Sequence[tuple[int, int]], span: CharSpan) -> tuple[int, int]`

Never first-match. Never re-tokenize a substring — that ignores surrounding context and produces off-by-token errors at boundaries.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments_utils/test_spans.py`:

```python
import pytest

from jlens_reasoning.experiments_utils.spans import (
    SPAN_AMBIGUOUS,
    SPAN_OK,
    SPAN_UNRESOLVED,
    CharSpan,
    char_span_to_token_span,
    find_all_spans,
    locate_unique_span,
)


def test_find_all_spans_finds_every_occurrence_including_overlapping_context() -> None:
    assert find_all_spans("abXabYab", "ab") == (
        CharSpan(0, 2),
        CharSpan(3, 5),
        CharSpan(6, 8),
    )


def test_locate_unique_span_returns_ok_for_single_match() -> None:
    span, status, count = locate_unique_span("hello world", "world")

    assert span == CharSpan(6, 11)
    assert status == SPAN_OK
    assert count == 1


def test_locate_unique_span_reports_ambiguous_without_guessing() -> None:
    span, status, count = locate_unique_span("ab ab", "ab")

    assert span is None
    assert status == SPAN_AMBIGUOUS
    assert count == 2


def test_locate_unique_span_reports_unresolved_when_absent() -> None:
    span, status, count = locate_unique_span("hello", "zzz")

    assert span is None
    assert status == SPAN_UNRESOLVED
    assert count == 0


def test_char_span_maps_to_enclosing_token_span() -> None:
    offsets = [(0, 2), (2, 5), (5, 9), (9, 12)]

    assert char_span_to_token_span(offsets, CharSpan(2, 9)) == (1, 3)


def test_char_span_covers_partially_overlapped_tokens() -> None:
    offsets = [(0, 4), (4, 8), (8, 12)]

    assert char_span_to_token_span(offsets, CharSpan(2, 6)) == (0, 2)


def test_char_span_to_token_span_rejects_span_outside_offsets() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        char_span_to_token_span([(0, 2)], CharSpan(50, 60))


def test_special_token_offsets_are_skipped() -> None:
    offsets = [(0, 0), (0, 3), (3, 7), (0, 0)]

    assert char_span_to_token_span(offsets, CharSpan(0, 7)) == (1, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments_utils/test_spans.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/jlens_reasoning/experiments_utils/spans.py`:

```python
"""Span location that never guesses between multiple matches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SPAN_OK = "ok"
SPAN_AMBIGUOUS = "ambiguous"
SPAN_UNRESOLVED = "unresolved"


@dataclass(frozen=True, slots=True)
class CharSpan:
    start: int
    end: int


def find_all_spans(text: str, needle: str) -> tuple[CharSpan, ...]:
    """Every occurrence of needle, scanning past each match rather than stopping."""
    if not needle:
        raise ValueError("Cannot locate an empty span")
    spans: list[CharSpan] = []
    start = text.find(needle)
    while start >= 0:
        spans.append(CharSpan(start, start + len(needle)))
        start = text.find(needle, start + 1)
    return tuple(spans)


def locate_unique_span(text: str, needle: str) -> tuple[CharSpan | None, str, int]:
    """Resolve a span only when exactly one match exists; otherwise report why."""
    spans = find_all_spans(text, needle)
    if len(spans) == 1:
        return spans[0], SPAN_OK, 1
    if not spans:
        return None, SPAN_UNRESOLVED, 0
    return None, SPAN_AMBIGUOUS, len(spans)


def char_span_to_token_span(
    offsets: Sequence[tuple[int, int]],
    span: CharSpan,
) -> tuple[int, int]:
    """Map a character span onto the half-open token range that covers it."""
    covering = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > start and start < span.end and end > span.start
    ]
    if not covering:
        raise ValueError(
            f"Character span ({span.start}, {span.end}) covers no tokens"
        )
    return covering[0], covering[-1] + 1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments_utils/test_spans.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/experiments_utils/spans.py tests/experiments_utils/test_spans.py
git commit -m "feat: locate spans without first-match guessing"
```

---

### Task 5: Bridge-entity extraction

**Files:**
- Create: `experiments/flenqa_length_drift/bridges.py`
- Test: `tests/experiments/flenqa_length_drift/__init__.py`, `tests/experiments/flenqa_length_drift/test_bridges.py` (`tests/experiments/__init__.py` already exists)

**Interfaces:**
- Consumes: `FlenqaPrompt` from Task 2; `PIR_TASK`, `MONOREL_TASK`, `RULETAKER_TASK` from `constants`.
- Produces:
  - `extract_bridge(prompt: FlenqaPrompt) -> str | None`
  - `bridge_candidate_surfaces(bridge: str) -> tuple[str, ...]`

The bridge is the entity in **both** key facts but **not** the question. A longest-common-substring rule was tested and rejected: it returns ragged spans on PIR (`"John's living room is"`) and boilerplate on MonoRel (`"This is a fact that has been established..."`). Extraction is therefore task-specific: PIR takes the possessive room phrase, MonoRel the person name absent from the question. RuleTaker has no entity bridge and returns `None`.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/__init__.py` (empty) and `tests/experiments/flenqa_length_drift/test_bridges.py`:

```python
from experiments.flenqa_length_drift.bridges import (
    bridge_candidate_surfaces,
    extract_bridge,
)
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt


def _prompt(**overrides: object) -> FlenqaPrompt:
    fields = {
        "prompt_id": "0" * 16,
        "problem_id": 0,
        "task": "PIR",
        "text": "unused",
        "question": "Is Ethan Washington in a marble-floored room?",
        "key_texts": (
            "John's living room is marble-floored, a reality that is clear.",
            "Ethan Washington is in John's living room, a fact well known.",
        ),
        "rule": None,
        "label": True,
        "mixin": "unused",
        "ctx_size_declared": 250,
        "source_row_ids": (0,),
        "padding_type_declared": ("books",),
        "dispersion_declared": ("first",),
    }
    fields.update(overrides)
    return FlenqaPrompt(**fields)


def test_pir_bridge_is_the_possessive_room_phrase() -> None:
    assert extract_bridge(_prompt()) == "John's living room"


def test_monorel_bridge_is_the_middle_person() -> None:
    prompt = _prompt(
        task="MonoRel",
        question="Is Samantha Arnold younger than Julian Barton?",
        key_texts=(
            "Julie Baker is younger than Julian Barton, a known fact.",
            "Samantha Arnold is younger than Julie Baker, also known.",
        ),
    )

    assert extract_bridge(prompt) == "Julie Baker"


def test_ruletaker_has_no_entity_bridge() -> None:
    prompt = _prompt(
        task="Simplified RuleTaker",
        question="Dave is loud.",
        key_texts=("Dave is small.", "Dave is good."),
    )

    assert extract_bridge(prompt) is None


def test_bridge_is_absent_from_the_question() -> None:
    bridge = extract_bridge(_prompt())

    assert bridge is not None
    assert bridge.lower() not in _prompt().question.lower()


def test_unresolvable_bridge_returns_none_rather_than_guessing() -> None:
    prompt = _prompt(key_texts=("Nothing shared here.", "Entirely different text."))

    assert extract_bridge(prompt) is None


def test_candidate_surfaces_cover_head_word_and_leading_space() -> None:
    surfaces = bridge_candidate_surfaces("John's living room")

    assert " room" in surfaces
    assert "room" in surfaces
    assert "John's living room" in surfaces
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_bridges.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments.flenqa_length_drift.bridges'`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/bridges.py`:

```python
"""Task-specific bridge-entity extraction.

The bridge is the entity present in both key facts but absent from the
question. A longest-common-substring rule was tested and rejected: on PIR it
returns ragged spans such as "John's living room is", and on MonoRel it returns
boilerplate filler rather than any entity. Do not reintroduce it.
"""

from __future__ import annotations

import re

from experiments.flenqa_length_drift.constants import (
    MONOREL_TASK,
    PIR_TASK,
    RULETAKER_TASK,
)
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt

_POSSESSIVE_ROOM = re.compile(r"[A-Z][a-z]+'s(?: [a-z]+)+?(?= is| ,|,|\.)")
_PERSON_NAME = re.compile(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b")


def _shared_absent_from_question(
    pattern: re.Pattern[str],
    prompt: FlenqaPrompt,
) -> str | None:
    """Candidates appearing in every key fact but never in the question."""
    question = prompt.question.lower()
    per_fact = [set(pattern.findall(text)) for text in prompt.key_texts]
    if not per_fact:
        return None
    shared = set.intersection(*per_fact)
    candidates = sorted(
        (name for name in shared if name.lower() not in question),
        key=len,
        reverse=True,
    )
    return candidates[0] if candidates else None


def extract_bridge(prompt: FlenqaPrompt) -> str | None:
    """The unspoken intermediate entity, or None when there is none."""
    if prompt.task == RULETAKER_TASK:
        return None
    if prompt.task == PIR_TASK:
        return _shared_absent_from_question(_POSSESSIVE_ROOM, prompt)
    if prompt.task == MONOREL_TASK:
        return _shared_absent_from_question(_PERSON_NAME, prompt)
    raise ValueError(f"Unknown task {prompt.task!r}")


def bridge_candidate_surfaces(bridge: str) -> tuple[str, ...]:
    """Surfaces worth measuring: the full phrase, its head word, and variants."""
    head = bridge.split()[-1]
    surfaces: list[str] = []
    for base in (bridge, head, head.capitalize()):
        for surface in (base, f" {base}"):
            if surface not in surfaces:
                surfaces.append(surface)
    return tuple(surfaces)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_bridges.py -v`
Expected: PASS (6 tests)

If the PIR or MonoRel regex fails on the real corpus, tighten it here — the Task 12 gate is the authority, and this task is not done until that gate passes on all 300 problems.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/bridges.py tests/experiments
git commit -m "feat: extract FLenQA bridge entities per task"
```

---

### Task 6: Labelled anchors and summary positions

**Files:**
- Create: `experiments/flenqa_length_drift/anchors.py`
- Test: `tests/experiments/flenqa_length_drift/test_anchors.py`

**Interfaces:**
- Consumes: `CharSpan`, `char_span_to_token_span` from Task 4; budgets from `constants`.
- Produces:
  - Label constants `ANCHOR_FACT_A_END`, `ANCHOR_FACT_B_END`, `ANCHOR_BRIDGE_FACT_A`, `ANCHOR_BRIDGE_FACT_B`, `ANCHOR_QUESTION_END`, `ANCHOR_FINAL_PROMPT`, `ANCHOR_SAMPLED_PADDING`
  - `Anchor` frozen dataclass: `label: str`, `position: int`
  - `select_anchors(*, n_tokens: int, key_token_spans: Sequence[tuple[int, int] | None], bridge_token_positions: Sequence[int | None], question_token_span: tuple[int, int] | None, seed: int) -> tuple[Anchor, ...]`
  - `select_summary_positions(*, n_tokens: int, anchors: Sequence[Anchor], key_token_spans: Sequence[tuple[int, int] | None], seed: int) -> tuple[int, ...]`

A bare integer position is not comparable across prompts of differing length, so the label is what analysis joins on. Unlocatable anchors are omitted, never faked.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_anchors.py`:

```python
from experiments.flenqa_length_drift.anchors import (
    ANCHOR_BRIDGE_FACT_A,
    ANCHOR_FACT_A_END,
    ANCHOR_FACT_B_END,
    ANCHOR_FINAL_PROMPT,
    ANCHOR_SAMPLED_PADDING,
    Anchor,
    select_anchors,
    select_summary_positions,
)
from experiments.flenqa_length_drift.constants import (
    ANCHOR_BUDGET,
    SUMMARY_POSITION_BUDGET,
)


def _anchors(**overrides: object) -> tuple[Anchor, ...]:
    kwargs = {
        "n_tokens": 200,
        "key_token_spans": [(10, 30), (40, 60)],
        "bridge_token_positions": [15, 45],
        "question_token_span": (180, 190),
        "seed": 7,
    }
    kwargs.update(overrides)
    return select_anchors(**kwargs)


def test_final_prompt_anchor_is_the_last_token() -> None:
    anchors = _anchors()

    assert Anchor(ANCHOR_FINAL_PROMPT, 199) in anchors


def test_fact_end_anchors_are_the_last_token_of_each_span() -> None:
    anchors = _anchors()

    assert Anchor(ANCHOR_FACT_A_END, 29) in anchors
    assert Anchor(ANCHOR_FACT_B_END, 59) in anchors


def test_bridge_anchors_use_the_supplied_positions() -> None:
    anchors = _anchors()

    assert Anchor(ANCHOR_BRIDGE_FACT_A, 15) in anchors


def test_unlocatable_anchors_are_omitted_not_faked() -> None:
    anchors = _anchors(bridge_token_positions=[None, None])

    assert all(anchor.label != ANCHOR_BRIDGE_FACT_A for anchor in anchors)


def test_padding_anchors_are_deterministic_for_a_seed() -> None:
    assert _anchors(seed=3) == _anchors(seed=3)


def test_padding_anchors_differ_between_seeds() -> None:
    padding_for = lambda seed: [  # noqa: E731
        anchor.position
        for anchor in _anchors(seed=seed)
        if anchor.label == ANCHOR_SAMPLED_PADDING
    ]

    assert padding_for(1) != padding_for(2)


def test_anchor_count_stays_within_budget() -> None:
    assert len(_anchors()) <= ANCHOR_BUDGET


def test_all_anchor_positions_are_in_range() -> None:
    assert all(0 <= anchor.position < 200 for anchor in _anchors())


def test_summary_positions_include_every_anchor() -> None:
    anchors = _anchors()
    positions = select_summary_positions(
        n_tokens=200,
        anchors=anchors,
        key_token_spans=[(10, 30), (40, 60)],
        seed=7,
    )

    assert {anchor.position for anchor in anchors} <= set(positions)


def test_summary_positions_are_sorted_unique_and_within_budget() -> None:
    positions = select_summary_positions(
        n_tokens=200,
        anchors=_anchors(),
        key_token_spans=[(10, 30), (40, 60)],
        seed=7,
    )

    assert list(positions) == sorted(set(positions))
    assert len(positions) <= SUMMARY_POSITION_BUDGET
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_anchors.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/anchors.py`:

```python
"""Labelled readout positions.

A bare integer position is not comparable across prompts of differing length,
so every anchor carries the label that analysis joins on.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

from experiments.flenqa_length_drift.constants import (
    ANCHOR_BUDGET,
    ANCHOR_PADDING_COUNT,
    FINAL_POSITION_COUNT,
    KEY_SPAN_SUMMARY_CAP,
    SUMMARY_POSITION_BUDGET,
)

ANCHOR_FACT_A_END = "fact_a_end"
ANCHOR_FACT_B_END = "fact_b_end"
ANCHOR_BRIDGE_FACT_A = "bridge_fact_a"
ANCHOR_BRIDGE_FACT_B = "bridge_fact_b"
ANCHOR_QUESTION_END = "question_end"
ANCHOR_FINAL_PROMPT = "final_prompt"
ANCHOR_SAMPLED_PADDING = "sampled_padding"

_FACT_END_LABELS = (ANCHOR_FACT_A_END, ANCHOR_FACT_B_END)
_BRIDGE_LABELS = (ANCHOR_BRIDGE_FACT_A, ANCHOR_BRIDGE_FACT_B)


@dataclass(frozen=True, slots=True)
class Anchor:
    label: str
    position: int


def _key_positions(key_token_spans: Sequence[tuple[int, int] | None]) -> set[int]:
    covered: set[int] = set()
    for span in key_token_spans:
        if span is not None:
            covered.update(range(span[0], span[1]))
    return covered


def select_anchors(
    *,
    n_tokens: int,
    key_token_spans: Sequence[tuple[int, int] | None],
    bridge_token_positions: Sequence[int | None],
    question_token_span: tuple[int, int] | None,
    seed: int,
) -> tuple[Anchor, ...]:
    """Anchors we know a priori matter; unlocatable ones are omitted."""
    if n_tokens <= 0:
        raise ValueError("Cannot select anchors for an empty prompt")

    anchors: list[Anchor] = []
    for label, span in zip(_FACT_END_LABELS, key_token_spans, strict=False):
        if span is not None:
            anchors.append(Anchor(label, span[1] - 1))
    for label, position in zip(_BRIDGE_LABELS, bridge_token_positions, strict=False):
        if position is not None:
            anchors.append(Anchor(label, position))
    if question_token_span is not None:
        anchors.append(Anchor(ANCHOR_QUESTION_END, question_token_span[1] - 1))
    anchors.append(Anchor(ANCHOR_FINAL_PROMPT, n_tokens - 1))

    reserved = _key_positions(key_token_spans) | {anchor.position for anchor in anchors}
    available = [index for index in range(n_tokens) if index not in reserved]
    if available:
        rng = random.Random(seed)
        count = min(ANCHOR_PADDING_COUNT, len(available))
        for position in sorted(rng.sample(available, count)):
            anchors.append(Anchor(ANCHOR_SAMPLED_PADDING, position))

    if len(anchors) > ANCHOR_BUDGET:
        raise ValueError(
            f"Selected {len(anchors)} anchors, exceeding budget {ANCHOR_BUDGET}"
        )
    return tuple(anchors)


def select_summary_positions(
    *,
    n_tokens: int,
    anchors: Sequence[Anchor],
    key_token_spans: Sequence[tuple[int, int] | None],
    seed: int,
) -> tuple[int, ...]:
    """Anchors, key-span tails, final positions, then seeded padding fill."""
    positions: set[int] = {anchor.position for anchor in anchors}
    for span in key_token_spans:
        if span is not None:
            start = max(span[0], span[1] - KEY_SPAN_SUMMARY_CAP)
            positions.update(range(start, span[1]))
    positions.update(range(max(0, n_tokens - FINAL_POSITION_COUNT), n_tokens))
    positions = {index for index in positions if 0 <= index < n_tokens}

    remaining = SUMMARY_POSITION_BUDGET - len(positions)
    if remaining > 0:
        available = [index for index in range(n_tokens) if index not in positions]
        if available:
            rng = random.Random(seed)
            positions.update(rng.sample(available, min(remaining, len(available))))

    selected = tuple(sorted(positions))
    if len(selected) > SUMMARY_POSITION_BUDGET:
        raise ValueError(
            f"Selected {len(selected)} summary positions, exceeding budget "
            f"{SUMMARY_POSITION_BUDGET}"
        )
    return selected
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_anchors.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/anchors.py tests/experiments/flenqa_length_drift/test_anchors.py
git commit -m "feat: select labelled anchor and summary positions"
```

---

### Task 7: Atomic Parquet shard storage

**Files:**
- Create: `src/jlens_reasoning/experiments_utils/storage.py`
- Modify: `pyproject.toml` (add `pyarrow` to `dependencies`)
- Test: `tests/experiments_utils/test_storage.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `write_shard(directory: Path, table: str, index: int, rows: Sequence[Mapping[str, Any]]) -> Path`
  - `read_table(directory: Path, table: str) -> list[dict[str, Any]]`
  - `completed_prompt_ids(directory: Path, table: str) -> set[str]`

An interrupted write must never be mistaken for a complete shard, so writes go to a temporary path and are renamed on completion.

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, add `"pyarrow",` to `[project].dependencies`, keeping the list alphabetical:

```toml
dependencies = [
  "huggingface-hub",
  "jlens",
  "numpy",
  "pyarrow",
  "torch",
  "transformers>=5.5",
]
```

Then run: `uv sync --locked --extra experiment` (if the lock rejects the change, run `uv sync --extra experiment` to refresh it and commit the updated `uv.lock`).

- [ ] **Step 2: Write the failing test**

Create `tests/experiments_utils/test_storage.py`:

```python
from pathlib import Path

from jlens_reasoning.experiments_utils.storage import (
    completed_prompt_ids,
    read_table,
    write_shard,
)


def _rows(prompt_id: str) -> list[dict[str, object]]:
    return [
        {"prompt_id": prompt_id, "layer": 0, "value": 1.5},
        {"prompt_id": prompt_id, "layer": 1, "value": 2.5},
    ]


def test_written_shard_round_trips(tmp_path: Path) -> None:
    write_shard(tmp_path, "summary", 0, _rows("aaa"))

    assert read_table(tmp_path, "summary") == _rows("aaa")


def test_multiple_shards_concatenate_in_index_order(tmp_path: Path) -> None:
    write_shard(tmp_path, "summary", 1, _rows("bbb"))
    write_shard(tmp_path, "summary", 0, _rows("aaa"))

    assert read_table(tmp_path, "summary") == _rows("aaa") + _rows("bbb")


def test_tables_are_isolated_from_each_other(tmp_path: Path) -> None:
    write_shard(tmp_path, "summary", 0, _rows("aaa"))
    write_shard(tmp_path, "topk", 0, _rows("bbb"))

    assert read_table(tmp_path, "topk") == _rows("bbb")


def test_missing_table_reads_as_empty(tmp_path: Path) -> None:
    assert read_table(tmp_path, "summary") == []


def test_completed_prompt_ids_are_recovered(tmp_path: Path) -> None:
    write_shard(tmp_path, "scoring", 0, _rows("aaa"))
    write_shard(tmp_path, "scoring", 1, _rows("bbb"))

    assert completed_prompt_ids(tmp_path, "scoring") == {"aaa", "bbb"}


def test_no_temporary_file_survives_a_successful_write(tmp_path: Path) -> None:
    write_shard(tmp_path, "summary", 0, _rows("aaa"))

    assert list(tmp_path.glob("**/*.tmp")) == []


def test_partial_shard_is_ignored_by_readers(tmp_path: Path) -> None:
    write_shard(tmp_path, "summary", 0, _rows("aaa"))
    (tmp_path / "summary" / "shard-00001.parquet.tmp").write_bytes(b"garbage")

    assert read_table(tmp_path, "summary") == _rows("aaa")


def test_empty_rows_write_nothing(tmp_path: Path) -> None:
    write_shard(tmp_path, "summary", 0, [])

    assert read_table(tmp_path, "summary") == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/experiments_utils/test_storage.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

Create `src/jlens_reasoning/experiments_utils/storage.py`:

```python
"""Atomic Parquet shard storage for large experiment readouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

SHARD_SUFFIX = ".parquet"
TEMP_SUFFIX = ".parquet.tmp"


def _shard_path(directory: Path, table: str, index: int) -> Path:
    return directory / table / f"shard-{index:05d}{SHARD_SUFFIX}"


def write_shard(
    directory: Path,
    table: str,
    index: int,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Write one shard atomically; readers never observe a partial file."""
    path = _shard_path(directory, table, index)
    if not rows:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix("").with_suffix(TEMP_SUFFIX)
    pq.write_table(pa.Table.from_pylist(list(rows)), temporary)
    temporary.replace(path)
    return path


def read_table(directory: Path, table: str) -> list[dict[str, Any]]:
    """Read every complete shard for a table, in shard order."""
    table_directory = directory / table
    if not table_directory.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(table_directory.glob(f"shard-*{SHARD_SUFFIX}")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def completed_prompt_ids(directory: Path, table: str) -> set[str]:
    """Prompt ids already persisted, for resume."""
    table_directory = directory / table
    if not table_directory.is_dir():
        return set()
    completed: set[str] = set()
    for path in sorted(table_directory.glob(f"shard-*{SHARD_SUFFIX}")):
        column = pq.read_table(path, columns=["prompt_id"]).column("prompt_id")
        completed.update(column.to_pylist())
    return completed
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/experiments_utils/test_storage.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add pyproject.toml uv.lock src/jlens_reasoning/experiments_utils/storage.py tests/experiments_utils/test_storage.py
git commit -m "feat: add atomic Parquet shard storage"
```

---

### Task 8: Deterministic scoring

**Files:**
- Create: `experiments/flenqa_length_drift/scoring.py`
- Test: `tests/experiments/flenqa_length_drift/test_scoring.py`

**Interfaces:**
- Consumes: `concept_token_variants` from `jlens_reasoning.experiments_utils.tokens`.
- Produces:
  - `AnswerScore` frozen dataclass: `logit_true: float`, `logit_false: float`, `predicted: bool`, `extracted: bool | None`, `generated_text: str`, `correct: bool`, `agrees: bool`
  - `logit_answer(final_logits: torch.Tensor, tokenizer: Any) -> tuple[float, float, bool]`
  - `extract_answer(text: str) -> bool | None`
  - `score_answer(*, final_logits: torch.Tensor, generated_text: str, tokenizer: Any, label: bool) -> AnswerScore`

`label` is a balanced binary, so an LLM grader would add cost and nondeterminism for nothing. The logit score is free — `apply()` already returns `model_logits`.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_scoring.py`:

```python
import torch

from experiments.flenqa_length_drift.scoring import (
    extract_answer,
    logit_answer,
    score_answer,
)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pieces = {
            "True": [1],
            " True": [2],
            "true": [3],
            " true": [4],
            "TRUE": [90, 91],
            " TRUE": [92, 93],
            "False": [5],
            " False": [6],
            "false": [7],
            " false": [8],
            "FALSE": [94, 95],
            " FALSE": [96, 97],
        }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [99, 100])


def _logits(true_value: float, false_value: float) -> torch.Tensor:
    logits = torch.zeros(16)
    logits[1] = true_value
    logits[5] = false_value
    return logits


def test_logit_answer_prefers_the_higher_scoring_variant() -> None:
    logit_true, logit_false, predicted = logit_answer(_logits(4.0, 1.0), FakeTokenizer())

    assert (logit_true, logit_false, predicted) == (4.0, 1.0, True)


def test_logit_answer_predicts_false_when_false_dominates() -> None:
    _, _, predicted = logit_answer(_logits(0.5, 3.0), FakeTokenizer())

    assert predicted is False


def test_extract_answer_reads_the_leading_verdict() -> None:
    assert extract_answer(" True, because the room is marble-floored.") is True
    assert extract_answer("False.") is False


def test_extract_answer_is_case_insensitive() -> None:
    assert extract_answer("  TRUE") is True


def test_extract_answer_returns_none_when_absent() -> None:
    assert extract_answer("I am not sure about this one.") is None


def test_score_answer_marks_correct_against_the_label() -> None:
    score = score_answer(
        final_logits=_logits(4.0, 1.0),
        generated_text=" True",
        tokenizer=FakeTokenizer(),
        label=True,
    )

    assert score.correct is True
    assert score.agrees is True
    assert score.extracted is True


def test_score_answer_records_disagreement_between_measures() -> None:
    score = score_answer(
        final_logits=_logits(4.0, 1.0),
        generated_text=" False",
        tokenizer=FakeTokenizer(),
        label=True,
    )

    assert score.predicted is True
    assert score.extracted is False
    assert score.agrees is False


def test_score_answer_correctness_follows_the_logit_measure() -> None:
    score = score_answer(
        final_logits=_logits(1.0, 4.0),
        generated_text=" True",
        tokenizer=FakeTokenizer(),
        label=True,
    )

    assert score.correct is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/scoring.py`:

```python
"""Deterministic True/False scoring.

The label is a balanced binary, so an LLM grader would add cost and
nondeterminism for nothing. The logit measure is free: apply() already returns
the model's final-position logits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch

from jlens_reasoning.experiments_utils.tokens import concept_token_variants

_VERDICT = re.compile(r"\b(true|false)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class AnswerScore:
    logit_true: float
    logit_false: float
    predicted: bool
    extracted: bool | None
    generated_text: str
    correct: bool
    agrees: bool


def _best_logit(logits: torch.Tensor, tokenizer: Any, concept: str) -> float:
    variants = concept_token_variants(tokenizer, (concept,))
    return max(float(logits[variant.token_id].item()) for variant in variants)


def logit_answer(
    final_logits: torch.Tensor,
    tokenizer: Any,
) -> tuple[float, float, bool]:
    """Compare True and False token logits at the final position."""
    logit_true = _best_logit(final_logits, tokenizer, "True")
    logit_false = _best_logit(final_logits, tokenizer, "False")
    return logit_true, logit_false, logit_true > logit_false


def extract_answer(text: str) -> bool | None:
    """Read the first explicit verdict, or None when there is none."""
    match = _VERDICT.search(text)
    if match is None:
        return None
    return match.group(1).lower() == "true"


def score_answer(
    *,
    final_logits: torch.Tensor,
    generated_text: str,
    tokenizer: Any,
    label: bool,
) -> AnswerScore:
    """Score one answer by both measures; disagreement is itself diagnostic."""
    logit_true, logit_false, predicted = logit_answer(final_logits, tokenizer)
    extracted = extract_answer(generated_text)
    return AnswerScore(
        logit_true=logit_true,
        logit_false=logit_false,
        predicted=predicted,
        extracted=extracted,
        generated_text=generated_text,
        correct=predicted == label,
        agrees=extracted is not None and extracted == predicted,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_scoring.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/scoring.py tests/experiments/flenqa_length_drift/test_scoring.py
git commit -m "feat: score FLenQA answers deterministically"
```

---

### Task 9: Readout reduction

**Files:**
- Create: `experiments/flenqa_length_drift/readout.py`
- Test: `tests/experiments/flenqa_length_drift/test_readout.py`

**Interfaces:**
- Consumes: `Anchor` from Task 6; `TOP_K`, `MAX_SEQ_LEN` from `constants`.
- Produces:
  - `LENS_JACOBIAN = "jacobian"`, `LENS_LOGIT = "logit"`
  - `exact_rank_and_logit(logits: torch.Tensor, token_id: int) -> tuple[int, float]`
  - `topk_records(*, prompt_id: str, lens_kind: str, logits_by_layer: Mapping[int, torch.Tensor], anchors: Sequence[Anchor], anchor_index: Mapping[int, int], k: int) -> list[dict[str, Any]]`
  - `bridge_records(*, prompt_id: str, lens_kind: str, logits_by_layer, anchors, anchor_index, candidate_token_ids: Sequence[int]) -> list[dict[str, Any]]`
  - `summary_records(*, prompt_id: str, lens_kind: str, logits_by_layer, positions: Sequence[int], provenance: Mapping[int, str]) -> list[dict[str, Any]]`
  - `assert_readout_preconditions(*, n_tokens: int, positions: Sequence[int] | None) -> None`

`logits_by_layer` maps layer → tensor of shape `[n_selected_positions, vocab]`. `anchor_index` maps a token position to its row in that tensor. Ranks are computed against the **full vocabulary**, because a bridge token may rank far below `TOP_K` — which at long contexts is the expected finding.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_readout.py`:

```python
import pytest
import torch

from experiments.flenqa_length_drift.anchors import (
    ANCHOR_FINAL_PROMPT,
    Anchor,
)
from experiments.flenqa_length_drift.readout import (
    LENS_JACOBIAN,
    assert_readout_preconditions,
    bridge_records,
    exact_rank_and_logit,
    summary_records,
    topk_records,
)


def _logits_by_layer() -> dict[int, torch.Tensor]:
    return {0: torch.tensor([[0.0, 3.0, 1.0, 2.0]])}


def test_exact_rank_is_one_based() -> None:
    logits = torch.tensor([0.0, 3.0, 1.0, 2.0])

    assert exact_rank_and_logit(logits, 1) == (1, 3.0)
    assert exact_rank_and_logit(logits, 3) == (2, 2.0)
    assert exact_rank_and_logit(logits, 0) == (4, 0.0)


def test_exact_rank_is_not_truncated_by_top_k() -> None:
    logits = torch.arange(1000, dtype=torch.float32)
    rank, _ = exact_rank_and_logit(logits, 0)

    assert rank == 1000


def test_topk_records_are_ranked_and_labelled() -> None:
    records = topk_records(
        prompt_id="abc",
        lens_kind=LENS_JACOBIAN,
        logits_by_layer=_logits_by_layer(),
        anchors=[Anchor(ANCHOR_FINAL_PROMPT, 5)],
        anchor_index={5: 0},
        k=2,
    )

    assert records == [
        {
            "prompt_id": "abc",
            "layer": 0,
            "anchor_label": ANCHOR_FINAL_PROMPT,
            "lens_kind": LENS_JACOBIAN,
            "rank": 1,
            "token_id": 1,
            "logit": 3.0,
        },
        {
            "prompt_id": "abc",
            "layer": 0,
            "anchor_label": ANCHOR_FINAL_PROMPT,
            "lens_kind": LENS_JACOBIAN,
            "rank": 2,
            "token_id": 3,
            "logit": 2.0,
        },
    ]


def test_topk_records_store_token_ids_not_strings() -> None:
    records = topk_records(
        prompt_id="abc",
        lens_kind=LENS_JACOBIAN,
        logits_by_layer=_logits_by_layer(),
        anchors=[Anchor(ANCHOR_FINAL_PROMPT, 5)],
        anchor_index={5: 0},
        k=1,
    )

    assert "token" not in records[0]


def test_bridge_records_carry_exact_rank_per_candidate() -> None:
    records = bridge_records(
        prompt_id="abc",
        lens_kind=LENS_JACOBIAN,
        logits_by_layer=_logits_by_layer(),
        anchors=[Anchor(ANCHOR_FINAL_PROMPT, 5)],
        anchor_index={5: 0},
        candidate_token_ids=[0, 1],
    )

    assert [(record["candidate_token_id"], record["rank"]) for record in records] == [
        (0, 4),
        (1, 1),
    ]


def test_summary_records_include_entropy_and_provenance() -> None:
    (record,) = summary_records(
        prompt_id="abc",
        lens_kind=LENS_JACOBIAN,
        logits_by_layer=_logits_by_layer(),
        positions=[5],
        provenance={5: "padding"},
    )

    assert record["provenance"] == "padding"
    assert record["top1_token_id"] == 1
    assert record["max_logit"] == 3.0
    assert record["entropy"] > 0.0


def test_preconditions_reject_none_positions() -> None:
    with pytest.raises(ValueError, match="positions"):
        assert_readout_preconditions(n_tokens=100, positions=None)


def test_preconditions_reject_out_of_range_positions() -> None:
    with pytest.raises(ValueError, match="out of range"):
        assert_readout_preconditions(n_tokens=10, positions=[0, 10])


def test_preconditions_reject_prompts_longer_than_max_seq_len() -> None:
    with pytest.raises(ValueError, match="max_seq_len"):
        assert_readout_preconditions(n_tokens=99_999, positions=[0])


def test_preconditions_accept_a_valid_selection() -> None:
    assert_readout_preconditions(n_tokens=10, positions=[0, 9])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_readout.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/readout.py`:

```python
"""Reduce one lens pass into persisted records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from experiments.flenqa_length_drift.anchors import Anchor
from experiments.flenqa_length_drift.constants import MAX_SEQ_LEN

LENS_JACOBIAN = "jacobian"
LENS_LOGIT = "logit"


def assert_readout_preconditions(
    *,
    n_tokens: int,
    positions: Sequence[int] | None,
) -> None:
    """Guard the two silent failures that would invalidate the whole run."""
    if positions is None:
        raise ValueError(
            "positions=None would accumulate the full vocabulary at every "
            "position in host RAM; an explicit selection is required"
        )
    if n_tokens > MAX_SEQ_LEN:
        raise ValueError(
            f"Prompt has {n_tokens} tokens, exceeding max_seq_len {MAX_SEQ_LEN}; "
            "the tokenizer would truncate it silently"
        )
    out_of_range = [index for index in positions if not 0 <= index < n_tokens]
    if out_of_range:
        raise ValueError(f"Positions out of range for {n_tokens} tokens: {out_of_range}")


def exact_rank_and_logit(logits: torch.Tensor, token_id: int) -> tuple[int, float]:
    """One-based rank against the full vocabulary, never truncated by top-k."""
    value = logits[token_id]
    rank = int((logits > value).sum().item()) + 1
    return rank, float(value.item())


def topk_records(
    *,
    prompt_id: str,
    lens_kind: str,
    logits_by_layer: Mapping[int, torch.Tensor],
    anchors: Sequence[Anchor],
    anchor_index: Mapping[int, int],
    k: int,
) -> list[dict[str, Any]]:
    """Top-k tokens per layer and anchor, storing ids rather than strings."""
    records: list[dict[str, Any]] = []
    for layer, layer_logits in sorted(logits_by_layer.items()):
        for anchor in anchors:
            row = layer_logits[anchor_index[anchor.position]]
            values, indices = torch.topk(row, k=min(k, row.numel()))
            for rank, (value, token_id) in enumerate(
                zip(values.tolist(), indices.tolist(), strict=True), start=1
            ):
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "layer": layer,
                        "anchor_label": anchor.label,
                        "lens_kind": lens_kind,
                        "rank": rank,
                        "token_id": int(token_id),
                        "logit": float(value),
                    }
                )
    return records


def bridge_records(
    *,
    prompt_id: str,
    lens_kind: str,
    logits_by_layer: Mapping[int, torch.Tensor],
    anchors: Sequence[Anchor],
    anchor_index: Mapping[int, int],
    candidate_token_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Exact rank and logit per bridge candidate; the primary signal."""
    records: list[dict[str, Any]] = []
    for layer, layer_logits in sorted(logits_by_layer.items()):
        for anchor in anchors:
            row = layer_logits[anchor_index[anchor.position]]
            for token_id in candidate_token_ids:
                rank, logit = exact_rank_and_logit(row, token_id)
                records.append(
                    {
                        "prompt_id": prompt_id,
                        "layer": layer,
                        "anchor_label": anchor.label,
                        "lens_kind": lens_kind,
                        "candidate_token_id": int(token_id),
                        "rank": rank,
                        "logit": logit,
                    }
                )
    return records


def summary_records(
    *,
    prompt_id: str,
    lens_kind: str,
    logits_by_layer: Mapping[int, torch.Tensor],
    positions: Sequence[int],
    provenance: Mapping[int, str],
) -> list[dict[str, Any]]:
    """Scalar summaries at every selected position."""
    records: list[dict[str, Any]] = []
    for layer, layer_logits in sorted(logits_by_layer.items()):
        for index, position in enumerate(positions):
            row = layer_logits[index]
            probabilities = torch.softmax(row, dim=-1)
            entropy = float(
                -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum().item()
            )
            records.append(
                {
                    "prompt_id": prompt_id,
                    "layer": layer,
                    "position": int(position),
                    "lens_kind": lens_kind,
                    "provenance": provenance.get(position, "other"),
                    "entropy": entropy,
                    "max_logit": float(row.max().item()),
                    "top1_token_id": int(row.argmax().item()),
                }
            )
    return records
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_readout.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/readout.py tests/experiments/flenqa_length_drift/test_readout.py
git commit -m "feat: reduce lens output into topk, bridge, and summary records"
```

---

### Task 10: Config hash and resume

**Files:**
- Create: `experiments/flenqa_length_drift/experiment.py`
- Test: `tests/experiments/flenqa_length_drift/test_experiment.py`

**Interfaces:**
- Consumes: `completed_prompt_ids`, `read_table`, `write_shard` from Task 7; `SHARD_SIZE` from `constants`.
- Produces:
  - `RunConfig` frozen dataclass: `model_name: str`, `lens_revision: str`, `prompt_template_version: str`, `top_k: int`, `anchor_budget: int`, `summary_budget: int`, `bridge_rule_version: str`, `dedup_rule_version: str`
  - `config_hash(config: RunConfig) -> str`
  - `write_run_meta(directory: Path, config: RunConfig, extra: Mapping[str, Any]) -> None`
  - `assert_resumable(directory: Path, config: RunConfig) -> None`
  - `pending_prompt_ids(directory: Path, all_prompt_ids: Sequence[str]) -> tuple[str, ...]`
  - `shard_index_for(position: int) -> int`

A resume that blends shards from incompatible configurations would silently corrupt the dataset, so a mismatch aborts.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_experiment.py`:

```python
from pathlib import Path

import pytest

from experiments.flenqa_length_drift.experiment import (
    RunConfig,
    assert_resumable,
    config_hash,
    pending_prompt_ids,
    shard_index_for,
    write_run_meta,
)
from jlens_reasoning.experiments_utils.storage import write_shard


def _config(**overrides: object) -> RunConfig:
    fields = {
        "model_name": "Qwen/Qwen3.5-4B",
        "lens_revision": "qwen-n1000",
        "prompt_template_version": "v1",
        "top_k": 25,
        "anchor_budget": 12,
        "summary_budget": 48,
        "bridge_rule_version": "v1",
        "dedup_rule_version": "v1",
    }
    fields.update(overrides)
    return RunConfig(**fields)


def test_config_hash_is_stable_for_identical_config() -> None:
    assert config_hash(_config()) == config_hash(_config())


def test_config_hash_changes_with_any_field() -> None:
    assert config_hash(_config()) != config_hash(_config(top_k=10))


def test_resume_accepts_a_matching_config(tmp_path: Path) -> None:
    write_run_meta(tmp_path, _config(), {"layer_count": 36})

    assert_resumable(tmp_path, _config())


def test_resume_aborts_on_config_mismatch(tmp_path: Path) -> None:
    write_run_meta(tmp_path, _config(), {"layer_count": 36})

    with pytest.raises(ValueError, match="configuration"):
        assert_resumable(tmp_path, _config(top_k=10))


def test_resume_accepts_a_fresh_directory(tmp_path: Path) -> None:
    assert_resumable(tmp_path, _config())


def test_pending_excludes_completed_prompts(tmp_path: Path) -> None:
    write_shard(tmp_path, "scoring", 0, [{"prompt_id": "aaa", "value": 1}])

    assert pending_prompt_ids(tmp_path, ["aaa", "bbb", "ccc"]) == ("bbb", "ccc")


def test_pending_preserves_input_order(tmp_path: Path) -> None:
    assert pending_prompt_ids(tmp_path, ["ccc", "aaa"]) == ("ccc", "aaa")


def test_shard_index_groups_by_shard_size() -> None:
    assert shard_index_for(0) == 0
    assert shard_index_for(499) == 0
    assert shard_index_for(500) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_experiment.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/experiment.py`:

```python
"""Run configuration, resume safety, and shard bookkeeping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from experiments.flenqa_length_drift.constants import SHARD_SIZE
from jlens_reasoning.experiments_utils.storage import completed_prompt_ids

RUN_META_FILENAME = "run_meta.json"
SCORING_TABLE = "scoring"


@dataclass(frozen=True, slots=True)
class RunConfig:
    model_name: str
    lens_revision: str
    prompt_template_version: str
    top_k: int
    anchor_budget: int
    summary_budget: int
    bridge_rule_version: str
    dedup_rule_version: str


def config_hash(config: RunConfig) -> str:
    """Stable digest of everything that would make shards incompatible."""
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_run_meta(
    directory: Path,
    config: RunConfig,
    extra: Mapping[str, Any],
) -> None:
    """Record the configuration so a later resume can verify compatibility."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "config_hash": config_hash(config),
        **dict(extra),
    }
    (directory / RUN_META_FILENAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def assert_resumable(directory: Path, config: RunConfig) -> None:
    """Abort rather than blend shards written under a different configuration."""
    path = directory / RUN_META_FILENAME
    if not path.is_file():
        return
    recorded = json.loads(path.read_text(encoding="utf-8")).get("config_hash")
    current = config_hash(config)
    if recorded != current:
        raise ValueError(
            f"Existing run used configuration {recorded}, current is {current}; "
            "resuming would blend incompatible shards"
        )


def pending_prompt_ids(
    directory: Path,
    all_prompt_ids: Sequence[str],
) -> tuple[str, ...]:
    """Prompts not yet persisted, in the caller's order."""
    completed = completed_prompt_ids(directory, SCORING_TABLE)
    return tuple(
        prompt_id for prompt_id in all_prompt_ids if prompt_id not in completed
    )


def shard_index_for(position: int) -> int:
    """Which shard a prompt at this position in the pending list belongs to."""
    return position // SHARD_SIZE
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_experiment.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/experiment.py tests/experiments/flenqa_length_drift/test_experiment.py
git commit -m "feat: add config-aware resume and shard bookkeeping"
```

---

### Task 11: Prompt preparation pipeline

**Files:**
- Modify: `src/jlens_reasoning/benchmarks/flenqa.py` (append `load_records`)
- Create: `experiments/flenqa_length_drift/preparation.py`
- Test: `tests/experiments/flenqa_length_drift/test_preparation.py`

**Interfaces:**
- Consumes: everything from Tasks 1–6.
- Produces:
  - `PreparedPrompt` frozen dataclass: `prompt: FlenqaPrompt`, `conditions: PromptConditions`, `bridge: str | None`, `n_tokens: int`, `anchors: tuple[Anchor, ...]`, `summary_positions: tuple[int, ...]`, `anchor_index: dict[int, int]`, `provenance: dict[int, str]`, `bridge_token_ids: tuple[int, ...]`, `span_status: str`
  - `prepare_prompt(prompt: FlenqaPrompt, tokenizer: Any) -> PreparedPrompt`
  - `load_records(cache_dir: Path | None = None) -> list[dict[str, Any]]` in `flenqa.py`
  - `prompts_table_rows(prepared: Sequence[PreparedPrompt]) -> list[dict[str, Any]]`
  - `source_rows_table_rows(prompts: Sequence[FlenqaPrompt]) -> list[dict[str, Any]]`

This is the join point: it turns a `FlenqaPrompt` into everything the GPU loop needs, and it is fully testable with a fake tokenizer.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_preparation.py`:

```python
from experiments.flenqa_length_drift.anchors import (
    ANCHOR_FACT_A_END,
    ANCHOR_FINAL_PROMPT,
)
from experiments.flenqa_length_drift.preparation import (
    prepare_prompt,
    prompts_table_rows,
    source_rows_table_rows,
)
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt


class WordTokenizer:
    """One token per whitespace-delimited word, with real character offsets."""

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        offsets: list[tuple[int, int]] = []
        ids: list[int] = []
        position = 0
        for word in text.split(" "):
            if word:
                start = text.index(word, position)
                offsets.append((start, start + len(word)))
                ids.append(abs(hash(word)) % 1000)
                position = start + len(word)
        return {"input_ids": [ids], "offset_mapping": [offsets]}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        words = [word for word in text.split(" ") if word]
        return [abs(hash(word)) % 1000 for word in words]


def _prompt(**overrides: object) -> FlenqaPrompt:
    mixin = "AAA bridge here " + "pad " * 20 + "BBB bridge here"
    fields = {
        "prompt_id": "0" * 16,
        "problem_id": 0,
        "task": "PIR",
        "text": mixin + " Question here",
        "question": "Question here",
        "key_texts": ("AAA bridge here", "BBB bridge here"),
        "rule": None,
        "label": True,
        "mixin": mixin,
        "ctx_size_declared": 500,
        "source_row_ids": (0, 1),
        "padding_type_declared": ("books",),
        "dispersion_declared": ("first", "random"),
    }
    fields.update(overrides)
    return FlenqaPrompt(**fields)


def test_prepare_returns_positive_token_count() -> None:
    prepared = prepare_prompt(_prompt(), WordTokenizer())

    assert prepared.n_tokens > 0


def test_prepare_always_produces_a_final_prompt_anchor() -> None:
    prepared = prepare_prompt(_prompt(), WordTokenizer())

    labels = {anchor.label for anchor in prepared.anchors}
    assert ANCHOR_FINAL_PROMPT in labels


def test_prepare_locates_key_span_anchors() -> None:
    prepared = prepare_prompt(_prompt(), WordTokenizer())

    labels = {anchor.label for anchor in prepared.anchors}
    assert ANCHOR_FACT_A_END in labels


def test_anchor_index_maps_every_anchor_into_summary_positions() -> None:
    prepared = prepare_prompt(_prompt(), WordTokenizer())

    for anchor in prepared.anchors:
        assert prepared.summary_positions[prepared.anchor_index[anchor.position]] == (
            anchor.position
        )


def test_provenance_labels_key_span_tokens() -> None:
    prepared = prepare_prompt(_prompt(), WordTokenizer())

    assert "fact_a" in set(prepared.provenance.values())


def test_prompts_table_carries_effective_conditions() -> None:
    prepared = prepare_prompt(_prompt(), WordTokenizer())
    (row,) = prompts_table_rows([prepared])

    assert row["padding_type_effective"] == "books"
    assert row["dispersion_effective"] in {"first", "middle", "last", "scattered"}
    assert row["n_source_rows"] == 2


def test_source_rows_table_maps_every_row_to_its_prompt() -> None:
    rows = source_rows_table_rows([_prompt()])

    assert [row["source_row_id"] for row in rows] == [0, 1]
    assert {row["prompt_id"] for row in rows} == {"0" * 16}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_preparation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Append the loader to `flenqa.py`**

Add these imports to the top of `src/jlens_reasoning/benchmarks/flenqa.py`:

```python
from pathlib import Path
```

and add `DATASET_REPO, DATASET_SPLIT` to the existing `constants` import. Then append:

```python
def load_records(cache_dir: Path | None = None) -> list[dict[str, Any]]:
    """Download the FLenQA eval split and verify it before use."""
    from datasets import load_dataset

    dataset = load_dataset(
        DATASET_REPO,
        split=DATASET_SPLIT,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    records = [dict(record) for record in dataset]
    verify_schema(records)
    return records
```

- [ ] **Step 4: Write the preparation module**

Create `experiments/flenqa_length_drift/preparation.py`:

```python
"""Turn a deduplicated prompt into everything the GPU loop needs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from experiments.flenqa_length_drift.anchors import (
    Anchor,
    select_anchors,
    select_summary_positions,
)
from experiments.flenqa_length_drift.bridges import (
    bridge_candidate_surfaces,
    extract_bridge,
)
from experiments.flenqa_length_drift.constants import MAX_SEQ_LEN
from jlens_reasoning.benchmarks.flenqa_conditions import (
    PromptConditions,
    derive_conditions,
)
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt
from jlens_reasoning.experiments_utils.spans import (
    SPAN_OK,
    CharSpan,
    char_span_to_token_span,
    locate_unique_span,
)

_FACT_PROVENANCE = ("fact_a", "fact_b")


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    prompt: FlenqaPrompt
    conditions: PromptConditions
    bridge: str | None
    n_tokens: int
    anchors: tuple[Anchor, ...]
    summary_positions: tuple[int, ...]
    anchor_index: dict[int, int]
    provenance: dict[int, str]
    bridge_token_ids: tuple[int, ...]
    span_status: str


def _seed(prompt_id: str) -> int:
    return int(prompt_id[:8], 16)


def _bridge_token_ids(tokenizer: Any, bridge: str | None) -> tuple[int, ...]:
    if bridge is None:
        return ()
    ids: list[int] = []
    for surface in bridge_candidate_surfaces(bridge):
        encoded = tokenizer.encode(surface, add_special_tokens=False)
        if len(encoded) == 1 and encoded[0] not in ids:
            ids.append(encoded[0])
    return tuple(ids)


def prepare_prompt(prompt: FlenqaPrompt, tokenizer: Any) -> PreparedPrompt:
    """Locate spans, bridges, anchors, and summary positions for one prompt."""
    encoded = tokenizer(
        prompt.text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=MAX_SEQ_LEN,
    )
    offsets = list(encoded["offset_mapping"][0])
    n_tokens = len(offsets)

    statuses: list[str] = []
    key_token_spans: list[tuple[int, int] | None] = []
    for key in prompt.key_texts:
        span, status, _ = locate_unique_span(prompt.text, key)
        statuses.append(status)
        key_token_spans.append(
            char_span_to_token_span(offsets, span) if span is not None else None
        )

    bridge = extract_bridge(prompt)
    bridge_token_positions: list[int | None] = []
    for key, token_span in zip(prompt.key_texts, key_token_spans, strict=True):
        if bridge is None or token_span is None or bridge not in key:
            bridge_token_positions.append(None)
            continue
        offset_in_key = key.rfind(bridge)
        key_start = prompt.text.find(key)
        char_span = CharSpan(
            key_start + offset_in_key,
            key_start + offset_in_key + len(bridge),
        )
        bridge_token_positions.append(char_span_to_token_span(offsets, char_span)[1] - 1)

    question_span, _, _ = locate_unique_span(prompt.text, prompt.question)
    question_token_span = (
        char_span_to_token_span(offsets, question_span)
        if question_span is not None
        else None
    )

    seed = _seed(prompt.prompt_id)
    anchors = select_anchors(
        n_tokens=n_tokens,
        key_token_spans=key_token_spans,
        bridge_token_positions=bridge_token_positions,
        question_token_span=question_token_span,
        seed=seed,
    )
    summary_positions = select_summary_positions(
        n_tokens=n_tokens,
        anchors=anchors,
        key_token_spans=key_token_spans,
        seed=seed,
    )

    provenance: dict[int, str] = {}
    for label, token_span in zip(_FACT_PROVENANCE, key_token_spans, strict=False):
        if token_span is not None:
            for index in range(token_span[0], token_span[1]):
                provenance[index] = label
    if question_token_span is not None:
        for index in range(question_token_span[0], question_token_span[1]):
            provenance[index] = "question"
    for index in summary_positions:
        provenance.setdefault(index, "padding")

    return PreparedPrompt(
        prompt=prompt,
        conditions=derive_conditions(prompt),
        bridge=bridge,
        n_tokens=n_tokens,
        anchors=anchors,
        summary_positions=summary_positions,
        anchor_index={
            position: index for index, position in enumerate(summary_positions)
        },
        provenance=provenance,
        bridge_token_ids=_bridge_token_ids(tokenizer, bridge),
        span_status=SPAN_OK if all(s == SPAN_OK for s in statuses) else statuses[0],
    )


def prompts_table_rows(prepared: Sequence[PreparedPrompt]) -> list[dict[str, Any]]:
    """One row per observation, carrying the analysis variables."""
    return [
        {
            "prompt_id": item.prompt.prompt_id,
            "problem_id": item.prompt.problem_id,
            "task": item.prompt.task,
            "label": item.prompt.label,
            "n_tokens_actual": item.n_tokens,
            "ctx_size_declared": item.prompt.ctx_size_declared,
            "padding_type_effective": item.conditions.padding_type_effective,
            "dispersion_effective": item.conditions.dispersion_effective,
            "frac_padding_before": item.conditions.frac_padding_before,
            "frac_padding_between": item.conditions.frac_padding_between,
            "frac_padding_after": item.conditions.frac_padding_after,
            "bridge_resolved": item.bridge is not None,
            "span_status": item.span_status,
            "n_source_rows": len(item.prompt.source_row_ids),
        }
        for item in prepared
    ]


def source_rows_table_rows(prompts: Sequence[FlenqaPrompt]) -> list[dict[str, Any]]:
    """Provenance only: every dataset row mapped to the prompt it produced."""
    return [
        {
            "source_row_id": source_row_id,
            "prompt_id": prompt.prompt_id,
            "problem_id": prompt.problem_id,
            "ctx_size_declared": prompt.ctx_size_declared,
        }
        for prompt in prompts
        for source_row_id in prompt.source_row_ids
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_preparation.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/benchmarks/flenqa.py experiments/flenqa_length_drift/preparation.py tests/experiments/flenqa_length_drift/test_preparation.py
git commit -m "feat: prepare FLenQA prompts for the readout loop"
```

---

### Task 12: Bridge-extractor gate over all 300 problems

**Files:**
- Create: `experiments/flenqa_length_drift/gate.py`
- Test: `tests/experiments/flenqa_length_drift/test_gate.py`

**Interfaces:**
- Consumes: `extract_bridge` from Task 5; `FlenqaPrompt` from Task 2.
- Produces:
  - `GateResult` frozen dataclass: `total: int`, `resolved: int`, `unresolved_problem_ids: tuple[int, ...]`, `passed: bool`
  - `run_bridge_gate(prompts: Sequence[FlenqaPrompt]) -> GateResult`

The bridge extractor is the most fragile part of the design and it costs nothing to check on CPU. This gate must pass before any GPU time is spent.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_gate.py`:

```python
from experiments.flenqa_length_drift.gate import run_bridge_gate
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt


def _prompt(problem_id: int, task: str, key_texts: tuple[str, ...]) -> FlenqaPrompt:
    return FlenqaPrompt(
        prompt_id=f"{problem_id:016d}",
        problem_id=problem_id,
        task=task,
        text="unused",
        question="Is Ethan Washington in a marble-floored room?",
        key_texts=key_texts,
        rule=None,
        label=True,
        mixin="unused",
        ctx_size_declared=250,
        source_row_ids=(problem_id,),
        padding_type_declared=("books",),
        dispersion_declared=("first",),
    )


_RESOLVABLE = (
    "John's living room is marble-floored, a known reality.",
    "Ethan Washington is in John's living room, a known fact.",
)
_UNRESOLVABLE = ("Nothing shared.", "Entirely different.")


def test_gate_passes_when_every_problem_resolves() -> None:
    result = run_bridge_gate([_prompt(0, "PIR", _RESOLVABLE)])

    assert result.passed is True
    assert result.resolved == 1


def test_gate_fails_and_names_unresolved_problems() -> None:
    result = run_bridge_gate(
        [_prompt(0, "PIR", _RESOLVABLE), _prompt(1, "PIR", _UNRESOLVABLE)]
    )

    assert result.passed is False
    assert result.unresolved_problem_ids == (1,)


def test_ruletaker_is_excluded_rather_than_counted_as_a_failure() -> None:
    result = run_bridge_gate(
        [_prompt(0, "Simplified RuleTaker", ("Dave is small.", "Dave is good."))]
    )

    assert result.total == 0
    assert result.passed is True


def test_gate_counts_each_problem_once_across_variants() -> None:
    result = run_bridge_gate([_prompt(0, "PIR", _RESOLVABLE)] * 5)

    assert result.total == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_gate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/gate.py`:

```python
"""CPU gate that must pass before any GPU time is spent."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from experiments.flenqa_length_drift.bridges import extract_bridge
from experiments.flenqa_length_drift.constants import RULETAKER_TASK
from jlens_reasoning.benchmarks.flenqa_prompts import FlenqaPrompt


@dataclass(frozen=True, slots=True)
class GateResult:
    total: int
    resolved: int
    unresolved_problem_ids: tuple[int, ...]
    passed: bool


def run_bridge_gate(prompts: Sequence[FlenqaPrompt]) -> GateResult:
    """Every PIR and MonoRel problem must yield a bridge entity."""
    seen: dict[int, bool] = {}
    for prompt in prompts:
        if prompt.task == RULETAKER_TASK:
            continue
        if prompt.problem_id in seen:
            continue
        seen[prompt.problem_id] = extract_bridge(prompt) is not None

    unresolved = tuple(sorted(pid for pid, ok in seen.items() if not ok))
    return GateResult(
        total=len(seen),
        resolved=sum(seen.values()),
        unresolved_problem_ids=unresolved,
        passed=not unresolved,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_gate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run the gate against the real dataset**

This is the point of the task. Run:

```bash
uv run python -c "
from jlens_reasoning.benchmarks.flenqa import load_records, normalize_rows
from jlens_reasoning.benchmarks.flenqa_prompts import deduplicate
from experiments.flenqa_length_drift.gate import run_bridge_gate
prompts = deduplicate(normalize_rows(load_records()))
print('unique prompts:', len(prompts))
result = run_bridge_gate(prompts)
print(result)
"
```

Expected: `unique prompts: 9862` and `passed=True` with `total=200`.

If `passed=False`, tighten the regexes in `bridges.py` until every listed problem resolves, then re-run. **Do not proceed to Task 13 until this gate passes.** If the unique-prompt count differs from 9862, that means the prompt template changed the grouping — investigate before continuing, and update the spec's asserted count.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/gate.py tests/experiments/flenqa_length_drift/test_gate.py
git commit -m "feat: gate the bridge extractor over all problems"
```

---

### Task 13: Lens-validity pre-flight

**Files:**
- Create: `experiments/flenqa_length_drift/preflight.py`
- Test: `tests/experiments/flenqa_length_drift/test_preflight.py`

**Interfaces:**
- Consumes: `LENS_JACOBIAN`, `LENS_LOGIT` from Task 9.
- Produces:
  - `PreflightCheck` frozen dataclass: `n_tokens: int`, `jacobian_rank: int`, `logit_rank: int`, `passed: bool`
  - `PreflightResult` frozen dataclass: `checks: tuple[PreflightCheck, ...]`, `passed: bool`
  - `pad_prompt(prompt: str, filler: str, target_words: int) -> str`
  - `evaluate_preflight(checks: Sequence[PreflightCheck]) -> PreflightResult`

Whether the Jacobian stays valid at 3000 tokens is an assumption **confounded with the variable under study**. If the J-Lens advantage collapses at length, the full run is not worth starting.

- [ ] **Step 1: Write the failing test**

Create `tests/experiments/flenqa_length_drift/test_preflight.py`:

```python
from experiments.flenqa_length_drift.preflight import (
    PreflightCheck,
    evaluate_preflight,
    pad_prompt,
)


def test_padding_reaches_the_target_word_count() -> None:
    padded = pad_prompt("core prompt here", "filler", 50)

    assert len(padded.split()) >= 50


def test_padding_preserves_the_original_prompt_at_the_end() -> None:
    padded = pad_prompt("core prompt here", "filler", 50)

    assert padded.endswith("core prompt here")


def test_padding_is_a_no_op_when_already_long_enough() -> None:
    assert pad_prompt("a b c d", "filler", 2) == "a b c d"


def test_preflight_passes_when_jacobian_beats_logit_at_every_length() -> None:
    result = evaluate_preflight(
        [
            PreflightCheck(250, jacobian_rank=1, logit_rank=40, passed=True),
            PreflightCheck(3000, jacobian_rank=3, logit_rank=50, passed=True),
        ]
    )

    assert result.passed is True


def test_preflight_fails_when_the_advantage_collapses_at_length() -> None:
    result = evaluate_preflight(
        [
            PreflightCheck(250, jacobian_rank=1, logit_rank=40, passed=True),
            PreflightCheck(3000, jacobian_rank=900, logit_rank=50, passed=False),
        ]
    )

    assert result.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_preflight.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `experiments/flenqa_length_drift/preflight.py`:

```python
"""Go/no-go gate: does the lens still work at full FLenQA length?

The lens is fitted on short wikitext prompts, so its validity at 3000 tokens is
an assumption confounded with the variable under study. This is a small gate,
not the Phase 2 control.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

SPIDER_PROMPT = (
    "The animal that spins a web and has eight legs is best described as a"
)
SPIDER_CONCEPT = "spider"
PREFLIGHT_LENGTHS = (250, 1000, 3000)
PREFLIGHT_MAX_RANK = 25


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    n_tokens: int
    jacobian_rank: int
    logit_rank: int
    passed: bool


@dataclass(frozen=True, slots=True)
class PreflightResult:
    checks: tuple[PreflightCheck, ...]
    passed: bool


def pad_prompt(prompt: str, filler: str, target_words: int) -> str:
    """Prepend filler so the prompt stays last, as in dispersion=last."""
    words = prompt.split()
    needed = target_words - len(words)
    if needed <= 0:
        return prompt
    return " ".join([filler] * needed + words)


def check_passed(jacobian_rank: int, logit_rank: int) -> bool:
    """The J-Lens must surface the concept and still beat the logit lens."""
    return jacobian_rank <= PREFLIGHT_MAX_RANK and jacobian_rank < logit_rank


def evaluate_preflight(checks: Sequence[PreflightCheck]) -> PreflightResult:
    """The run is worth starting only if every length passes."""
    ordered = tuple(checks)
    return PreflightResult(checks=ordered, passed=all(c.passed for c in ordered))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/experiments/flenqa_length_drift/test_preflight.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift/preflight.py tests/experiments/flenqa_length_drift/test_preflight.py
git commit -m "feat: add lens-validity pre-flight gate"
```

---

### Task 14: Notebooks and documentation

**Files:**
- Create: `experiments/flenqa_length_drift/flenqa_smoke.ipynb`
- Create: `experiments/flenqa_length_drift/flenqa_length_drift.ipynb`
- Modify: `README.md`
- Modify: `tests/test_notebooks.py` (the registry assertion below)

**Interfaces:**
- Consumes: every module from Tasks 1–13.
- Produces: two runnable Colab notebooks.

The smoke notebook must run the **identical** code path — a smoke test that bypasses the real path is worthless — and it must replace the 1.5–3 s per-prompt estimate with a measurement.

`tests/test_notebooks.py` auto-discovers `experiments/*/*.ipynb`, so three of its existing tests apply to the new notebooks without change: no saved outputs or execution counts, no credentials, and `initialize_colab` usage without `PROJECT_DIR`/`rev-parse`. Two constraints follow from that, and one test must be updated.

- [ ] **Step 1: Read the existing notebook conventions**

Run: `uv run pytest tests/test_notebooks.py -v` and read `notebooks/_template.ipynb` plus `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`.

Two hard constraints, both enforced by `test_notebooks_share_one_canonical_drive_loader_cell`:
- **Cell 0 of each new notebook must be byte-identical** to cell 0 of `notebooks/_template.ipynb`. Copy it, do not retype or reformat it.
- No notebook may contain saved outputs or execution counts; clear them before committing.

- [ ] **Step 1b: Update the notebook registry test**

`test_experiment_notebooks_are_discovered_without_a_registry` asserts the exact notebook list and will fail once the new notebooks exist. In `tests/test_notebooks.py`, update it to:

```python
def test_experiment_notebooks_are_discovered_without_a_registry() -> None:
    assert EXPERIMENT_NOTEBOOKS == [
        Path("experiments/flenqa_length_drift/flenqa_length_drift.ipynb"),
        Path("experiments/flenqa_length_drift/flenqa_smoke.ipynb"),
        Path("experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb"),
    ]
    assert not Path("notebooks/01_jlens_readout_sanity.ipynb").exists()
```

- [ ] **Step 2: Write the smoke notebook**

Create `experiments/flenqa_length_drift/flenqa_smoke.ipynb` with these cells:

1. Cell 0: the loader cell copied **byte-for-byte** from cell 0 of `notebooks/_template.ipynb`.
2. Markdown: purpose, L4 runtime, ~10 minute expectation.
3. `context = initialize_colab(enable_wandb=False, require_cuda=True)`.
4. Load records, normalize, deduplicate; run `run_bridge_gate` and **raise** if it fails.
5. Select 2 problems, keep all their prompts, assert every `ctx_size` and both `padding_type` values appear.
6. Load model and lens from Drive; assert bf16; record `lens.source_layers` length.
7. Run `evaluate_preflight` and **raise** if it fails.
8. Loop the selected prompts, timing each with `time.perf_counter()`, writing shards to a temporary local directory.
9. Print a per-`ctx_size` timing table and the extrapolated full-run hours for 9,862 prompts. Print a clear warning if the projection exceeds 10 hours.
10. Read back each written table and print row counts, confirming the round-trip.

- [ ] **Step 3: Write the full-run notebook**

Create `experiments/flenqa_length_drift/flenqa_length_drift.ipynb`, identical in structure but: all 9,862 prompts; `assert_resumable` before starting; `write_run_meta` including `layer_count`, `placement_basis="characters"`, and the git commit; shards written to local `/content` with a background thread syncing each completed shard to Drive; and a final full sync.

- [ ] **Step 4: Run the notebook tests**

Run: `uv run pytest tests/test_notebooks.py -v`
Expected: PASS

- [ ] **Step 5: Document the experiment in the README**

Add a section after the J-Lens sanity section, following its tone:

```markdown
## FLenQA length-drift readout

`experiments/flenqa_length_drift/` measures how the J-Lens readout changes as
the same reasoning task is given longer inputs, using the FLenQA benchmark.

The unit of analysis is the prompt, not the dataset row. FLenQA's 12,000 rows
contain 9,862 unique prompts: at `ctx_size=250` there is no padding at all, so
all eight padding and dispersion combinations are byte-identical. Condition
labels are derived from prompt content rather than row metadata.

Run `flenqa_smoke.ipynb` on an L4 first. It exercises the identical code path
on a small subset and reports measured wall-clock, which decides whether the
full run fits the budget. Then run `flenqa_length_drift.ipynb` on an A100.
Both refuse to start if the bridge-extractor gate or the lens-validity
pre-flight fails.

Results are written as Parquet shards beneath `runs/flenqa-length-drift/`.
```

- [ ] **Step 6: Run the full test suite and commit**

```bash
uv run pytest
uv run ruff format . && uv run ruff check .
git add experiments/flenqa_length_drift README.md
git commit -m "feat: add FLenQA length-drift notebooks and docs"
```

---

## Self-Review Notes

**Spec coverage:** identifiers (Tasks 1–2), content-derived conditions (Task 3), span robustness (Task 4), bridge extraction plus its gate (Tasks 5, 12), labelled anchors (Task 6), storage with atomic writes and no repeated token strings (Task 7), deterministic scoring (Task 8), exact bridge ranks (Task 9), config-aware resume (Task 10), the preparation join (Task 11), lens-validity pre-flight (Task 13), notebooks with measured timing and Drive sync (Task 14).

**Known gaps carried from the spec, to be resolved during implementation:**

1. **Prompt templates are invented, not the authors'.** Assumption 3 in the spec. Task 12 Step 5 will reveal whether the real templates change the 9,862 grouping. Extract the authors' prompts from their analysis notebook before the full run and update `prompt_template_version`.
2. **The `vocab` table is not built by any task.** `topk` and `bridge` store `token_id` only, which is the important half; dumping `tokenizer.get_vocab()` to a Parquet table is a one-cell addition in the Task 14 notebooks.
3. **`padding_type=same` bridge collisions** (spec assumption 5) remain unmeasured. Worth a CPU check before analysis, not before the run.
4. **Model identity** (spec assumption 1) — `Qwen/Qwen3.5-4B` layer count and `d_model` are unconfirmed; the smoke notebook records the real `lens.source_layers` length, which is when storage sizing becomes real.
