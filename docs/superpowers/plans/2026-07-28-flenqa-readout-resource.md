# FLenQA Lens-Readout Resource Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable `jlens_reasoning.flenqa` module and one notebook that turn the 12,000-row FLenQA dataset into a lens readout of all 9,862 unique prompts.

**Architecture:** A CPU build phase (`dataset.py`) converts rows into a cached prompt table carrying conditions, bridges, and key-span offsets. A GPU phase (`readout.py`) reads both lenses at ~8 labelled positions per prompt. `storage.py` appends atomic Parquet shards and supports resume. Every module is import-clean and testable on CPU without a model.

**Tech Stack:** Python 3.11, pandas, PyArrow, HuggingFace `datasets` + `transformers`, the `jlens` package, pytest, ruff.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-28-flenqa-readout-resource-design.md`. Read it before starting.
- Python imports go at the **top of the file**, never inline or mid-file.
- ruff: `line-length = 88`, lint rules `["B", "E", "F", "I", "UP", "W"]`. Run `uv run ruff format .` and `uv run ruff check .` before every commit.
- All new modules start with `from __future__ import annotations`.
- Tests are CPU-only and model-free. No HuggingFace, W&B, or Google Drive credentials. Follow the fake-tokenizer pattern in `tests/experiments_utils/test_tokens.py`.
- Run tests with `uv run pytest`.
- `max_seq_len` is **always passed explicitly** as `MAX_SEQ_LEN = 4096`. Never rely on the `jlens` default of 512.
- `positions` passed to `lens.apply()` is **never `None`**.
- The unit of analysis is `prompt_id`. `source_row_id` is provenance only — never a key or seed.
- This is a **resource**, not an experiment. No analysis, statistics, or length-comparison logic belongs in these modules.

---

## File Structure

**Create — module (`src/jlens_reasoning/flenqa/`):**

| File | Responsibility |
| --- | --- |
| `__init__.py` | public API re-exports |
| `dataset.py` | schema verification, templates, dedup, conditions, bridges, span offsets |
| `readout.py` | position selection, one lens pass, record building |
| `storage.py` | atomic Parquet shards, resume, config hash |

**Create — notebook:** `notebooks/02_build_flenqa_readout.ipynb`

**Create — tests:** `tests/flenqa/{__init__.py,test_dataset.py,test_readout.py,test_storage.py}`

**Modify:** `pyproject.toml` (add `pyarrow`), `README.md` (document the resource).

---

### Task 1: Schema verification and row loading

**Files:**
- Create: `src/jlens_reasoning/flenqa/__init__.py`, `src/jlens_reasoning/flenqa/dataset.py`
- Create: `tests/flenqa/__init__.py`, `tests/flenqa/test_dataset.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `TASKS`, `CTX_SIZES`, `PADDING_TYPES`, `DISPERSIONS`, `EXPECTED_ROWS`, `EXPECTED_PROBLEMS`, `EXPECTED_PROMPTS`, `MAX_SEQ_LEN`, `verify_schema(frame: pd.DataFrame) -> None`

- [ ] **Step 1: Add pyarrow to dependencies**

In `pyproject.toml`, add `"pyarrow>=18.0.0"` to the `dependencies` list. Then run `uv sync`.

- [ ] **Step 2: Write the failing test**

Create `tests/flenqa/__init__.py` (empty) and `tests/flenqa/test_dataset.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from jlens_reasoning.flenqa import dataset


def make_frame(**overrides) -> pd.DataFrame:
    """A minimal frame with the shape verify_schema expects."""
    base = {
        "dataset": ["PIR"],
        "global_sample_id": [1],
        "ctx_size": [250],
        "padding_type": ["books"],
        "dispersion": ["first"],
        "label": ["True"],
        "mixin": ["a\nb"],
        "assertion/question": ["q?"],
        "facts": [["a", "b"]],
        "statement": [None],
        "rule": [None],
        "sample_id": [0],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_verify_schema_accepts_a_well_formed_frame():
    dataset.verify_schema(make_frame())


def test_verify_schema_rejects_an_unknown_task():
    frame = make_frame(dataset=["Nonsense"])
    with pytest.raises(ValueError, match="unexpected dataset"):
        dataset.verify_schema(frame)


def test_verify_schema_rejects_an_unknown_ctx_size():
    frame = make_frame(ctx_size=[777])
    with pytest.raises(ValueError, match="unexpected ctx_size"):
        dataset.verify_schema(frame)


def test_verify_schema_rejects_an_unknown_padding_type():
    frame = make_frame(padding_type=["duplicate"])
    with pytest.raises(ValueError, match="unexpected padding_type"):
        dataset.verify_schema(frame)


def test_verify_schema_rejects_a_non_boolean_label():
    frame = make_frame(label=["maybe"])
    with pytest.raises(ValueError, match="unexpected label"):
        dataset.verify_schema(frame)


def test_verify_schema_rejects_a_missing_column():
    frame = make_frame().drop(columns=["mixin"])
    with pytest.raises(ValueError, match="missing columns"):
        dataset.verify_schema(frame)
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jlens_reasoning.flenqa'`

- [ ] **Step 4: Write the minimal implementation**

Create `src/jlens_reasoning/flenqa/__init__.py`:

```python
"""FLenQA lens-readout resource: prompt construction, readout, storage."""

from __future__ import annotations
```

Create `src/jlens_reasoning/flenqa/dataset.py`:

```python
"""Turn published FLenQA rows into the unique prompts a lens is read over.

Every constant here was verified against the published parquet (`alonj/FLenQA`,
`eval`) rather than the paper or the dataset card, both of which disagree with
the released data.
"""

from __future__ import annotations

import pandas as pd

PIR = "PIR"
MONOREL = "MonoRel"
RULETAKER = "Simplified RuleTaker"

TASKS = (PIR, MONOREL, RULETAKER)
CTX_SIZES = (250, 500, 1000, 2000, 3000)
PADDING_TYPES = ("books", "same")
DISPERSIONS = ("first", "middle", "last", "random")
LABELS = ("True", "False")

EXPECTED_ROWS = 12_000
EXPECTED_PROBLEMS = 300
EXPECTED_ROWS_PER_PROBLEM = 40
EXPECTED_PROMPTS = 9_862

MAX_SEQ_LEN = 4096

REQUIRED_COLUMNS = (
    "dataset",
    "global_sample_id",
    "ctx_size",
    "padding_type",
    "dispersion",
    "label",
    "mixin",
    "assertion/question",
    "facts",
    "statement",
    "rule",
)


def _check_values(frame: pd.DataFrame, column: str, allowed: tuple) -> None:
    unexpected = sorted(set(frame[column].dropna().unique()) - set(allowed))
    if unexpected:
        raise ValueError(f"unexpected {column}: {unexpected}; expected {list(allowed)}")


def verify_schema(frame: pd.DataFrame) -> None:
    """Fail loudly if the released data no longer matches what we verified."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing columns: {missing}")
    _check_values(frame, "dataset", TASKS)
    _check_values(frame, "ctx_size", CTX_SIZES)
    _check_values(frame, "padding_type", PADDING_TYPES)
    _check_values(frame, "dispersion", DISPERSIONS)
    _check_values(frame, "label", LABELS)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: PASS (8 tests)

- [ ] **Step 6: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add pyproject.toml uv.lock src/jlens_reasoning/flenqa tests/flenqa
git commit -m "feat(flenqa): verify the published dataset schema"
```

---

### Task 2: Prompt construction and dedup to prompt_id

**Files:**
- Modify: `src/jlens_reasoning/flenqa/dataset.py`
- Modify: `tests/flenqa/test_dataset.py`

**Interfaces:**
- Consumes: `TASKS`, `RULETAKER`, `verify_schema` from Task 1
- Produces: `TEMPLATES`, `TEMPLATE_VERSION`, `build_prompt(row: pd.Series) -> str`, `prompt_id(prompt: str) -> str`, `build_prompt_table(frame: pd.DataFrame) -> pd.DataFrame`

**Use the authors' own templates, verbatim.** They are not in the dataset repo and not in the paper — they are in the authors' analysis notebook, `alonj/Same-Task-More-Tokens` → `FLenQA analysis.ipynb`, cell 5, as a `prompt_structures` dict of per-task lambdas. Take the three non-CoT variants. Rendering them over all 12,000 rows was verified to yield exactly 9,862 unique prompts with the per-length breakdown 300/2368/2394/2400/2400.

Three details are easy to "fix" and must not be:

1. The RuleTaker template's last line has an unbalanced quote — `"True or "False"`. It is the authors' typo. Keep it.
2. `rule` is interpolated as the raw column value, which renders with list brackets: `Rule: ['If X is good and X is small then X is loud.']`. The parquet stores that column as a string already containing the bracketed form, and the authors' source data stores it as a list — interpolating the raw value reproduces the published prompt in both cases. Do **not** call `join` or `list` on it.
3. MonoRel states the question twice (once in the preamble, once before the answer). This is intended; downstream position code must use `rfind` to locate the operative question.

The RuleTaker `rule` never appears in `mixin`, so its template injects it or the task is unanswerable. All three templates end with a trailing newline, so the True/False answer token follows `\n`. `prompt_id` is a hash of the final templated string — the thing the model actually sees.

- [ ] **Step 1: Write the failing test**

Append to `tests/flenqa/test_dataset.py`:

```python
def test_build_prompt_includes_context_and_question():
    prompt = dataset.build_prompt(make_frame().iloc[0])
    assert "a\nb" in prompt
    assert "q?" in prompt
    # Every template ends with the answer instruction and a trailing newline.
    assert prompt.endswith("Answer only True or False.\n")


def test_build_prompt_injects_the_ruletaker_rule():
    row = make_frame(
        dataset=[dataset.RULETAKER],
        facts=[None],
        statement=[["Dave is small."]],
        rule=["['If X is small then X is loud.']"],
        mixin=["Dave is small. He is quite small."],
    ).iloc[0]
    prompt = dataset.build_prompt(row)
    assert "If X is small then X is loud." in prompt
    # The authors' unbalanced quote is reproduced, not repaired.
    assert prompt.endswith('Answer with either "True or "False".\n')


def test_monorel_prompt_states_the_question_twice():
    row = make_frame(dataset=[dataset.MONOREL]).iloc[0]
    prompt = dataset.build_prompt(row)
    assert prompt.count("q?") == 2
    assert prompt.find("q?") != prompt.rfind("q?")


def test_prompt_id_is_stable_and_content_determined():
    assert dataset.prompt_id("hello") == dataset.prompt_id("hello")
    assert dataset.prompt_id("hello") != dataset.prompt_id("hell0")
    assert len(dataset.prompt_id("hello")) == 16


def test_build_prompt_table_deduplicates_identical_prompts():
    frame = pd.concat([make_frame(), make_frame(dispersion=["last"])])
    table = dataset.build_prompt_table(frame)
    assert len(table) == 1
    assert table.iloc[0]["source_row_count"] == 2


def test_build_prompt_table_keeps_distinct_prompts_apart():
    frame = pd.concat([make_frame(), make_frame(mixin=["different"])])
    table = dataset.build_prompt_table(frame)
    assert len(table) == 2


def test_build_prompt_table_rejects_groups_that_mix_label():
    frame = pd.concat([make_frame(), make_frame(label=["False"])])
    with pytest.raises(ValueError, match="mixes label"):
        dataset.build_prompt_table(frame)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'build_prompt'`

- [ ] **Step 3: Write the minimal implementation**

Add to the imports at the top of `dataset.py`:

```python
import hashlib
```

Append to `dataset.py`:

```python
# Bumped whenever a template changes: it invalidates every prompt_id.
TEMPLATE_VERSION = "authors-noncot-v1"


def _ruletaker_prompt(row: pd.Series) -> str:
    # `rule` is interpolated raw, brackets and all, and the unbalanced quote on
    # the last line is the authors'. Both are in the published prompts.
    return f"""\
Answer whether the statement {row['assertion/question']} can be derived from the rule and the facts. Answer with either "True" or "False".
Rule: {row['rule']}
Facts: {row['mixin']}
Answer with either "True or "False".
"""


def _pir_prompt(row: pd.Series) -> str:
    return f"""\
{row['mixin']}
True/False Question: {row['assertion/question']}
Answer only True or False.
"""


def _monorel_prompt(row: pd.Series) -> str:
    # The question appears twice by design; use rfind to locate the operative one.
    return f"""\
Here are some facts. Answer the exact following question based on the text: {row['assertion/question']} Answer the question as it appears exactly.
{row['mixin']}
{row['assertion/question']}
Answer only True or False.
"""


TEMPLATES = {
    RULETAKER: _ruletaker_prompt,
    PIR: _pir_prompt,
    MONOREL: _monorel_prompt,
}


def build_prompt(row: pd.Series) -> str:
    """Render the exact string the model will see."""
    return TEMPLATES[row["dataset"]](row)


def prompt_id(prompt: str) -> str:
    """Stable content-determined identifier for a rendered prompt."""
    digest = hashlib.sha256(f"{TEMPLATE_VERSION}\x00{prompt}".encode())
    return digest.hexdigest()[:16]


def build_prompt_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse rows to unique prompts, keeping row identity as provenance."""
    rendered = frame.copy()
    rendered["prompt"] = [build_prompt(row) for _, row in rendered.iterrows()]
    rendered["prompt_id"] = rendered["prompt"].map(prompt_id)

    for column in ("ctx_size", "global_sample_id", "label"):
        varying = rendered.groupby("prompt_id")[column].nunique()
        offenders = varying[varying > 1]
        if len(offenders):
            raise ValueError(
                f"prompt group mixes {column}: {sorted(offenders.index)[:5]}"
            )

    grouped = rendered.groupby("prompt_id", sort=False)
    table = grouped.agg(
        prompt=("prompt", "first"),
        task=("dataset", "first"),
        problem_id=("global_sample_id", "first"),
        label=("label", "first"),
        ctx_size=("ctx_size", "first"),
        mixin=("mixin", "first"),
        question=("assertion/question", "first"),
        padding_type_declared=("padding_type", lambda s: sorted(set(s))),
        dispersion_declared=("dispersion", lambda s: sorted(set(s))),
        facts=("facts", "first"),
        statement=("statement", "first"),
        source_row_count=("prompt", "size"),
    ).reset_index()
    return table
```

`facts` and `statement` are carried forward because Tasks 3 and 4 locate spans and
bridges from them; `task` (not `dataset`) is the column name from here on.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/dataset.py tests/flenqa/test_dataset.py
git commit -m "feat(flenqa): render prompts and deduplicate to prompt_id"
```

---

### Task 3: Key-span location and content-derived conditions

**Files:**
- Modify: `src/jlens_reasoning/flenqa/dataset.py`
- Modify: `tests/flenqa/test_dataset.py`

**Interfaces:**
- Consumes: `RULETAKER`, `build_prompt_table` from Task 2
- Produces: `find_all(haystack: str, needle: str) -> list[int]`, `key_strings(task: str, facts, statement) -> list[str]`, `key_spans(mixin: str, strings: list[str]) -> list[dict]`, `is_unpadded(mixin: str, strings: list[str]) -> bool`, `placement(spans: list[dict], length: int, unpadded: bool) -> dict`, `add_conditions(table: pd.DataFrame) -> pd.DataFrame`

`key_spans` takes plain strings rather than a row, so it works on both raw
dataset rows and prompt-table rows (whose task column is named `task`, not
`dataset`).

Placement is computed from **character** offsets, keeping this module tokenizer-free and CI-testable. The values are fractions, so classification is equivalent to a token basis. Recorded as `placement_basis="characters"`.

PIR/MonoRel facts occur exactly once in `mixin` (verified); RuleTaker statements recur 2–14×, so the first occurrence is used and the count recorded.

- [ ] **Step 1: Write the failing test**

Append to `tests/flenqa/test_dataset.py`:

```python
def test_find_all_returns_every_occurrence():
    assert dataset.find_all("abcabc", "abc") == [0, 3]
    assert dataset.find_all("abc", "zzz") == []


def test_key_strings_selects_facts_or_statements_by_task():
    assert dataset.key_strings(dataset.PIR, ["a", "b"], None) == ["a", "b"]
    assert dataset.key_strings(dataset.RULETAKER, None, ["s"]) == ["s"]


def test_key_spans_locates_each_fact_once():
    spans = dataset.key_spans("PAD alpha MID beta END", ["alpha", "beta"])
    assert [s["start"] for s in spans] == [4, 14]
    assert all(s["status"] == "ok" and s["match_count"] == 1 for s in spans)


def test_key_spans_uses_the_first_occurrence_for_repeated_statements():
    spans = dataset.key_spans(
        "Dave is small. Truly, Dave is small.", ["Dave is small."]
    )
    assert spans[0]["start"] == 0
    assert spans[0]["match_count"] == 2
    assert spans[0]["status"] == "ok"


def test_key_spans_marks_a_missing_span_unresolved():
    spans = dataset.key_spans("nothing here", ["alpha"])
    assert spans[0]["status"] == "unresolved"
    assert spans[0]["start"] is None


def test_placement_measures_padding_fractions():
    spans = [{"start": 4, "end": 9}, {"start": 14, "end": 18}]
    result = dataset.placement(spans, length=22, unpadded=False)
    assert result["frac_padding_before"] == pytest.approx(4 / 13)
    assert result["frac_padding_between"] == pytest.approx(5 / 13)
    assert result["frac_padding_after"] == pytest.approx(4 / 13)


def test_placement_classifies_from_where_the_padding_sits():
    def label(spans, length):
        return dataset.placement(spans, length, unpadded=False)["placement_effective"]

    assert label([{"start": 0, "end": 5}], 10) == "facts_first"
    assert label([{"start": 5, "end": 10}], 10) == "facts_last"
    assert label([{"start": 4, "end": 6}], 10) == "facts_middle"


def test_placement_reports_not_applicable_when_unpadded():
    result = dataset.placement([{"start": 0, "end": 9}], 10, unpadded=True)
    assert result["placement_effective"] == "not_applicable"
    assert result["frac_padding_before"] == 0.0


def test_is_unpadded_ignores_whitespace_joining():
    assert dataset.is_unpadded("alpha\nbeta", ["alpha", "beta"]) is True
    assert dataset.is_unpadded("PAD alpha\nbeta", ["alpha", "beta"]) is False


def test_add_conditions_marks_the_shortest_prompt_per_problem_none():
    frame = pd.concat(
        [
            make_frame(mixin=["alpha\nbeta"]),
            make_frame(mixin=["PADDING alpha\nbeta"], ctx_size=[500]),
        ]
    )
    table = dataset.add_conditions(dataset.build_prompt_table(frame)).sort_values(
        "ctx_size"
    )
    assert table.iloc[0]["padding_type_effective"] == "none"
    assert table.iloc[1]["padding_type_effective"] == "books"


def test_add_conditions_cross_check_rejects_a_content_mismatch():
    # Shortest prompt for the problem still carries padding: the minimum-length
    # rule and the content check disagree, which must be a hard error.
    frame = pd.concat(
        [
            make_frame(mixin=["PAD alpha\nbeta"]),
            make_frame(mixin=["PAD PAD alpha\nbeta"], ctx_size=[500]),
        ]
    )
    with pytest.raises(ValueError, match="unpadded cross-check"):
        dataset.add_conditions(dataset.build_prompt_table(frame))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'find_all'`

- [ ] **Step 3: Write the minimal implementation**

Append to `dataset.py`:

```python
PLACEMENT_BASIS = "characters"
PLACEMENT_TOLERANCE = 0.02


def find_all(haystack: str, needle: str) -> list[int]:
    """Every start offset of `needle`, never just the first."""
    offsets = []
    start = haystack.find(needle)
    while start != -1:
        offsets.append(start)
        start = haystack.find(needle, start + 1)
    return offsets


def key_strings(task: str, facts, statement) -> list[str]:
    """The key paragraphs (PIR/MonoRel) or statements (RuleTaker) for a row."""
    return list(statement if task == RULETAKER else facts)


def key_spans(mixin: str, strings: list[str]) -> list[dict]:
    """Locate each key paragraph or statement inside `mixin`.

    PIR and MonoRel facts occur exactly once. RuleTaker statements recur, so the
    first occurrence anchors the span and the count is kept for the record.
    """
    spans = []
    for text in strings:
        offsets = find_all(mixin, text)
        if not offsets:
            spans.append(
                {"start": None, "end": None, "match_count": 0, "status": "unresolved"}
            )
            continue
        spans.append(
            {
                "start": offsets[0],
                "end": offsets[0] + len(text),
                "match_count": len(offsets),
                "status": "ok",
            }
        )
    return spans


def is_unpadded(mixin: str, strings: list[str]) -> bool:
    """True when the text is the key paragraphs and nothing else.

    Only meaningful for PIR and MonoRel. RuleTaker's `mixin` holds sentence
    expansions rather than the bare `statement` values, so this returns False
    there even for the unpadded baseline — hence the minimum-length rule below.
    """
    return "".join(mixin.split()) == "".join("".join(s.split()) for s in strings)


def placement(spans: list[dict], length: int, unpadded: bool) -> dict:
    """Padding fractions and the placement they imply, measured from content."""
    located = sorted(
        (s for s in spans if s["start"] is not None), key=lambda s: s["start"]
    )
    empty = {
        "frac_padding_before": None,
        "frac_padding_between": None,
        "frac_padding_after": None,
    }
    if not located:
        return {**empty, "placement_effective": "unresolved"}
    if unpadded:
        return {
            "frac_padding_before": 0.0,
            "frac_padding_between": 0.0,
            "frac_padding_after": 0.0,
            "placement_effective": "not_applicable",
        }

    first, last = located[0]["start"], located[-1]["end"]
    covered = sum(s["end"] - s["start"] for s in located)
    # Never divide by zero: a joining newline alone can leave padding_total at 1.
    padding_total = max(length - covered, 1)
    before = first
    after = length - last
    between = padding_total - before - after

    fractions = {
        "frac_padding_before": before / padding_total,
        "frac_padding_between": between / padding_total,
        "frac_padding_after": after / padding_total,
    }
    tol = PLACEMENT_TOLERANCE
    if fractions["frac_padding_before"] <= tol:
        label = "facts_first"
    elif fractions["frac_padding_after"] <= tol:
        label = "facts_last"
    else:
        label = "facts_middle"
    return {**fractions, "placement_effective": label}


def add_conditions(table: pd.DataFrame) -> pd.DataFrame:
    """Attach content-derived conditions; never trust the declared labels.

    The unpadded baseline is the *minimum-length prompt for each problem* — a
    property of the input, not a copy of `ctx_size`. On PIR and MonoRel this is
    cross-checked against strict content equality; the two rules were verified to
    agree on all 200 with zero false positives elsewhere.
    """
    enriched = table.copy()
    lengths = enriched["mixin"].str.len()
    baseline_ids = set(
        enriched.loc[lengths.groupby(enriched["problem_id"]).idxmin(), "prompt_id"]
    )

    rows = []
    for _, row in enriched.iterrows():
        strings = key_strings(row["task"], row["facts"], row["statement"])
        spans = key_spans(row["mixin"], strings)
        unpadded = row["prompt_id"] in baseline_ids

        if row["task"] != RULETAKER:
            by_content = is_unpadded(row["mixin"], strings)
            if by_content != unpadded:
                raise ValueError(
                    f"unpadded cross-check failed for {row['prompt_id']}: "
                    f"minimum-length={unpadded}, content={by_content}"
                )

        rows.append(
            {
                **placement(spans, len(row["mixin"]), unpadded),
                "key_spans": spans,
                "padding_type_effective": (
                    "none" if unpadded else row["padding_type_declared"][0]
                ),
                "placement_basis": PLACEMENT_BASIS,
            }
        )
    return pd.concat([enriched, pd.DataFrame(rows, index=enriched.index)], axis=1)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: PASS (24 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/dataset.py tests/flenqa/test_dataset.py
git commit -m "feat(flenqa): derive spans and placement conditions from content"
```

---

### Task 4: Bridge extraction

**Files:**
- Modify: `src/jlens_reasoning/flenqa/dataset.py`
- Modify: `tests/flenqa/test_dataset.py`

**Interfaces:**
- Consumes: `PIR`, `MONOREL`, `RULETAKER` from Task 1
- Produces: `extract_bridge(task: str, facts, question: str) -> tuple[str | None, str]`, `add_bridges(table: pd.DataFrame) -> pd.DataFrame`

The bridge is the entity in both facts but absent from the question — the concept the model must build internally and never emits. A generic longest-common-substring rule fails (fact paragraphs share filler boilerplate); these two task-specific rules were verified to resolve **300/300** problems with zero question leakage.

- [ ] **Step 1: Write the failing test**

Append to `tests/flenqa/test_dataset.py`:

```python
PIR_FACT_A = (
    "John's living room is marble-floored, a reality that is intrinsic to the "
    "building. It is a well-documented fact that John's living room is marble-floored."
)
PIR_FACT_B = (
    "Ethan Washington is in John's living room, a fact as much a part of the place "
    "as the walls. Ethan Washington is in John's living room."
)
MONO_FACT_A = "Julie Baker is younger than Julian Barton. This is a constant fact."
MONO_FACT_B = "Samantha Arnold is younger than Julie Baker. This is well known."


def test_extract_bridge_finds_the_pir_room():
    bridge, status = dataset.extract_bridge(
        dataset.PIR,
        [PIR_FACT_A, PIR_FACT_B],
        "Is Ethan Washington in a marble-floored room?",
    )
    assert bridge == "John's living room"
    assert status == "ok"


def test_extract_bridge_finds_the_monorel_middle_person():
    bridge, status = dataset.extract_bridge(
        dataset.MONOREL,
        [MONO_FACT_A, MONO_FACT_B],
        "Is Samantha Arnold younger than Julian Barton?",
    )
    assert bridge == "Julie Baker"
    assert status == "ok"


def test_extract_bridge_reports_none_for_ruletaker():
    bridge, status = dataset.extract_bridge(dataset.RULETAKER, None, "Dave is loud.")
    assert bridge is None
    assert status == "not_applicable"


def test_extract_bridge_never_returns_an_entity_named_in_the_question():
    bridge, _ = dataset.extract_bridge(
        dataset.MONOREL,
        [MONO_FACT_A, MONO_FACT_B],
        "Is Samantha Arnold younger than Julian Barton?",
    )
    assert bridge not in "Is Samantha Arnold younger than Julian Barton?"


def test_extract_bridge_reports_unresolved_when_facts_share_nothing():
    bridge, status = dataset.extract_bridge(
        dataset.MONOREL, ["Alpha Beta is tall.", "Gamma Delta is short."], "Who?"
    )
    assert bridge is None
    assert status == "unresolved"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'extract_bridge'`

- [ ] **Step 3: Write the minimal implementation**

Add to the imports at the top of `dataset.py`:

```python
import re
```

Append to `dataset.py`:

```python
NAME_PATTERN = re.compile(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b")
POSSESSIVE_PATTERN = re.compile(r"\b[A-Z][a-z]+'s\b")


def _common_prefix(left: str, right: str) -> str:
    size = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        size += 1
    return left[:size]


def _trim_to_word(prefix: str, left: str, right: str) -> str:
    """Cut a common prefix back to a whole-word, punctuation-free phrase."""
    prefix = re.split(r"[,.;:!?]", prefix)[0]
    cut_mid_word = any(
        len(rest) > len(prefix) and rest[len(prefix)].isalnum() for rest in (left, right)
    )
    if cut_mid_word and " " in prefix:
        prefix = prefix.rsplit(" ", 1)[0]
    return prefix.strip()


def _pir_bridge(fact_a: str, fact_b: str, question: str) -> set[str]:
    """Anchor on each possessive and extend while both facts agree."""
    shortest: dict[str, str] = {}
    for i in (m.start() for m in POSSESSIVE_PATTERN.finditer(fact_a)):
        for j in (m.start() for m in POSSESSIVE_PATTERN.finditer(fact_b)):
            if fact_a[i : i + 3] != fact_b[j : j + 3]:
                continue
            rest_a, rest_b = fact_a[i:], fact_b[j:]
            phrase = _trim_to_word(_common_prefix(rest_a, rest_b), rest_a, rest_b)
            if " " not in phrase:
                continue
            owner = phrase.split("'s")[0]
            # Keep the shortest agreement per owner: the phrase itself, not
            # whatever boilerplate happens to follow it in one pairing.
            if owner not in shortest or len(phrase) < len(shortest[owner]):
                shortest[owner] = phrase
    return {p for p in shortest.values() if p not in question}


def _monorel_bridge(fact_a: str, fact_b: str, question: str) -> set[str]:
    """The person named in both facts but not in the question."""
    both = set(NAME_PATTERN.findall(fact_a)) & set(NAME_PATTERN.findall(fact_b))
    return both - set(NAME_PATTERN.findall(question))


def extract_bridge(task: str, facts, question: str) -> tuple[str | None, str]:
    """The entity linking both facts and absent from the question."""
    if task == RULETAKER:
        return None, "not_applicable"
    rule = _pir_bridge if task == PIR else _monorel_bridge
    candidates = rule(facts[0], facts[1], question)
    if len(candidates) == 1:
        return candidates.pop(), "ok"
    if not candidates:
        return None, "unresolved"
    return None, "ambiguous"


def add_bridges(table: pd.DataFrame) -> pd.DataFrame:
    """Attach the bridge and its occurrence count to each prompt."""
    enriched = table.copy()
    bridges, statuses, counts = [], [], []
    for _, row in enriched.iterrows():
        bridge, status = extract_bridge(row["task"], row["facts"], row["question"])
        bridges.append(bridge)
        statuses.append(status)
        counts.append(row["prompt"].count(bridge) if bridge else 0)
    enriched["bridge"] = bridges
    enriched["bridge_status"] = statuses
    enriched["bridge_count"] = counts
    return enriched
```

`add_bridges` reads `row["facts"]`, which `build_prompt_table` already carries forward from Task 2. `extract_bridge` ignores `facts` entirely for RuleTaker, where it is null.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: PASS (29 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/dataset.py tests/flenqa/test_dataset.py
git commit -m "feat(flenqa): extract the bridge entity for PIR and MonoRel"
```

---

### Task 5: Cached prompt-table loading with asserted invariants

**Files:**
- Modify: `src/jlens_reasoning/flenqa/dataset.py`, `src/jlens_reasoning/flenqa/__init__.py`
- Modify: `tests/flenqa/test_dataset.py`

**Interfaces:**
- Consumes: everything from Tasks 1–4
- Produces: `load_prompts(paths, *, rebuild: bool = False) -> pd.DataFrame`, `verify_invariants(table: pd.DataFrame) -> None`

`load_prompts` is the module's main entry point. The full-scale invariants (300 problems, 9,862 prompts) are asserted here, not assumed — templating could merge or split groups.

- [ ] **Step 1: Write the failing test**

Append to `tests/flenqa/test_dataset.py`:

```python
def test_verify_invariants_accepts_a_full_size_table():
    table = pd.DataFrame(
        {
            "prompt_id": [f"id{i}" for i in range(dataset.EXPECTED_PROMPTS)],
            "problem_id": [i % dataset.EXPECTED_PROBLEMS for i in range(dataset.EXPECTED_PROMPTS)],
            "bridge_status": ["ok"] * dataset.EXPECTED_PROMPTS,
        }
    )
    dataset.verify_invariants(table)


def test_verify_invariants_rejects_a_wrong_prompt_count():
    table = pd.DataFrame(
        {"prompt_id": ["a"], "problem_id": [1], "bridge_status": ["ok"]}
    )
    with pytest.raises(ValueError, match="expected 9862 unique prompts"):
        dataset.verify_invariants(table)


def test_load_prompts_uses_the_cache_when_present(tmp_path, monkeypatch):
    cached = pd.DataFrame({"prompt_id": ["abc"], "prompt": ["hello"]})
    cache = tmp_path / "flenqa_prompts.parquet"
    cached.to_parquet(cache)

    class Paths:
        datasets = tmp_path

    def explode(*args, **kwargs):
        raise AssertionError("must not download when a cache exists")

    monkeypatch.setattr(dataset, "_download_rows", explode)
    table = dataset.load_prompts(Paths())
    assert table.iloc[0]["prompt_id"] == "abc"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'verify_invariants'`

- [ ] **Step 3: Write the minimal implementation**

Add to the imports at the top of `dataset.py`:

```python
from pathlib import Path

from datasets import load_dataset
```

Append to `dataset.py`:

```python
HF_REPO = "alonj/FLenQA"
HF_SPLIT = "eval"
CACHE_NAME = "flenqa_prompts.parquet"


def verify_invariants(table: pd.DataFrame) -> None:
    """Assert the verified shape of the deduplicated design."""
    if len(table) != EXPECTED_PROMPTS:
        raise ValueError(
            f"expected {EXPECTED_PROMPTS} unique prompts, got {len(table)}"
        )
    problems = table["problem_id"].nunique()
    if problems != EXPECTED_PROBLEMS:
        raise ValueError(f"expected {EXPECTED_PROBLEMS} problems, got {problems}")
    unresolved = table["bridge_status"].eq("unresolved").sum()
    if unresolved:
        raise ValueError(f"{unresolved} problems have an unresolved bridge")


def _download_rows() -> pd.DataFrame:
    return load_dataset(HF_REPO, split=HF_SPLIT).to_pandas()


def load_prompts(paths, *, rebuild: bool = False) -> pd.DataFrame:
    """The 9,862-prompt table, built once and cached to Parquet."""
    cache = Path(paths.datasets) / CACHE_NAME
    if cache.is_file() and not rebuild:
        return pd.read_parquet(cache)

    frame = _download_rows()
    verify_schema(frame)
    table = add_bridges(add_conditions(build_prompt_table(frame)))
    verify_invariants(table)

    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".parquet.tmp")
    table.to_parquet(temporary)
    temporary.replace(cache)
    return table
```

Replace the body of `src/jlens_reasoning/flenqa/__init__.py`:

```python
"""FLenQA lens-readout resource: prompt construction, readout, storage."""

from __future__ import annotations

from jlens_reasoning.flenqa.dataset import (
    EXPECTED_PROMPTS,
    MAX_SEQ_LEN,
    extract_bridge,
    load_prompts,
)

__all__ = ["EXPECTED_PROMPTS", "MAX_SEQ_LEN", "extract_bridge", "load_prompts"]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_dataset.py -v`
Expected: PASS (32 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa tests/flenqa
git commit -m "feat(flenqa): cache the prompt table and assert its invariants"
```

---

### Task 6: Atomic Parquet shard storage with resume

**Files:**
- Create: `src/jlens_reasoning/flenqa/storage.py`, `tests/flenqa/test_storage.py`

**Interfaces:**
- Produces: `config_hash(config: dict) -> str`, `write_shard(directory: Path, name: str, records: pd.DataFrame) -> Path`, `completed_prompt_ids(directory: Path) -> set[str]`, `check_config(directory: Path, config: dict) -> None`

Shards are written to a temp path and renamed, so a crash mid-write cannot leave a half-shard that resume would trust. A config mismatch aborts rather than blending incompatible runs.

- [ ] **Step 1: Write the failing test**

Create `tests/flenqa/test_storage.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from jlens_reasoning.flenqa import storage


def records(*prompt_ids: str) -> pd.DataFrame:
    return pd.DataFrame({"prompt_id": list(prompt_ids), "value": range(len(prompt_ids))})


def test_write_shard_round_trips(tmp_path):
    path = storage.write_shard(tmp_path, "shard_0000", records("a", "b"))
    assert pd.read_parquet(path)["prompt_id"].tolist() == ["a", "b"]


def test_write_shard_leaves_no_temporary_files(tmp_path):
    storage.write_shard(tmp_path, "shard_0000", records("a"))
    assert [p.name for p in tmp_path.iterdir()] == ["shard_0000.parquet"]


def test_completed_prompt_ids_unions_every_shard(tmp_path):
    storage.write_shard(tmp_path, "shard_0000", records("a", "b"))
    storage.write_shard(tmp_path, "shard_0001", records("c"))
    assert storage.completed_prompt_ids(tmp_path) == {"a", "b", "c"}


def test_completed_prompt_ids_is_empty_for_a_fresh_directory(tmp_path):
    assert storage.completed_prompt_ids(tmp_path) == set()


def test_completed_prompt_ids_ignores_a_partial_temporary_file(tmp_path):
    storage.write_shard(tmp_path, "shard_0000", records("a"))
    (tmp_path / "shard_0001.parquet.tmp").write_bytes(b"garbage")
    assert storage.completed_prompt_ids(tmp_path) == {"a"}


def test_config_hash_is_stable_and_order_independent(tmp_path):
    assert storage.config_hash({"a": 1, "b": 2}) == storage.config_hash({"b": 2, "a": 1})
    assert storage.config_hash({"a": 1}) != storage.config_hash({"a": 2})


def test_check_config_records_then_accepts_the_same_config(tmp_path):
    storage.check_config(tmp_path, {"model": "qwen"})
    storage.check_config(tmp_path, {"model": "qwen"})


def test_check_config_aborts_on_a_changed_config(tmp_path):
    storage.check_config(tmp_path, {"model": "qwen"})
    with pytest.raises(RuntimeError, match="config mismatch"):
        storage.check_config(tmp_path, {"model": "llama"})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_storage.py -v`
Expected: FAIL — `ImportError: cannot import name 'storage'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/jlens_reasoning/flenqa/storage.py`:

```python
"""Atomic Parquet shards with config-aware resume."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CONFIG_NAME = "run_config.json"


def config_hash(config: dict) -> str:
    """Order-independent hash of the run configuration."""
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_shard(directory: Path, name: str, records: pd.DataFrame) -> Path:
    """Write one shard atomically: temp path, then rename."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / f"{name}.parquet"
    temporary = directory / f"{name}.parquet.tmp"
    records.to_parquet(temporary, index=False)
    temporary.replace(final)
    return final


def completed_prompt_ids(directory: Path) -> set[str]:
    """Every prompt_id already persisted, ignoring partial writes."""
    directory = Path(directory)
    if not directory.is_dir():
        return set()
    done: set[str] = set()
    for shard in sorted(directory.glob("*.parquet")):
        done.update(pd.read_parquet(shard, columns=["prompt_id"])["prompt_id"])
    return done


def check_config(directory: Path, config: dict) -> None:
    """Record the run config, or abort if it changed since the last shard."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / CONFIG_NAME
    current = config_hash(config)
    if marker.is_file():
        previous = json.loads(marker.read_text())["config_hash"]
        if previous != current:
            raise RuntimeError(
                f"config mismatch: shards were written with {previous}, "
                f"this run is {current}; use a fresh directory"
            )
        return
    marker.write_text(json.dumps({"config_hash": current, "config": config}, default=str))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_storage.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/storage.py tests/flenqa/test_storage.py
git commit -m "feat(flenqa): add atomic shard storage with config-aware resume"
```

---

### Task 7: Position selection

**Files:**
- Create: `src/jlens_reasoning/flenqa/readout.py`, `tests/flenqa/test_readout.py`

**Interfaces:**
- Consumes: `MAX_SEQ_LEN` from Task 1
- Produces: `char_to_token(offsets, index: int) -> int | None`, `select_positions(tokenizer, prompt: str, row) -> dict[str, int]`

Positions carry labels because a bare index is not comparable across prompts. A position that does not exist — RuleTaker has no bridge, 250-token prompts have no padding — is simply absent from the mapping, and the prompt is still read.

- [ ] **Step 1: Write the failing test**

Create `tests/flenqa/test_readout.py`:

```python
from __future__ import annotations

import pytest

from jlens_reasoning.flenqa import readout


class FakeTokenizer:
    """One token per character, so offsets are trivially predictable."""

    def __call__(self, text, return_offsets_mapping=False, **kwargs):
        offsets = [(i, i + 1) for i in range(len(text))]
        result = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def test_char_to_token_maps_into_the_containing_token():
    offsets = [(0, 3), (3, 6), (6, 9)]
    assert readout.char_to_token(offsets, 0) == 0
    assert readout.char_to_token(offsets, 4) == 1
    assert readout.char_to_token(offsets, 8) == 2


def test_char_to_token_returns_none_past_the_end():
    assert readout.char_to_token([(0, 3)], 99) is None


def test_select_positions_always_includes_the_final_token():
    row = {"prompt": "hello world", "mixin": "hello", "question": "world",
           "key_spans": [], "bridge": None}
    positions = readout.select_positions(FakeTokenizer(), row["prompt"], row)
    assert positions["final_token"] == len("hello world") - 1


def test_select_positions_omits_the_bridge_when_absent():
    row = {"prompt": "hello world", "mixin": "hello", "question": "world",
           "key_spans": [], "bridge": None}
    positions = readout.select_positions(FakeTokenizer(), row["prompt"], row)
    assert "bridge_fact_a" not in positions


def test_select_positions_anchors_the_last_bridge_mention_in_each_fact():
    prompt = "AAA Zed BBB Zed CCC"
    row = {
        "prompt": prompt,
        "mixin": prompt,
        "question": "CCC",
        "key_spans": [
            {"start": 0, "end": 11, "status": "ok"},
            {"start": 11, "end": 19, "status": "ok"},
        ],
        "bridge": "Zed",
    }
    positions = readout.select_positions(FakeTokenizer(), prompt, row)
    assert positions["bridge_fact_a"] == prompt.index("Zed") + len("Zed") - 1
    assert positions["bridge_fact_b"] == prompt.rindex("Zed") + len("Zed") - 1


def test_select_positions_shifts_spans_past_template_preamble():
    """Spans are offsets into `mixin`; the template puts instructions first."""
    prompt = "PREAMBLE. AAA Zed BBB Zed CCC"
    mixin = "AAA Zed BBB Zed"
    row = {
        "prompt": prompt,
        "mixin": mixin,
        "question": "CCC",
        "key_spans": [{"start": 0, "end": 11, "status": "ok"}],
        "bridge": "Zed",
    }
    positions = readout.select_positions(FakeTokenizer(), prompt, row)
    # Span (0, 11) in mixin space is (10, 21) in prompt space.
    assert positions["fact_a_end"] == 20
    assert positions["bridge_fact_a"] == prompt.index("Zed") + len("Zed") - 1
    # Padding is sampled inside the mixin, never from the preamble.
    padding = [v for k, v in positions.items() if k.startswith("padding_")]
    assert all(10 <= v < 10 + len(mixin) for v in padding)


def test_select_positions_rejects_a_prompt_missing_its_mixin():
    row = {"prompt": "hello world", "mixin": "absent", "question": "world",
           "key_spans": [], "bridge": None}
    with pytest.raises(ValueError, match="mixin not found"):
        readout.select_positions(FakeTokenizer(), row["prompt"], row)


def test_select_positions_are_unique_and_in_range():
    prompt = "AAA Zed BBB Zed CCC"
    row = {
        "prompt": prompt,
        "mixin": prompt,
        "question": "CCC",
        "key_spans": [{"start": 0, "end": 11, "status": "ok"}],
        "bridge": "Zed",
    }
    positions = readout.select_positions(FakeTokenizer(), prompt, row)
    values = list(positions.values())
    assert len(values) == len(set(values))
    assert all(0 <= v < len(prompt) for v in values)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_readout.py -v`
Expected: FAIL — `ImportError: cannot import name 'readout'`

- [ ] **Step 3: Write the minimal implementation**

Create `src/jlens_reasoning/flenqa/readout.py`:

```python
"""Read both lenses at labelled positions within one prompt."""

from __future__ import annotations

import random

from jlens_reasoning.flenqa.dataset import MAX_SEQ_LEN, find_all

PADDING_SAMPLE_COUNT = 2


def char_to_token(offsets, index: int) -> int | None:
    """The token containing a character offset, via the tokenizer's own map."""
    for position, (start, end) in enumerate(offsets):
        if start <= index < end:
            return position
    return None


def select_positions(tokenizer, prompt: str, row) -> dict[str, int]:
    """Labelled token positions to read; absent labels simply do not appear."""
    encoded = tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offsets = encoded["offset_mapping"]
    n_tokens = len(offsets)

    def at(char_index: int | None) -> int | None:
        return None if char_index is None else char_to_token(offsets, char_index)

    positions: dict[str, int] = {"final_token": n_tokens - 1}

    # MonoRel states the question twice; the operative one is the last.
    question_index = prompt.rfind(row["question"])
    if question_index != -1:
        positions["question_end"] = at(question_index + len(row["question"]) - 1)

    # key_spans are offsets into `mixin`. The RuleTaker and MonoRel templates put
    # instruction text before the mixin, so shift every span into prompt space.
    mixin = row["mixin"]
    shift = prompt.find(mixin)
    if shift == -1:
        raise ValueError(f"mixin not found verbatim in prompt: {row.get('prompt_id')}")
    mixin_range = range(shift, shift + len(mixin))

    spans = [
        {**s, "start": s["start"] + shift, "end": s["end"] + shift}
        for s in row["key_spans"]
        if s.get("status") == "ok"
    ]
    for label, span in zip(("fact_a_end", "fact_b_end"), spans, strict=False):
        positions[label] = at(span["end"] - 1)

    bridge = row["bridge"]
    if bridge:
        for label, span in zip(
            ("bridge_fact_a", "bridge_fact_b"), spans, strict=False
        ):
            within = [
                offset
                for offset in find_all(prompt, bridge)
                if span["start"] <= offset < span["end"]
            ]
            if within:
                positions[label] = at(within[-1] + len(bridge) - 1)

    covered = {
        index
        for span in spans
        for index in range(span["start"], span["end"])
    }
    # Padding is the filler inside the mixin only — the template's instruction
    # text is neither padding nor key content.
    padding = [i for i in mixin_range if i not in covered]
    if padding:
        rng = random.Random(row["prompt_id"] if "prompt_id" in row else prompt)
        for n, offset in enumerate(rng.sample(padding, min(PADDING_SAMPLE_COUNT, len(padding)))):
            positions[f"padding_{n + 1}"] = at(offset)

    resolved = {k: v for k, v in positions.items() if v is not None}
    # Two labels can land on one token; keep the first and stay unique.
    seen: dict[int, str] = {}
    unique = {}
    for label, position in resolved.items():
        if position not in seen:
            seen[position] = label
            unique[label] = position
    return unique
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_readout.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/readout.py tests/flenqa/test_readout.py
git commit -m "feat(flenqa): select labelled read positions within a prompt"
```

---

### Task 8: Lens pass and record building

**Files:**
- Modify: `src/jlens_reasoning/flenqa/readout.py`, `tests/flenqa/test_readout.py`

**Interfaces:**
- Consumes: `select_positions` from Task 7, `MAX_SEQ_LEN` from Task 1
- Produces: `target_token_ids(tokenizer, text: str) -> list[int]`, `rank_of(logits, token_id: int) -> tuple[int, float]`, `read_prompt(model, lens, tokenizer, prompt, *, positions, targets, layers, top_k=25) -> pd.DataFrame`

Exact ranks are computed against the **full vocabulary**, not read off a truncated top-k: at long contexts a bridge token ranking far below 25 is the expected finding, so a top-k-only design would censor the primary signal. Both lenses are read at the same positions, so the comparison is matched.

- [ ] **Step 1: Write the failing test**

Append to `tests/flenqa/test_readout.py`:

```python
import torch


class FakeLens:
    source_layers = [0, 1]

    def __init__(self):
        self.calls = []

    def apply(self, model, prompt, *, positions, max_seq_len, use_jacobian, layers=None):
        self.calls.append(
            {"positions": positions, "max_seq_len": max_seq_len, "jacobian": use_jacobian}
        )
        offset = 0.0 if use_jacobian else 5.0
        lens_logits = {
            layer: torch.arange(8, dtype=torch.float32).repeat(len(positions), 1) + offset
            for layer in (layers or self.source_layers)
        }
        model_logits = torch.arange(8, dtype=torch.float32).repeat(len(positions), 1)
        input_ids = torch.zeros((1, 12), dtype=torch.long)
        return lens_logits, model_logits, input_ids


def test_rank_of_counts_strictly_higher_logits():
    logits = torch.tensor([3.0, 1.0, 2.0])
    assert readout.rank_of(logits, 0) == (1, 3.0)
    assert readout.rank_of(logits, 2) == (2, 2.0)
    assert readout.rank_of(logits, 1) == (3, 1.0)


def test_read_prompt_passes_the_explicit_max_seq_len():
    lens = FakeLens()
    readout.read_prompt(
        object(), lens, FakeTokenizer(), "hello",
        positions={"final_token": 3}, targets={"answer_true": 1}, layers=[0, 1],
    )
    assert all(call["max_seq_len"] == readout.MAX_SEQ_LEN for call in lens.calls)


def test_read_prompt_never_passes_none_positions():
    lens = FakeLens()
    readout.read_prompt(
        object(), lens, FakeTokenizer(), "hello",
        positions={"final_token": 3}, targets={"answer_true": 1}, layers=[0, 1],
    )
    assert all(call["positions"] is not None for call in lens.calls)


def test_read_prompt_reads_both_lenses_at_the_same_positions():
    lens = FakeLens()
    readout.read_prompt(
        object(), lens, FakeTokenizer(), "hello",
        positions={"final_token": 3}, targets={"answer_true": 1}, layers=[0, 1],
    )
    assert {call["jacobian"] for call in lens.calls} == {True, False}
    assert len({tuple(call["positions"]) for call in lens.calls}) == 1


def test_read_prompt_emits_one_row_per_position_layer_and_lens():
    lens = FakeLens()
    records = readout.read_prompt(
        object(), lens, FakeTokenizer(), "hello",
        positions={"final_token": 3, "question_end": 1},
        targets={"answer_true": 1}, layers=[0, 1],
    )
    assert len(records) == 2 * 2 * 2
    assert set(records["lens_kind"]) == {"jacobian", "logit"}
    assert set(records["anchor_label"]) == {"final_token", "question_end"}


def test_read_prompt_records_exact_target_ranks():
    lens = FakeLens()
    records = readout.read_prompt(
        object(), lens, FakeTokenizer(), "hello",
        positions={"final_token": 3}, targets={"answer_true": 6},
        layers=[0], top_k=3,
    )
    # Logits ascend 0..7, so token 6 is the second-highest of eight.
    assert set(records["rank_answer_true"]) == {2}


def test_read_prompt_raises_when_the_prompt_would_truncate():
    lens = FakeLens()
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        readout.read_prompt(
            object(), lens, FakeTokenizer(), "x" * (readout.MAX_SEQ_LEN + 1),
            positions={"final_token": 3}, targets={}, layers=[0],
        )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_readout.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'rank_of'`

- [ ] **Step 3: Write the minimal implementation**

Add to the imports at the top of `readout.py`:

```python
import pandas as pd
import torch
```

Append to `readout.py`:

```python
TOP_K = 25
LENS_KINDS = (("jacobian", True), ("logit", False))


def target_token_ids(tokenizer, text: str) -> list[int]:
    """Token ids for a target string, as it appears mid-prompt."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def rank_of(logits, token_id: int) -> tuple[int, float]:
    """Exact 1-based rank against the full vocabulary, plus the logit."""
    value = float(logits[token_id])
    rank = int((logits > logits[token_id]).sum()) + 1
    return rank, value


def read_prompt(
    model,
    lens,
    tokenizer,
    prompt: str,
    *,
    positions: dict[str, int],
    targets: dict[str, int],
    layers,
    top_k: int = TOP_K,
) -> pd.DataFrame:
    """Read both lenses at `positions` and return one row per cell.

    `positions` and `targets` come from the caller, not from a FLenQA row, so a
    later experiment can reuse this with synthetic prompts unchanged.
    """
    n_tokens = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if n_tokens > MAX_SEQ_LEN:
        raise ValueError(f"prompt of {n_tokens} tokens exceeds max_seq_len {MAX_SEQ_LEN}")
    if not positions:
        raise ValueError("positions must not be empty")

    labels = list(positions)
    indices = [positions[label] for label in labels]

    rows = []
    for kind, use_jacobian in LENS_KINDS:
        lens_logits, _, _ = lens.apply(
            model,
            prompt,
            layers=list(layers),
            positions=indices,
            max_seq_len=MAX_SEQ_LEN,
            use_jacobian=use_jacobian,
        )
        for layer, matrix in lens_logits.items():
            for offset, label in enumerate(labels):
                logits = matrix[offset]
                top = torch.topk(logits, k=min(top_k, logits.shape[-1]))
                record = {
                    "anchor_label": label,
                    "position": positions[label],
                    "layer": int(layer),
                    "lens_kind": kind,
                    "top_ids": top.indices.tolist(),
                    "top_logits": [round(v, 4) for v in top.values.tolist()],
                    "top1_id": int(top.indices[0]),
                    "max_logit": float(top.values[0]),
                    "entropy": float(
                        -(logits.softmax(-1) * logits.log_softmax(-1)).sum()
                    ),
                }
                for name, token_id in targets.items():
                    rank, value = rank_of(logits, token_id)
                    record[f"rank_{name}"] = rank
                    record[f"logit_{name}"] = value
                rows.append(record)
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_readout.py -v`
Expected: PASS (13 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/readout.py tests/flenqa/test_readout.py
git commit -m "feat(flenqa): read both lenses and record exact target ranks"
```

---

### Task 9: Deterministic scoring

**Files:**
- Modify: `src/jlens_reasoning/flenqa/readout.py`, `tests/flenqa/test_readout.py`

**Interfaces:**
- Consumes: `rank_of` from Task 8
- Produces: `score_answer(model_logits, true_id: int, false_id: int, label: str) -> dict`

`label` is exactly `"True"`/`"False"` and balanced, so scoring compares the two answer logits at the final position. Deterministic and free — `apply()` already returns `model_logits`. **No generation pass:** it roughly doubles GPU time to produce a noisier version of the same binary answer.

- [ ] **Step 1: Write the failing test**

Append to `tests/flenqa/test_readout.py`:

```python
def test_score_answer_marks_a_correct_prediction():
    logits = torch.tensor([[0.0, 9.0, 1.0]])
    result = readout.score_answer(logits, true_id=1, false_id=2, label="True")
    assert result["predicted"] == "True"
    assert result["correct"] is True
    assert result["margin"] == pytest.approx(8.0)


def test_score_answer_marks_an_incorrect_prediction():
    logits = torch.tensor([[0.0, 1.0, 9.0]])
    result = readout.score_answer(logits, true_id=1, false_id=2, label="True")
    assert result["predicted"] == "False"
    assert result["correct"] is False
    assert result["margin"] == pytest.approx(-8.0)


def test_score_answer_reads_only_the_final_position():
    logits = torch.tensor([[9.0, 0.0, 0.0], [0.0, 0.0, 9.0]])
    result = readout.score_answer(logits, true_id=1, false_id=2, label="False")
    assert result["predicted"] == "False"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/flenqa/test_readout.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'score_answer'`

- [ ] **Step 3: Write the minimal implementation**

Append to `readout.py`:

```python
def score_answer(model_logits, true_id: int, false_id: int, label: str) -> dict:
    """Compare the True/False logits at the final position. No generation."""
    final = model_logits[-1]
    true_logit = float(final[true_id])
    false_logit = float(final[false_id])
    predicted = "True" if true_logit >= false_logit else "False"
    margin = true_logit - false_logit
    return {
        "predicted": predicted,
        "correct": predicted == label,
        "logit_true": true_logit,
        "logit_false": false_logit,
        "margin": margin if label == "True" else -margin,
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/flenqa/test_readout.py -v`
Expected: PASS (16 tests)

- [ ] **Step 5: Commit**

```bash
uv run ruff format . && uv run ruff check .
git add src/jlens_reasoning/flenqa/readout.py tests/flenqa/test_readout.py
git commit -m "feat(flenqa): score answers from final-position logits"
```

---

### Task 10: The notebook

**Files:**
- Create: `notebooks/02_build_flenqa_readout.ipynb`
- Modify: `README.md`, `src/jlens_reasoning/flenqa/__init__.py`

**Interfaces:**
- Consumes: `load_prompts` (Task 5), `write_shard`/`completed_prompt_ids`/`check_config` (Task 6), `select_positions`/`read_prompt`/`score_answer`/`target_token_ids` (Tasks 7–9)

Follows `notebooks/_template.ipynb`: two bootstrap cells, then the work. A `LIMIT` constant covers the smoke run, so there is no second notebook.

- [ ] **Step 1: Export the remaining public API**

Replace `src/jlens_reasoning/flenqa/__init__.py`:

```python
"""FLenQA lens-readout resource: prompt construction, readout, storage."""

from __future__ import annotations

from jlens_reasoning.flenqa.dataset import (
    EXPECTED_PROMPTS,
    MAX_SEQ_LEN,
    extract_bridge,
    load_prompts,
)
from jlens_reasoning.flenqa.readout import (
    read_prompt,
    score_answer,
    select_positions,
    target_token_ids,
)
from jlens_reasoning.flenqa.storage import (
    check_config,
    completed_prompt_ids,
    write_shard,
)

__all__ = [
    "EXPECTED_PROMPTS",
    "MAX_SEQ_LEN",
    "check_config",
    "completed_prompt_ids",
    "extract_bridge",
    "load_prompts",
    "read_prompt",
    "score_answer",
    "select_positions",
    "target_token_ids",
    "write_shard",
]
```

- [ ] **Step 2: Create the notebook**

Create `notebooks/02_build_flenqa_readout.ipynb`. Copy the two bootstrap cells verbatim from `notebooks/_template.ipynb`, then add these cells in order.

Cell — build the prompt table (CPU, no GPU needed):

```python
from pathlib import Path

import pandas as pd
import torch

from jlens_reasoning.config import create_artifact_paths
from jlens_reasoning.flenqa import (
    completed_prompt_ids,
    check_config,
    load_prompts,
    read_prompt,
    score_answer,
    select_positions,
    target_token_ids,
    write_shard,
)

LIMIT = 40          # None for the full 9,862-prompt run
SHARD_SIZE = 500
TOP_K = 25

paths = create_artifact_paths()
prompts = load_prompts(paths)
print(f"{len(prompts)} unique prompts")
prompts.groupby("ctx_size").size()
```

Cell — pre-flight lens-validity gate:

```python
# The lens was fitted on wikitext at n=1000, almost certainly short prompts.
# Whether the Jacobian still holds at 3000 tokens is confounded with the very
# variable under study, so check before spending hours of GPU time.
from experiments.jlens_readout_sanity.constants import MODEL_PATH, LENS_PATH

SPIDER_PROMPT = "The spider spun its web. The spider has eight"
for pad_tokens in (0, 750, 2750):
    padded = ("lorem ipsum dolor sit amet. " * pad_tokens) + SPIDER_PROMPT
    positions = {"final_token": len(tokenizer(padded)["input_ids"]) - 1}
    records = read_prompt(
        model, lens, tokenizer, padded,
        positions=positions,
        targets={"legs": target_token_ids(tokenizer, " legs")[0]},
        layers=lens.source_layers, top_k=TOP_K,
    )
    best = records.groupby("lens_kind")["rank_legs"].min()
    print(f"pad={pad_tokens:>4} jacobian={best['jacobian']:>6} logit={best['logit']:>6}")
```

The J-Lens rank should stay low and beat the logit lens at every length. If the advantage collapses at 3000, stop and build a lens-length control first.

Cell — the run loop with resume:

```python
shard_dir = Path(paths.runs) / "flenqa_readout"
config = {
    "model": MODEL_PATH,
    "lens": LENS_PATH,
    "template_version": "v1",
    "top_k": TOP_K,
    "max_seq_len": MAX_SEQ_LEN,
}
check_config(shard_dir, config)
done = completed_prompt_ids(shard_dir)
print(f"{len(done)} prompts already complete")

true_id = target_token_ids(tokenizer, " True")[0]
false_id = target_token_ids(tokenizer, " False")[0]

todo = prompts[~prompts.prompt_id.isin(done)]
if LIMIT:
    todo = todo.groupby("ctx_size", group_keys=False).head(max(1, LIMIT // 5))

buffer, shard_index = [], len(list(shard_dir.glob("*.parquet")))
for n, (_, row) in enumerate(todo.iterrows(), start=1):
    targets = {"answer_true": true_id, "answer_false": false_id}
    if row["bridge"]:
        targets["bridge"] = target_token_ids(tokenizer, " " + row["bridge"])[-1]

    positions = select_positions(tokenizer, row["prompt"], row)
    records = read_prompt(
        model, lens, tokenizer, row["prompt"],
        positions=positions, targets=targets,
        layers=lens.source_layers, top_k=TOP_K,
    )
    records["prompt_id"] = row["prompt_id"]
    records["problem_id"] = row["problem_id"]
    records["ctx_size"] = row["ctx_size"]
    buffer.append(records)

    if len(buffer) >= SHARD_SIZE or n == len(todo):
        write_shard(shard_dir, f"shard_{shard_index:04d}", pd.concat(buffer))
        buffer, shard_index = [], shard_index + 1
        print(f"{n}/{len(todo)} prompts, {shard_index} shards")
```

Cell — timing extrapolation for the smoke run:

```python
# On the smoke run this replaces the 1.5-3 s/prompt estimate with a measurement.
import time

sample = todo.head(10)
start = time.perf_counter()
for _, row in sample.iterrows():
    read_prompt(
        model, lens, tokenizer, row["prompt"],
        positions=select_positions(tokenizer, row["prompt"], row),
        targets={"answer_true": true_id}, layers=lens.source_layers, top_k=TOP_K,
    )
per_prompt = (time.perf_counter() - start) / len(sample)
print(f"{per_prompt:.2f} s/prompt → {per_prompt * 9862 / 3600:.1f} h for the full run")
```

- [ ] **Step 3: Verify the notebook is valid JSON and imports resolve**

Run:
```bash
uv run python -c "import json; json.load(open('notebooks/02_build_flenqa_readout.ipynb')); print('valid')"
uv run python -c "import jlens_reasoning.flenqa as f; print(sorted(f.__all__))"
```
Expected: `valid`, then the exported names.

- [ ] **Step 4: Document the resource in the README**

Add a section describing `jlens_reasoning.flenqa`: what the resource is, that `load_prompts` returns 9,862 unique prompts with conditions and bridges attached, that `read_prompt` takes prompts and positions from the caller so future experiments can reuse it, and that `notebooks/02_build_flenqa_readout.ipynb` builds the readout with `LIMIT` controlling smoke versus full runs.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest && uv run ruff format --check . && uv run ruff check .`
Expected: all tests pass, formatting clean.

- [ ] **Step 6: Commit**

```bash
git add notebooks/02_build_flenqa_readout.ipynb README.md src/jlens_reasoning/flenqa/__init__.py
git commit -m "feat(flenqa): add the readout notebook and document the resource"
```

---

## Self-Review Notes

**Spec coverage.** Schema facts → Task 1. Templates, dedup, `prompt_id` → Task 2. Conditions, placement, spans → Task 3. Bridge → Task 4. Cache and invariants → Task 5. Storage, resume, config hash → Task 6. Positions → Task 7. Measurements, exact ranks, both lenses, truncation guard → Task 8. Scoring without generation → Task 9. Notebook, pre-flight gate, smoke via `LIMIT`, README → Task 10.

**Deliberate deviations from the spec, all recorded in the artifacts:**
- Placement is computed from **character** offsets, keeping `dataset.py` tokenizer-free and CI-testable; the values are fractions so classification is equivalent (`placement_basis="characters"`).
- `padding_type_effective` takes the first declared value when padding is present. The 300 unpadded groups are the only ones that mix declared padding types, and those resolve to `none`, so no information is lost.

**Three corrections found by running the logic against the real data before writing it down.** Each would have shipped a silent defect:
- Unpadded prompts were classified `facts_first`, not `not_applicable` — the newline joining the two facts leaves `padding_total = 1`, so the "no padding" branch never fired and the `padding_type_effective = "none"` guard would have marked **zero** prompts as the baseline. Now driven by the minimum-length rule, cross-checked against content.
- A four-way rule with a `scattered` bucket swallowed 123 of 144 declared-`middle` prompts. Replaced by a three-way rule that reproduces the declared labels where they are unambiguous (`first` 144/144, `middle` 144/144, `last` 96/144 with the rest genuinely facts-in-the-middle).
- Content equality cannot detect RuleTaker's unpadded baseline at all, because its `mixin` holds sentence expansions rather than the bare `statement` values. The minimum-length rule is what makes the test work across all three tasks.

**Verified before implementation, not assumed:** the bridge rules resolve 300/300 problems with zero question leakage; the bridge never appears in padding and its occurrence count is constant across lengths; dedup gives exactly 9,862 prompts; PIR/MonoRel facts occur exactly once in `mixin` while RuleTaker statements recur 2–14×.

**Known open assumptions** (from the spec, unchanged): the prompt template is ours rather than the authors', so `EXPECTED_PROMPTS = 9862` is asserted in Task 5 and will need updating if templating merges groups; `lens.source_layers` may be a subset of the 36 estimated layers; token counts were measured with Qwen3-4B as a proxy for Qwen3.5-4B.
