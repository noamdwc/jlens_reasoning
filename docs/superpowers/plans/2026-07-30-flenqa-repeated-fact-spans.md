# FLenQA Repeated Fact Spans Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the FLenQA runner preserve every matching key-fact paragraph
without treating repeated semantic content as an unresolved span.

**Architecture:** Keep the existing `SpanDiagnostic` and `LabeledPosition`
contracts. Emit one successful fact diagnostic per matching paragraph, retain
the logical fact ordinal on every diagnostic, derive labels from that ordinal,
and validate that every declared logical fact has at least one fully mapped
span. The existing unique-position execution path continues to deduplicate lens
calls.

**Tech Stack:** Python 3.11, pytest, PyArrow, existing FLenQA span utilities.

---

### Task 1: Preserve repeated fact occurrences

**Files:**
- Modify: `tests/benchmarks/flenqa/test_positions.py`
- Modify: `src/jlens_reasoning/benchmarks/flenqa/positions.py`

- [ ] **Step 1: Write failing repeated-occurrence tests**

Replace the old expectation that a repeated PIR paragraph is ambiguous, and
add end-to-end coverage for repeated RuleTaker facts and identical logical
facts:

```python
def test_resolve_facts_preserves_every_matching_paragraph() -> None:
    prompt = _prompt(
        mixin="Exact.\nRepeated.\nRepeated.",
        key_texts=("Exact.", "Repeated.", "Missing."),
    )

    diagnostics = resolve_key_paragraphs(prompt)

    assert [
        (item.ordinal, item.status, item.match_count)
        for item in diagnostics
    ] == [
        (0, SpanStatus.OK, 1),
        (1, SpanStatus.OK, 2),
        (1, SpanStatus.OK, 2),
        (2, SpanStatus.UNRESOLVED, 0),
    ]
    assert [
        prompt.mixin[item.char_start : item.char_end]
        for item in diagnostics
        if item.status is SpanStatus.OK
    ] == ["Exact.", "Repeated.", "Repeated."]


def test_prepare_prompt_labels_all_repeated_ruletaker_facts() -> None:
    first = "Dave is good. First expansion."
    second = "Dave is small. Second expansion."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{first}\nPadding content.\n{second}\n{first}",
        key_texts=("Dave is good.", "Dave is small."),
        question="Dave is loud.",
        rule="If someone is good and small then they are loud.",
    )

    prepared = validate_prepared_prompt(
        prepare_prompt(prompt, RecordingCharTokenizer())
    )

    fact_a_positions = {
        item.position for item in prepared.positions if item.label == "fact_a_end"
    }
    assert fact_a_positions == {
        prompt.text.index(first) + len(first),
        prompt.text.rindex(first) + len(first),
    }
    fact_spans = {
        (item.char_start, item.char_end)
        for item in _diagnostics(prepared, "fact")
        if item.status is SpanStatus.OK
    }
    assert all(
        not any(start < span_end and end > span_start for span_start, span_end in fact_spans)
        for position in padding_content_positions(prepared)
        for start, end in (prepared.offsets[position],)
    )


def test_identical_logical_facts_share_positions_without_duplicate_execution() -> None:
    fact = "Gary is dumb. Expanded fact."
    prompt = _prompt(
        task="Simplified RuleTaker",
        mixin=f"{fact}\n{fact}",
        key_texts=("Gary is dumb.", "Gary is dumb."),
        question="Gary is red.",
        rule="If someone is dumb then they are red.",
    )

    prepared = validate_prepared_prompt(
        prepare_prompt(prompt, RecordingCharTokenizer())
    )

    fact_a = {
        item.position for item in prepared.positions if item.label == "fact_a_end"
    }
    fact_b = {
        item.position for item in prepared.positions if item.label == "fact_b_end"
    }
    assert fact_a == fact_b
    assert len(fact_a) == 2
    assert set(fact_a) <= set(prepared.unique_positions)
```

Add `padding_content_positions` to the imports from
`jlens_reasoning.benchmarks.flenqa.positions`.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
PYTHONPATH=src \
  /Users/noamc/repos/jlens_reasoning/.venv/bin/python -m pytest \
  tests/benchmarks/flenqa/test_positions.py \
  -k 'preserves_every_matching or labels_all_repeated or identical_logical' -q
```

Expected: failures showing repeated matches still have
`SpanStatus.AMBIGUOUS`, validation raises
`Required FLenQA fact spans are unresolved`, or the repeated labels are absent.

- [ ] **Step 3: Emit and label every matching fact span**

Change `resolve_key_paragraphs` so zero matches retain one unresolved
diagnostic, while every nonempty match produces an `OK` diagnostic carrying
the logical fact ordinal and the total match count:

```python
for ordinal, surface in enumerate(prompt.key_texts):
    if not surface:
        matches: tuple[CharSpan, ...] = ()
    elif prompt.task == RULETAKER_TASK:
        matches = tuple(
            payload_span
            for payload_span in payload_spans
            if surface in prompt.mixin[payload_span.start : payload_span.end]
        )
    else:
        matches = tuple(
            payload_span
            for payload_span in payload_spans
            if prompt.mixin[payload_span.start : payload_span.end] == surface
        )
    if not matches:
        diagnostics.append(
            _diagnostic(
                kind="fact",
                ordinal=ordinal,
                surface=surface,
                status=SpanStatus.UNRESOLVED,
                match_count=0,
            )
        )
        continue
    diagnostics.extend(
        _diagnostic(
            kind="fact",
            ordinal=ordinal,
            surface=surface,
            status=SpanStatus.OK,
            match_count=len(matches),
            span=match,
        )
        for match in matches
    )
```

In `_select_positions`, derive fact and bridge labels from their logical
ordinals rather than zipping diagnostics by occurrence:

```python
for diagnostic in _fact_diagnostics(prepared):
    if diagnostic.status is SpanStatus.OK and diagnostic.token_end is not None:
        selected.append(
            LabeledPosition(
                FACT_LABELS[diagnostic.ordinal],
                diagnostic.token_end - 1,
            )
        )

for diagnostic in bridge_diagnostics:
    if (
        diagnostic.status is SpanStatus.OK
        and diagnostic.token_end is not None
        and diagnostic.fact_ordinal is not None
    ):
        selected.append(
            LabeledPosition(
                BRIDGE_LABELS[diagnostic.fact_ordinal],
                diagnostic.token_end - 1,
            )
        )
```

In `validate_prepared_prompt`, validate successful facts by logical ordinal and
compute expected labels from `len(prepared.prompt.key_texts)`:

```python
facts = _fact_diagnostics(prepared)
expected_ordinals = set(range(len(prepared.prompt.key_texts)))
resolved_ordinals = {
    diagnostic.ordinal
    for diagnostic in facts
    if diagnostic.status is SpanStatus.OK
    and diagnostic.token_start is not None
    and diagnostic.token_end is not None
}
if (
    {diagnostic.ordinal for diagnostic in facts} != expected_ordinals
    or resolved_ordinals != expected_ordinals
    or any(
        diagnostic.status is not SpanStatus.OK
        or diagnostic.token_start is None
        or diagnostic.token_end is None
        for diagnostic in facts
    )
):
    raise ValueError("Required FLenQA fact spans are unresolved")
```

Use `len(prepared.prompt.key_texts)`, not `len(facts)`, when slicing
`FACT_LABELS` and `BRIDGE_LABELS` for the expected-label gate.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command again.

Expected: all selected tests pass.

- [ ] **Step 5: Run the complete FLenQA position tests**

Run:

```bash
PYTHONPATH=src \
  /Users/noamc/repos/jlens_reasoning/.venv/bin/python -m pytest \
  tests/benchmarks/flenqa/test_positions.py -q
```

Expected: all position tests pass.

- [ ] **Step 6: Commit the behavior fix**

```bash
git add \
  src/jlens_reasoning/benchmarks/flenqa/positions.py \
  tests/benchmarks/flenqa/test_positions.py
git commit -m "fix: preserve repeated FLenQA fact spans"
```

### Task 2: Audit the published dataset and verify the repository

**Files:**
- No production-file changes.

- [ ] **Step 1: Run the published-data fact-resolution audit**

Using the downloaded public parquet at `/tmp/flenqa-eval.parquet`, run:

```bash
PYTHONPATH=src \
  /Users/noamc/repos/jlens_reasoning/.venv/bin/python - <<'PY'
import pyarrow.parquet as pq

from jlens_reasoning.benchmarks.flenqa.dataset import deduplicate, normalize_rows
from jlens_reasoning.benchmarks.flenqa.positions import resolve_key_paragraphs
from jlens_reasoning.experiments_utils.spans import SpanStatus

raw_rows = pq.read_table("/tmp/flenqa-eval.parquet").to_pylist()
prompts = deduplicate(normalize_rows(raw_rows, full=True))
bad = [
    (prompt.prompt_id, diagnostic.ordinal, diagnostic.status)
    for prompt in prompts
    for diagnostic in resolve_key_paragraphs(prompt)
    if diagnostic.status is not SpanStatus.OK
]
assert len(prompts) == 9_862
assert bad == []
print("9,862 published prompts have fully resolved fact spans")
PY
```

Expected:

```text
9,862 published prompts have fully resolved fact spans
```

- [ ] **Step 2: Run formatting, lint, and all tests**

Run:

```bash
PYTHONPATH=src \
  /Users/noamc/repos/jlens_reasoning/.venv/bin/python -m ruff format --check .
PYTHONPATH=src \
  /Users/noamc/repos/jlens_reasoning/.venv/bin/python -m ruff check .
PYTHONPATH=src \
  /Users/noamc/repos/jlens_reasoning/.venv/bin/python -m pytest -q
```

Expected: formatting and lint pass; all tests pass.

- [ ] **Step 3: Verify the lockfile and worktree**

Run:

```bash
UV_CACHE_DIR=/tmp/jlens-reasoning-uv-cache uv lock --check
git diff --check
git status --short --branch
```

Expected: the lockfile is current, no whitespace errors are reported, and only
the committed plan plus implementation commits are present.

