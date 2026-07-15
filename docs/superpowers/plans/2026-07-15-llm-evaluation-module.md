# Simple Factual LLM Evaluation Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, dependency-free evaluator for simple factual LLM sanity-test outputs, with exhaustive regression tests and auditable immutable results.

**Architecture:** Put the complete production implementation in one module and its tests in one test module. `evaluate()` only converts shorthand inputs and delegates to a callable factual evaluator. The default evaluator composes a declared reasoning parser, a gold-blind front-loaded extractor, minimal normalization, and exact reference comparison. Frozen dataclasses preserve raw output, statuses, normalized text, references, and component versions.

**Tech Stack:** Python 3.11 standard library (`dataclasses`, `enum`, `re`, `typing`, `unicodedata`), pytest, Ruff.

---

## Scope and simplicity guardrails

- Production file: `src/jlens_reasoning/evaluation.py`
- Test file: `tests/test_evaluation.py`
- No Inspect AI dependency, registry, abstract base class, plugin loader, task-family hierarchy, or integration with `readout_sanity.py` in this change.
- Target at most 250 physical lines in the production module. If it exceeds that, remove duplication or unnecessary abstraction before splitting files.
- Keep extraction gold-blind: neither the extractor nor the reasoning parser may receive accepted references.

### Task 1: Add the immutable input contract

**Files:**

- Create: `src/jlens_reasoning/evaluation.py`
- Create: `tests/test_evaluation.py`

- [ ] Write failing tests for string-valued enums and `ModelOutput` validation:

```python
import pytest

from jlens_reasoning.evaluation import GenerationStatus, ModelOutput


def test_model_output_preserves_raw_token_artifact() -> None:
    output = ModelOutput(
        text=" 8.",
        token_ids=(220, 23, 13),
        token_pieces=(" ", "8", "."),
        finish_reason="eos",
    )

    assert output.text == " 8."
    assert output.token_ids == (220, 23, 13)
    assert output.token_pieces == (" ", "8", ".")
    assert output.generation_status is GenerationStatus.COMPLETE


def test_model_output_rejects_mismatched_token_metadata() -> None:
    with pytest.raises(ValueError, match="same length"):
        ModelOutput(text="8", token_ids=(23,), token_pieces=())


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (GenerationStatus.GENERATION_ERROR, None),
        (GenerationStatus.GENERATION_ERROR, ""),
        (GenerationStatus.COMPLETE, "boom"),
        (GenerationStatus.TRUNCATED, "boom"),
    ],
)
def test_model_output_rejects_inconsistent_generation_error(
    status: GenerationStatus, message: str | None
) -> None:
    with pytest.raises(ValueError):
        ModelOutput(
            text="",
            generation_status=status,
            generation_error=message,
        )
```

- [ ] Run the focused tests and confirm they fail because the module does not exist:

```bash
.venv/bin/pytest tests/test_evaluation.py -q
```

Expected: collection error for `jlens_reasoning.evaluation`.

- [ ] Implement only the input types:

```python
from dataclasses import dataclass
from enum import StrEnum


class GenerationStatus(StrEnum):
    COMPLETE = "complete"
    TRUNCATED = "truncated"
    GENERATION_ERROR = "generation_error"


class ReasoningStatus(StrEnum):
    NOT_PRESENT = "not_present"
    PARSED = "parsed"
    MALFORMED = "malformed_reasoning"


class AnswerStatus(StrEnum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNPARSEABLE = "unparseable"
    NOT_GRADED = "not_graded"


@dataclass(frozen=True, slots=True)
class ComponentId:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not self.name or not self.version:
            raise ValueError("component name and version must be non-empty")


@dataclass(frozen=True, slots=True)
class ModelOutput:
    text: str
    token_ids: tuple[int, ...] = ()
    token_pieces: tuple[str, ...] = ()
    generation_status: GenerationStatus = GenerationStatus.COMPLETE
    finish_reason: str | None = None
    generation_error: str | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.token_pieces):
            raise ValueError("token IDs and pieces must have the same length")
        has_error = bool(self.generation_error)
        expects_error = self.generation_status is GenerationStatus.GENERATION_ERROR
        if has_error != expects_error:
            raise ValueError("generation_error status and message must agree")
```

- [ ] Run the focused tests and confirm they pass.

- [ ] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: add immutable evaluation contract"
```

### Task 2: Implement simple factual scoring and provenance

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] Add failing tests for the spider regression, string shorthand, multiple references, outer whitespace, case folding, terminal punctuation, NFC equivalence, internal punctuation preservation, and gold-blind first-segment extraction.

The critical assertions are:

```python
def test_spider_regression() -> None:
    result = evaluate(
        " 8.\n\nThis conclusion is based on...",
        ("8", "eight"),
    )

    assert result.raw_output.text == " 8.\n\nThis conclusion is based on..."
    assert result.visible_text == "8.\n\nThis conclusion is based on..."
    assert result.extracted_answer == "8"
    assert result.normalized_answer == "8"
    assert result.matched_reference == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed


def test_extraction_does_not_search_for_reference() -> None:
    result = evaluate("6. The answer is 8.", "8")

    assert result.extracted_answer == "6"
    assert result.answer_status is AnswerStatus.INCORRECT


def test_normalization_does_not_remove_meaningful_characters() -> None:
    assert evaluate("Cote d'Ivoire", "Côte d'Ivoire").answer_status is AnswerStatus.INCORRECT
```

- [ ] Add the frozen result contract, callable protocol, and default evaluator. Keep helpers private and single-purpose:

```python
@dataclass(frozen=True, slots=True)
class EvaluationResult:
    evaluator: ComponentId
    reasoning_parser: ComponentId
    extractor: ComponentId
    normalizer: ComponentId
    accepted_references: tuple[str, ...]
    generation_status: GenerationStatus
    reasoning_status: ReasoningStatus
    answer_status: AnswerStatus
    generation_error: str | None
    raw_output: ModelOutput
    visible_text: str
    extracted_answer: str | None
    normalized_answer: str | None
    matched_reference: str | None

    @property
    def passed(self) -> bool:
        return (
            self.answer_status is AnswerStatus.CORRECT
            and self.reasoning_status is not ReasoningStatus.MALFORMED
            and self.generation_status is not GenerationStatus.GENERATION_ERROR
        )


class FactualEvaluator(Protocol):
    def __call__(
        self, output: ModelOutput, accepted_references: tuple[str, ...]
    ) -> EvaluationResult: ...


def evaluate(
    output: str | ModelOutput,
    accepted_references: str | Sequence[str],
    evaluator: FactualEvaluator | None = None,
) -> EvaluationResult:
    model_output = ModelOutput(output) if isinstance(output, str) else output
    references = (
        (accepted_references,)
        if isinstance(accepted_references, str)
        else tuple(accepted_references)
    )
    return (evaluator or SimpleFactualEvaluator())(model_output, references)
```

Use these exact initial component IDs:

```python
SIMPLE_FACTUAL = ComponentId("simple_factual", "v1")
NO_REASONING = ComponentId("none", "v1")
FRONT_LOADED = ComponentId("front_loaded_segment", "v1")
MINIMAL_TEXT = ComponentId("minimal_text", "v1")
```

Implement `_normalize()` with Unicode NFC, `strip()`, `casefold()`, and trailing `.?!` removal only. Implement `_extract()` as the stripped text before the first `.`, `!`, `?`, or newline. Validate references before scoring and preserve their original surface forms in the result.

- [ ] Validate result consistency in `EvaluationResult.__post_init__`: generation fields equal the raw output, any match belongs to accepted references, and `correct` requires extracted/normalized answers plus a match.

- [ ] Run focused tests and confirm they pass.

- [ ] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: evaluate simple factual outputs"
```

### Task 3: Add declared reasoning parsing and generation failures

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] Add failing tests for:

```python
def test_visible_answer_after_thinking_is_graded() -> None:
    result = evaluate(
        "<think>A spider has eight legs.</think>\n 8.",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=THINK_TAGS),
    )
    assert result.reasoning_status is ReasoningStatus.PARSED
    assert result.extracted_answer == "8"
    assert result.passed


def test_answer_only_inside_thinking_does_not_count() -> None:
    result = evaluate(
        "<think>The answer is 8.</think>\n6",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=THINK_TAGS),
    )
    assert result.extracted_answer == "6"
    assert result.answer_status is AnswerStatus.INCORRECT


@pytest.mark.parametrize(
    "text",
    [
        "<think>unfinished",
        "stray</think>8",
        "<think>outer <think>nested</think></think>8",
    ],
)
def test_malformed_thinking_is_not_graded(text: str) -> None:
    result = evaluate(
        text,
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=THINK_TAGS),
    )
    assert result.reasoning_status is ReasoningStatus.MALFORMED
    assert result.answer_status is AnswerStatus.NOT_GRADED
    assert not result.passed
```

Also test absent tags, multiple balanced non-nested spans, and a `generation_error` result with the raw text and message intact.

- [ ] Implement a small frozen `ReasoningParser` containing its `ComponentId` and callable. Export `THINK_TAGS`; keep the no-reasoning parser as the default. Parse balanced, non-nested `<think>...</think>` spans and fail closed on nesting, stray closers, or unclosed openers.

- [ ] In `SimpleFactualEvaluator`, validate references first, then return `not_graded` for generation errors or malformed reasoning. Use one private result-construction helper so every exit records identical provenance fields without duplicating constructors.

- [ ] Run focused tests and commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: parse declared thinking output"
```

### Task 4: Enforce safe truncation boundaries

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] Add failing truncation regressions:

```python
@pytest.mark.parametrize("text", ["8 or", "8", "partial answer"])
def test_ambiguous_truncation_is_not_graded(text: str) -> None:
    output = ModelOutput(text, generation_status=GenerationStatus.TRUNCATED)
    result = evaluate(output, "8")

    assert result.raw_output.text == text
    assert result.generation_status is GenerationStatus.TRUNCATED
    assert result.answer_status is AnswerStatus.NOT_GRADED


@pytest.mark.parametrize(
    "text",
    ["8.\nThis sentence is incom", "8! trailing frag"],
)
def test_complete_front_loaded_answer_survives_truncation(text: str) -> None:
    output = ModelOutput(text, generation_status=GenerationStatus.TRUNCATED)
    result = evaluate(output, "8")

    assert result.raw_output.text == text
    assert result.extracted_answer == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed
```

Add a newline-boundary case and an empty safe prefix case.

- [ ] Implement one safe-prefix helper. Its only boundaries are `.`, `!`, `?`, and newline. Keep the last boundary in evaluation text. If no boundary exists, return `not_graded`. Whitespace may be trimmed around the retained prefix but must never establish completeness.

```python
def _safe_truncated_text(text: str) -> str | None:
    boundary = max(text.rfind(character) for character in ".!?\n")
    return None if boundary < 0 else text[: boundary + 1].strip()
```

- [ ] Run focused tests. Explicitly verify truncated `"8 or"` cannot pass.

- [ ] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: reject ambiguous truncated answers"
```

### Task 5: Harden validation, extension point, and final audit

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [ ] Add tests rejecting an empty reference tuple, empty reference strings, and references that normalize to empty (`"..."`, `" ? "`).

- [ ] Add tests for every `passed` branch, inconsistent `EvaluationResult` construction (use `dataclasses.replace`), immutable tuple fields, and `normalized_answer` on both correct and incorrect results.

- [ ] Prove the open-closed extension point with a tiny test-only callable evaluator passed to `evaluate()`. Assert the runner delegates unchanged without a registry or modification to production dispatch logic.

- [ ] Run all verification commands:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
wc -l src/jlens_reasoning/evaluation.py
```

Expected: all tests and Ruff checks pass; production module is at most 250 physical lines.

- [ ] Manually audit the final module against both policy documents:

  - raw output and token metadata remain unchanged;
  - extraction never receives references;
  - normalized answer, accepted references, and component IDs are always recorded;
  - answer, reasoning, and generation statuses remain separate;
  - empty complete visible output is `unparseable`;
  - generation error and malformed reasoning are `not_graded`;
  - ambiguous truncation such as `8 or` is `not_graded`;
  - each required spider/thinking example has a named regression test; and
  - no functionality outside simple factual evaluation was introduced.

- [ ] If the production file exceeds the line target or contains repeated branches, simplify it before completion. Do not create a framework layer merely to reduce physical line count.

- [ ] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "test: harden factual output evaluation"
```
