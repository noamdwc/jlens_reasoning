# Simple Factual LLM Evaluation Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, dependency-free evaluator for simple factual LLM sanity-test outputs, with exhaustive regression tests and auditable immutable results.

**Architecture:** Put the complete production implementation in one module and its tests in one test module. `evaluate()` only converts shorthand inputs and delegates to a callable factual evaluator. Version 1 accepts one raw text output; an optional declared parser may remove inline `<think>...</think>` spans. The default evaluator produces one final `evaluation_text`, extracts directly from it, minimally normalizes the answer, and compares it exactly with accepted references.

**Tech Stack:** Python 3.11 standard library (`dataclasses`, `enum`, `re`, `typing`, `unicodedata`), pytest, Ruff.

---

## Scope and simplicity guardrails

- Production file: `src/jlens_reasoning/evaluation.py`
- Test file: `tests/test_evaluation.py`
- No Inspect AI dependency, registry, abstract base class, plugin loader, task-family hierarchy, or integration with `readout_sanity.py` in this change.
- Version 1 has no separate reasoning field or visible-answer field. `ModelOutput.text` is the single source string.
- Never modify `raw_output.text`. Build `evaluation_text` separately by applying declared reasoning removal, safe truncation cleanup, and outer trimming in that order.
- Store no intermediate reasoning-removed or truncation-cleaned strings. Apart from statuses and required provenance, the only text-processing artifacts are the raw model output, final `evaluation_text`, extracted answer, and normalized answer. Extraction runs directly on `evaluation_text`.
- `evaluation_text` is the implementation name for the policy's gradeable visible text after all pre-extraction cleanup; it is not a second model-output field.
- Target at most 250 physical lines in the production module. If it exceeds that, remove duplication or unnecessary abstraction before splitting files.
- Keep extraction gold-blind: neither the extractor nor the reasoning parser may receive accepted references.

### Task 1: Add the immutable input contract

**Files:**

- Create: `src/jlens_reasoning/evaluation.py`
- Create: `tests/test_evaluation.py`

- [x] Write failing tests for string-valued enums and `ModelOutput` validation:

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

- [x] Run the focused tests and confirm they fail because the module does not exist:

```bash
.venv/bin/pytest tests/test_evaluation.py -q
```

Expected: collection error for `jlens_reasoning.evaluation`.

- [x] Implement only the input types:

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

- [x] Run the focused tests and confirm they pass.

- [x] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: add immutable evaluation contract"
```

### Task 2: Implement simple factual scoring and provenance

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [x] Add failing tests for the spider regression, string shorthand, multiple references, outer whitespace, case folding, terminal punctuation, NFC equivalence, internal punctuation preservation, and gold-blind first-segment extraction.

The critical assertions are:

```python
def test_spider_regression() -> None:
    result = evaluate(
        " 8.\n\nThis conclusion is based on...",
        ("8", "eight"),
    )

    assert result.raw_output.text == " 8.\n\nThis conclusion is based on..."
    assert result.evaluation_text == "8.\n\nThis conclusion is based on..."
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

- [x] Add the frozen result contract, callable protocol, and default evaluator. Keep helpers private and single-purpose:

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
    evaluation_text: str
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

Introduce the small parser value type with the default no-op parser in this task:

```python
ReasoningFunction = Callable[[str], tuple[str, ReasoningStatus]]


@dataclass(frozen=True, slots=True)
class ReasoningParser:
    component_id: ComponentId
    parse: ReasoningFunction

    def __call__(self, text: str) -> tuple[str, ReasoningStatus]:
        return self.parse(text)


def _no_reasoning(text: str) -> tuple[str, ReasoningStatus]:
    return text, ReasoningStatus.NOT_PRESENT
```

Keep parser objects distinct from their component IDs:

```python
SIMPLE_FACTUAL = ComponentId("simple_factual", "v1")
FRONT_LOADED = ComponentId("front_loaded_segment", "v1")
MINIMAL_TEXT = ComponentId("minimal_text", "v1")
NO_REASONING_ID = ComponentId("none", "v1")
NO_REASONING_PARSER = ReasoningParser(NO_REASONING_ID, _no_reasoning)
```

`SimpleFactualEvaluator.reasoning_parser` is a `ReasoningParser` and defaults to `NO_REASONING_PARSER`. `EvaluationResult.reasoning_parser` stores only that object's `component_id` for provenance.

Implement `_normalize()` with Unicode NFC, `strip()`, `casefold()`, and trailing `.?!` removal only. Implement `_extract()` as the stripped text before the first `.`, `!`, `?`, or newline. It receives only the final `evaluation_text`. Validate references before scoring and preserve their original surface forms in the result.

- [x] Validate result consistency in `EvaluationResult.__post_init__`: generation fields equal the raw output, any match belongs to accepted references, and `correct` requires extracted/normalized answers plus a match.

- [x] Run focused tests and confirm they pass.

- [x] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: evaluate simple factual outputs"
```

### Task 3: Add declared reasoning parsing and generation failures

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [x] Add failing tests for:

```python
def test_answer_after_inline_thinking_is_graded() -> None:
    result = evaluate(
        "<think>A spider has eight legs.</think>\n 8.",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=THINK_TAGS_PARSER),
    )
    assert result.reasoning_status is ReasoningStatus.PARSED
    assert result.evaluation_text == "8."
    assert result.extracted_answer == "8"
    assert result.passed


def test_answer_only_inside_thinking_does_not_count() -> None:
    result = evaluate(
        "<think>The answer is 8.</think>\n6",
        "8",
        evaluator=SimpleFactualEvaluator(reasoning_parser=THINK_TAGS_PARSER),
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
        evaluator=SimpleFactualEvaluator(reasoning_parser=THINK_TAGS_PARSER),
    )
    assert result.reasoning_status is ReasoningStatus.MALFORMED
    assert result.answer_status is AnswerStatus.NOT_GRADED
    assert not result.passed
```

Also test absent tags, multiple balanced non-nested spans, and a `generation_error` result with the raw text and message intact.

- [x] Add only the inline `<think>...</think>` implementation to the `ReasoningParser` type introduced in Task 2. Define `THINK_TAGS_ID = ComponentId("think_tags", "v1")` and export the distinct parser object `THINK_TAGS_PARSER = ReasoningParser(THINK_TAGS_ID, _parse_think_tags)`. Keep `NO_REASONING_PARSER` as the default. Parse balanced, non-nested spans and fail closed on nesting, stray closers, or unclosed openers. Do not add separate reasoning or visible-answer inputs.

- [x] In `SimpleFactualEvaluator`, validate references first, then return `not_graded` for generation errors or malformed reasoning. For a gradeable output, keep `raw_output.text` untouched, use one local string as it passes through reasoning removal, truncation cleanup, and `strip()`, store only the resulting `evaluation_text`, and call `_extract(evaluation_text)` directly. Use one private result-construction helper so every exit records identical provenance fields without duplicating constructors.

- [x] Run focused tests and commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: parse declared thinking output"
```

### Task 4: Enforce safe truncation boundaries

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [x] Add failing truncation regressions:

```python
@pytest.mark.parametrize("text", ["8 or", "8", "partial answer"])
def test_ambiguous_truncation_is_not_graded(text: str) -> None:
    output = ModelOutput(text, generation_status=GenerationStatus.TRUNCATED)
    result = evaluate(output, "8")

    assert result.raw_output.text == text
    assert result.evaluation_text == ""
    assert result.extracted_answer is None
    assert result.normalized_answer is None
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
    assert result.evaluation_text in {"8.", "8!"}
    assert result.extracted_answer == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed
```

Add a newline-boundary case and an empty safe prefix case.

- [x] Implement one safe-prefix helper. Its only boundaries are `.`, `!`, `?`, and newline. Keep the last boundary in evaluation text. If no boundary exists, return `not_graded`. Whitespace may be trimmed around the retained prefix but must never establish completeness.

```python
def _safe_truncated_text(text: str) -> str | None:
    boundary = max(text.rfind(character) for character in ".!?\n")
    return None if boundary < 0 else text[: boundary + 1].strip()
```

- [x] Run focused tests. Explicitly verify truncated `"8 or"` cannot pass.

- [x] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "feat: reject ambiguous truncated answers"
```

### Task 5: Harden validation, extension point, and final audit

**Files:**

- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `tests/test_evaluation.py`

- [x] Add tests rejecting an empty reference tuple, empty reference strings, and references that normalize to empty (`"..."`, `" ? "`).

- [x] Add tests for every `passed` branch, inconsistent `EvaluationResult` construction (use `dataclasses.replace`), immutable tuple fields, and `normalized_answer` on both correct and incorrect results. Inspect the result dataclass fields once to assert that `evaluation_text` exists and no `visible_text`, reasoning-removed-text, or truncation-cleanup-text field exists.

- [x] Prove the open-closed extension point with a tiny test-only callable evaluator passed to `evaluate()`. Assert the runner delegates unchanged without a registry or modification to production dispatch logic.

- [x] Run all verification commands:

```bash
.venv/bin/pytest tests/test_evaluation.py -v
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
wc -l src/jlens_reasoning/evaluation.py
```

Expected: all tests and Ruff checks pass; production module is at most 250 physical lines.

- [x] Manually audit the final module against both policy documents:

  - raw output and token metadata remain unchanged;
  - extraction never receives references;
  - normalized answer, accepted references, and component IDs are always recorded;
  - answer, reasoning, and generation statuses remain separate;
  - `evaluation_text` is the only processed-text field and extraction consumes it directly;
  - empty complete `evaluation_text` is `unparseable`;
  - generation error and malformed reasoning are `not_graded`;
  - ambiguous truncation such as `8 or` is `not_graded`;
  - each required spider/thinking example has a named regression test; and
  - no functionality outside simple factual evaluation was introduced.

- [x] If the production file exceeds the line target or contains repeated branches, simplify it before completion. Do not create a framework layer merely to reduce physical line count.

- [x] Commit:

```bash
git add src/jlens_reasoning/evaluation.py tests/test_evaluation.py
git commit -m "test: harden factual output evaluation"
```
