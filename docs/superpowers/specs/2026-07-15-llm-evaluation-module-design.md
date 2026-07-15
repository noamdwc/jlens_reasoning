# LLM Evaluation Module Design

**Date:** 2026-07-15

## Purpose

Add a short, deterministic evaluation core for simple factual model sanity
tests. Its first implementation follows `docs/llm-answer-evaluation.md`.
Future factual evaluators must be addable without changing the runner or result
contract.

This change adds a standalone module and unit tests only. It does not modify
`readout_sanity.py`, adopt Inspect AI, or promise a public API for multiple
choice, numeric, symbolic, structured, open-ended, or model-judged evaluation.

## Public API

The module lives at `src/jlens_reasoning/evaluation.py`:

```python
evaluate(
    output: str | ModelOutput,
    accepted_references: str | Sequence[str],
    evaluator: FactualEvaluator | None = None,
) -> EvaluationResult
```

A string is shorthand for a complete `ModelOutput` with empty token metadata.
A reference string is shorthand for a one-item reference tuple. The runner
converts these inputs and delegates to the selected `FactualEvaluator` callable
protocol. The default is `SimpleFactualEvaluator`.

The stability promise covers simple factual evaluators only. A future task
family may use a different entry point and result type.

## Immutable Data Types

String enums define generation, reasoning, and answer statuses.

`ModelOutput` is a frozen dataclass containing:

- complete raw text;
- token IDs as `tuple[int, ...]`;
- decoded token pieces as `tuple[str, ...]`;
- generation status and optional finish reason; and
- an optional generation-error message.

The token tuples must have equal lengths. A `generation_error` status requires
a non-empty error message; other generation statuses reject an error message.

`EvaluationResult` is a frozen dataclass containing:

- evaluator name and version;
- reasoning-parser, extractor, and normalizer names and versions;
- accepted references as `tuple[str, ...]`;
- generation, reasoning, and answer statuses;
- optional generation-error message;
- raw `ModelOutput`;
- visible text and extracted answer;
- normalized answer and matched reference; and
- the policy-defined `passed` property.

`normalized_answer` is always present. It contains the exact normalized answer
used for comparison, or `None` when no answer was extracted. There is no flag
that hides it.

Construction rejects inconsistent results: mismatched generation fields, a
matched reference not in the accepted tuple, or `correct` without an extracted
answer and match. Reference normalization is performed by the evaluator before
result construction.

## Simple Factual Evaluation

`SimpleFactualEvaluator` performs these steps:

1. Return `not_graded` for a generation error.
2. Apply only the explicitly configured reasoning parser.
3. Return `not_graded` for malformed reasoning.
4. For truncated generation, derive safe evaluation text as described below.
5. Trim the visible text and extract its first non-empty front-loaded segment.
6. Return `unparseable` for an empty complete response.
7. Normalize the answer and references with Unicode NFC, outer whitespace
   removal, case folding, and trailing `.`, `!`, or `?` removal only.
8. Compare by exact equality with any accepted normalized reference.

Extraction never receives accepted references and cannot search for the gold
answer.

Reasoning removal is an injected callable with a declared name and version. It
returns visible text and a reasoning status. The default declares no reasoning
protocol. A supplied `<think>...</think>` parser accepts balanced, non-nested
spans and fails closed on malformed delimiters.

## Truncated Generation

`ModelOutput.text` always retains the complete truncated output. After reasoning
parsing and before answer extraction, the evaluator removes an incomplete
trailing fragment from a separate evaluation-text value:

1. Cut after the last `.`, `!`, `?`, or newline, keeping that boundary.
2. If none exists, cut at the last whitespace boundary only to discard a
   partial final word.
3. If no safe boundary exists, return `not_graded`.

If safe text retains an earlier complete front-loaded answer, grade it normally
while keeping generation status `truncated`. If it does not retain an
extractable answer, return `not_graded`, not `unparseable`.

## Errors and Validation

Invalid caller input raises `ValueError`; model behavior is represented by
statuses. Empty reference collections and references that become empty under
the declared normalizer are invalid. Incorrect, unparseable, truncated,
malformed-reasoning, and generation-error outputs remain distinguishable.

All parser, extractor, and normalizer components expose fixed names and
versions. These identifiers and the accepted references are stored in every
result so a score can be audited without relying on process-global state.
The initial identifiers are `simple_factual/v1`, `none/v1` or `think_tags/v1`,
`front_loaded_segment/v1`, and `minimal_text/v1`.

## Testing

Unit tests cover:

- string shorthand, structured output, and multiple references;
- the spider regression, case, punctuation, whitespace, NFC, and prohibited
  broad normalization;
- valid, absent, multiple, nested, and unbalanced thinking spans;
- empty and normalized-empty references;
- empty visible text, generation errors, and every pass-rule branch;
- truncated text ending after a sentence, line, partial word, or no safe
  boundary, while asserting raw text is unchanged;
- token metadata and generation-field validation;
- normalized answer and all component names and versions in results;
- gold-blind extraction; and
- a second factual evaluator passed through the runner unchanged.

Critical regressions assert the complete result dataclass so extraction,
provenance, or status changes cannot pass silently.

## Future Integration

A future Inspect AI adapter may translate `TaskState.output` and `Target` into
this API and translate the result into an Inspect `Score`. It must call this
evaluation core rather than duplicate its rules.
