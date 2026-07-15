# LLM Evaluation Module Design

**Date:** 2026-07-15

## Purpose

Provide a short, deterministic evaluator for simple factual model sanity tests.
This version does not cover multiple choice, numeric or symbolic problems,
structured output, open-ended answers, or model judges.

## Public API

```python
evaluate(
    output: str | ModelOutput,
    accepted_references: str | Sequence[str],
    evaluator: FactualEvaluator | None = None,
) -> EvaluationResult
```

A string output is shorthand for a complete `ModelOutput`. A string reference
is shorthand for one accepted reference. The runner delegates to a callable
factual evaluator; no registry or framework hierarchy is involved.

Version 1 accepts one raw text field only. Inline `<think>...</think>` parsing
is optional and injected as a plain callable. Separate reasoning and visible
answer input fields are out of scope.

## Data Types

`ModelOutput` is a frozen dataclass containing the complete raw text, token IDs,
decoded token pieces, generation status, finish reason, and optional generation
error. Raw text is never modified. Token collections are immutable tuples with
equal lengths.

`EvaluationResult` is a frozen dataclass containing only:

- the raw `ModelOutput`;
- final `evaluation_text` after reasoning removal, truncation cleanup, and trim;
- extracted answer;
- normalized answer;
- reasoning status; and
- answer status.

Generation status and error are exposed directly from `raw_output`; they are not
stored twice. `passed` implements the policy pass rule.

The result intentionally does not store component IDs, parser names, intermediate
cleanup strings, accepted references, or matched references. Code behavior is
versioned by the project Git commit. Experiment artifacts remain responsible for
saving their prompt and accepted references alongside the result.

## Reusable Utilities

`src/jlens_reasoning/evaluation_utils.py` contains the small reusable functions:

- no-reasoning and inline `<think>` parsing;
- minimal Unicode text normalization;
- gold-blind front-loaded extraction;
- exact accepted-reference matching; and
- safe truncation cleanup.

The `<think>` parser removes complete, non-nested spans. Any leftover opening or
closing tag marks reasoning as malformed.

## Simple Factual Evaluation

The default evaluator:

1. validates that every accepted reference normalizes to non-empty text;
2. returns `not_graded` for a generation error;
3. applies the configured reasoning parser to raw text;
4. returns `not_graded` for malformed reasoning;
5. trims a truncated output back to the last `.`, `!`, `?`, or newline;
6. stores the resulting trimmed value as `evaluation_text`;
7. extracts directly from `evaluation_text` without seeing references;
8. minimally normalizes the answer; and
9. compares it exactly with normalized accepted references.

Whitespace is never a safe truncation boundary. Truncated `8 or` is therefore
`not_graded`, while an earlier complete front-loaded answer may still be graded.

## Testing

Tests cover the spider regression, string and structured inputs, minimal
normalization, gold-blind extraction, inline and malformed thinking, generation
errors, safe truncation including `8 or`, immutable raw artifacts, invalid
references, result invariants, and delegation to another factual evaluator.
