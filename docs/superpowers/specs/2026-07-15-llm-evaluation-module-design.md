# LLM Evaluation Module Design

**Date:** 2026-07-15

## Purpose

Add a short, deterministic evaluation core for model tests and experiments. Its
first evaluator implements `docs/llm-answer-evaluation.md` for simple factual
sanity questions. Future evaluation types must be addable without changing the
public runner or result contract.

This change adds the standalone module and unit tests only. It does not modify
`readout_sanity.py`, adopt Inspect AI, or define policies for other task types.

## Public API

The module lives at `src/jlens_reasoning/evaluation.py`:

```python
evaluate(
    output: str | ModelOutput,
    expected_answers: str | Sequence[str],
    evaluator: Evaluator | None = None,
    *,
    include_normalized_text: bool = False,
) -> EvaluationResult
```

A string is shorthand for a complete `ModelOutput` with no token metadata or
generation error. A single expected string is shorthand for one accepted
reference. Empty reference collections are rejected.

`Evaluator` is a callable protocol. The runner converts shorthand inputs and
delegates to it. The default is `SimpleFactualEvaluator`; a new evaluation type
is introduced by implementing the protocol, not by editing `evaluate()`.

## Stable Data Types

String enums define generation, reasoning, and answer statuses so misspelled
status values cannot silently enter result artifacts.

`ModelOutput` is an immutable dataclass containing:

- complete raw text;
- token IDs and decoded token pieces when available;
- generation status and finish reason; and
- an optional generation-error message.

`EvaluationResult` is an immutable dataclass containing:

- evaluator name and version;
- generation, reasoning, and answer statuses;
- optional generation-error message;
- raw `ModelOutput`;
- visible text and extracted answer;
- matched accepted reference;
- optional evaluator-specific metadata; and
- optional `normalized_answer`.

`normalized_answer` is `None` by default. When
`include_normalized_text=True`, it contains the exact normalized extracted
answer used for comparison. This option changes observability only, never the
score.

The result exposes a `passed` property implementing the policy's overall pass
rule. Generation errors and malformed reasoning cannot pass.

## Simple Factual Evaluation

`SimpleFactualEvaluator` follows the policy in this order:

1. Stop with `not_graded` on a generation error.
2. Apply only the explicitly configured reasoning parser.
3. Stop with `not_graded` on malformed reasoning.
4. Trim visible text and extract its first non-empty, front-loaded segment.
5. Classify an empty complete response as `unparseable`.
6. Normalize the extracted answer and predefined references with Unicode NFC,
   outer whitespace removal, case folding, and terminal `.`, `!`, or `?`
   removal only.
7. Use exact equality against any normalized accepted reference.

A truncated generation with a complete extractable answer may still be graded.
A truncated generation without one is `not_graded`.

Reasoning removal is an injected callable returning visible text and a
reasoning status. The default declares no reasoning protocol and grades all
text as visible. A supplied `<think>...</think>` parser supports non-nested,
balanced spans and fails closed on malformed delimiters. This keeps model
protocol handling explicit while avoiding an evaluator inheritance hierarchy.

Extraction never receives accepted references. It cannot scan for whichever
substring matches the gold answer.

## Error Handling

Invalid caller configuration, such as no accepted references, raises
`ValueError`. Model behavior is returned as a status rather than raised as an
exception. In particular, incorrect, unparseable, truncated, and
malformed-reasoning outputs remain distinguishable.

## Testing

Unit tests cover every rule and status transition in the current policy:

- plain-string and structured inputs;
- one or multiple accepted references;
- the required spider regression;
- leading whitespace, multi-token text, case, terminal punctuation, and NFC;
- preservation of internal punctuation, signs, articles, and diacritics;
- answers after valid thinking spans and answers only inside thinking;
- absent, multiple, nested, and unbalanced thinking spans;
- empty visible text and no extractable segment;
- generation errors and truncation with and without an answer;
- a wrong front-loaded answer followed by the gold answer;
- normalized text omitted by default and included only when requested; and
- a fake evaluator passed through the public runner without runner changes.

Tests assert the complete result dataclass for critical regressions so a status
or extraction change cannot pass silently.

## Future Integration

When experiment orchestration becomes necessary, a thin Inspect AI scorer can
translate `TaskState.output` and `Target` into this API and translate the result
back into an Inspect `Score`. Inspect remains outside the evaluation core, and
the adapter must call these rules rather than duplicate them.
