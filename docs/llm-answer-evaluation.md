# Evaluating Answers in Simple Model Sanity Tests

**Status:** Normative project policy

**Version:** 1.0

**Date:** 2026-07-15

## Scope

This document defines how `jlens-reasoning` grades simple factual questions in
model sanity tests, such as the spider and France prompts. These tests have a
small, predefined list of accepted factual answers and are intended to confirm
that a loaded model produces a sensible visible answer.

This version does not define evaluation policy for FLenQA, multiple choice,
numeric problems, symbolic mathematics, structured output, open-ended answers,
or model judges. Those require separate, versioned policies in future work.

These rules apply only to the current sanity tests. Later experiments and tests
may require different answer-extraction, normalization, or scoring methods;
those methods must be specified separately rather than inferred from this
policy.

The central rule is:

> Preserve the raw response, extract the model's answer without consulting the
> gold answer, and only then compare it with predefined accepted references.

Searching arbitrary output specifically for the gold answer is prohibited. It
can incorrectly accept text such as `8 is wrong; the spider has 6 legs`.

## Required Evaluation Process

### 1. Preserve the complete raw output

Save the generated token IDs, individually decoded token pieces, complete
decoded text, finish reason, and generation settings. Keep thinking tokens in
this raw artifact. Never replace the raw output with cleaned or extracted text.

Generation must continue until EOS, a declared stop condition, or a declared
safety limit. Do not grade only the first generated token. A visible answer may
span multiple tokens or follow tokenizer-specific whitespace tokens.

### 2. Isolate the visible answer text

Remove reasoning content only through the model's declared protocol:

- remove complete declared reasoning spans such as `<think>...</think>` and
  retain the text outside them.

Version 1 accepts a single raw text output only. Separate reasoning and visible
answer input fields are out of scope.

Do not infer reasoning delimiters from the output after seeing a result.
Unsupported nesting or unbalanced declared delimiters set reasoning status to
`malformed_reasoning`; the response is not graded as an ordinary incorrect
answer.

An answer mentioned only inside removed reasoning content does not count.

### 3. Trim outer whitespace

Remove leading and trailing whitespace from the visible answer text. Preserve
internal whitespace. This ensures a tokenizer-separated leading space does not
make `" 8."` incorrect.

### 4. Extract the front-loaded answer without the gold answer

Take the first non-empty segment of the trimmed visible text, ending at the
first newline or terminal punctuation character (`.`, `!`, or `?`). Then trim
leading and trailing whitespace from that segment. The extractor must not
receive or inspect the accepted references.

These sanity-test prompts are designed to produce a short factual answer at the
start of the visible response. Do not search later text for a reference when
the first segment is absent, ambiguous, or incorrect.

Examples:

- `8. This conclusion...` extracts `8`.
- `Paris. France's capital city...` extracts `Paris`.

### 5. Compare against predefined references

Accepted references must be declared before running the model. Normalize the
extracted answer and every reference using only:

- Unicode NFC normalization;
- leading and trailing whitespace removal;
- Unicode-aware case folding; and
- removal of trailing terminal punctuation (`.`, `!`, and `?`).

Do not remove articles, internal punctuation, signs, units, diacritics, or
arbitrary non-alphanumeric characters. Do not add answer aliases after
inspecting evaluation outputs.

The answer is `correct` when its normalized form exactly equals one normalized
accepted reference. Otherwise it is `incorrect`.

## Statuses

Record generation, reasoning parsing, and answer grading separately.

### Generation status

- `complete`: generation reached EOS or a declared stop condition;
- `truncated`: generation reached the safety limit;
- `generation_error`: inference failed.

### Reasoning status

- `not_present`: the output contains no declared reasoning channel;
- `parsed`: declared reasoning was removed successfully;
- `malformed_reasoning`: declared reasoning output could not be parsed safely.

### Answer status

- `correct`;
- `incorrect`;
- `unparseable`;
- `not_graded`.

If the visible text is empty or no non-empty answer segment can be extracted,
set `answer_status` to `unparseable`.

A generation error or malformed reasoning output produces `not_graded`, not
`incorrect`. A truncated generation remains explicitly `truncated`. If a
complete answer segment was already obtained before the safety limit, it may
still receive `correct` or `incorrect`; otherwise it is `not_graded`.

### Overall pass rule

A sanity test passes when `answer_status` is `correct`, `reasoning_status` is
not `malformed_reasoning`, and generation did not end with `generation_error`.

## Required Spider Regression

```text
Prompt:
The number of legs on the animal that spins webs is

Raw output:
" 8.\n\nThis conclusion is based on..."

Accepted references:
["8", "eight"]

Extracted answer:
"8"

Result:
correct
```

The leading standalone whitespace token is preserved in the raw artifact. It
is removed only from the visible text before gold-blind extraction. The first
answer segment is `8`, which exactly matches an accepted reference.

## Additional Required Regressions

### Visible answer after thinking

```text
<think>A spider has eight legs.</think>
 8.
```

Visible text is `8.`, extracted answer is `8`, and the result is `correct`.

### Answer only inside thinking

```text
<think>The answer is 8.</think>
6
```

Visible text is `6`, so reference `8` produces `incorrect`. The `8` inside
thinking does not count.

## Result Artifact

At minimum, save:

```json
{
  "prompt": "...",
  "accepted_references": ["..."],
  "generation": {
    "settings": {},
    "finish_reason": "eos",
    "status": "complete"
  },
  "raw": {
    "token_ids": [],
    "token_pieces": [],
    "text": "..."
  },
  "evaluation": {
    "reasoning_status": "not_present",
    "evaluation_text": "...",
    "extracted_answer": "...",
    "normalized_answer": "...",
    "matched_reference": null,
    "answer_status": "incorrect"
  }
}
```

Also record the model and tokenizer names and revisions, project commit,
dependency versions, device, and dtype.

## Implementation Requirements

- The reasoning parser, front-loaded answer extractor, normalizer, reference
  comparator, and truncation cleanup must be separate reusable functions.
- Results must record the accepted references and original matched reference.
  The project Git commit identifies the evaluator implementation.
- Extraction must receive `evaluation_text`, but not accepted references.
- Comparison receives only the extracted answer and predefined references.
- Every real artifact failure becomes a regression test before behavior is
  changed.
- Tests must cover tokenizer-separated leading whitespace, multi-token output,
  thinking removal, malformed thinking, truncation, generation errors, case
  differences, terminal punctuation, and truncated punctuation-only output.

## Known Limits

This policy intentionally accepts only front-loaded short factual answers. A
verbose response that hides its answer later may be `incorrect` or
`unparseable` even if a human can infer the answer. References containing
sentence-terminal punctuation require a future extractor version because
punctuation currently ends the answer segment.

## FLenQA Binary Verdict Policy (Version 1.0)

FLenQA has a fixed binary answer shape and is scored without an LLM judge.
Constrained scoring compares the best predefined single-token variants of
`True` and `False` at the final prompt position. Both stored target ranks use
`best_token_rank`, including its lower-token-ID tie break; the lower rank is the
verdict. The raw logits, ranks, verdict, gold label, and correctness are stored.

An optional short generation is diagnostic only. Preserve its raw text, apply
the same gold-blind front-loaded extraction, normalization, and reference
matching defined above, and accept only an extracted `True` or `False`.
Store the generated verdict, correctness, and agreement with constrained
scoring separately. An absent or unparseable generated verdict never changes
the constrained verdict.

## FLenQA Paper-Compatible Generated Verdict

Behavioral comparisons with the published FLenQA results use a separate,
explicitly paper-compatible rule. Search the raw generated response for
standalone `True` and `False` words without receiving the gold label, ignore
case, and use the final occurrence as the verdict. A response with no verdict
is incorrect. A response truncated at the declared generation limit is scored
from the text that was actually generated; it is incorrect when that text has
no verdict.

This rule exists only to reproduce the paper's generated-answer methodology.
It does not replace constrained-logit scoring, and it must not be substituted
for the front-loaded factual evaluator in other experiments.
