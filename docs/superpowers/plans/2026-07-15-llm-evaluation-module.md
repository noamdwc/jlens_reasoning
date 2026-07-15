# Simple Factual LLM Evaluation Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement each behavior.

**Goal:** Add a short, dependency-free evaluator for simple factual LLM sanity-test outputs.

**Architecture:** Keep the public evaluator and immutable result contract in `evaluation.py`. Put reusable parsing, extraction, normalization, comparison, and truncation helpers in `evaluation_utils.py`. Use plain callables for extension; do not add component IDs, registries, or class hierarchies.

**Tech Stack:** Python 3.11 standard library, pytest, Ruff.

---

## Final contract

- `ModelOutput` preserves raw text and immutable token metadata.
- Version 1 accepts no separate reasoning or visible-answer fields.
- `EvaluationResult` stores raw output, final `evaluation_text`, extracted answer,
  normalized answer, reasoning status, and answer status only.
- Generation status and generation error are read from `raw_output` properties.
- Inline `<think>...</think>` parsing is an optional plain callable.
- Extraction is gold-blind and runs directly on `evaluation_text`.
- Whitespace is never treated as a safe truncation boundary.
- Git commits version evaluator behavior; no runtime component IDs are stored.

## Completed TDD tasks

- [x] Add and validate immutable `ModelOutput`.
- [x] Add the spider, normalization, and gold-blind extraction regressions.
- [x] Add inline, absent, multiple, nested, and malformed thinking regressions.
- [x] Add generation-error and pass-rule regressions.
- [x] Add safe truncation regressions, including `8 or` as `not_graded`.
- [x] Reject empty and normalized-empty references.
- [x] Keep only final evaluation artifacts in `EvaluationResult`.
- [x] Remove component IDs and redundant generation fields.
- [x] Move reusable text-processing functions to `evaluation_utils.py`.
- [x] Prove another factual evaluator can be passed directly to `evaluate()`.
- [x] Run focused tests, full tests, Ruff, formatting, and diff checks.

## Verification commands

```bash
.venv/bin/pytest tests/test_evaluation.py -v
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
wc -l src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py
```
