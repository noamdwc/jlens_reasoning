# FLenQA Preflight Position Budget

Date: 2026-07-29
Status: approved design

## Problem

`evaluate_lens_preflight` currently requests every prompt position. At the
3,000-token preflight length, `JacobianLens.apply` returns a
`[position, vocabulary]` tensor for every selected layer, so selecting all
positions exhausts Colab host RAM. This also contradicts the FLenQA experiment
design, which requires position subsampling.

## Position policy

Keep the existing 48-position budget.

For prepared FLenQA prompts, semantic positions remain mandatory: fact tails,
bridge locations, the question end, and the final prompt positions. The
selector fills only the unused budget with deterministic padding samples. It
must raise an explicit error if mandatory positions ever exceed 48 rather than
silently dropping semantic positions or exceeding the compute budget.

The synthetic spider preflight has no FLenQA span metadata. It will retain the
complete short semantic prompt suffix, including the final position before
generation, and deterministically sample padding-prefix positions to fill the
remaining budget. Jacobian and logit-lens passes will receive exactly the same
positions.

## Interfaces and data flow

`run_preflight` knows both the original prompt and each padded prompt, so it
will derive the semantic suffix boundary from their untruncated token counts.
It will pass the selected positions into the evaluation callback.
`evaluate_lens_preflight` will use those positions directly and validate that
they are nonempty, unique, ordered, within the tokenized prompt, and no larger
than 48.

The preflight result schema and notebook cells remain unchanged.

## Failure behavior

- Reject an empty or invalid position selection.
- Reject more than 48 selected positions.
- Reject a semantic suffix that alone exceeds the budget.
- Preserve existing token-ID and model-logit equality checks between lens
  modes.

## Tests

Regression tests will prove that:

- a 3,000-token preflight never sends more than 48 positions to either runner;
- the complete semantic suffix and final prompt position are selected;
- padding samples are deterministic;
- both lens modes receive identical positions;
- mandatory FLenQA positions exceeding 48 fail explicitly;
- existing exact-token-count and rank-gate behavior remains intact.
