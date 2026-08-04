# FLenQA Accuracy by Prompt Length

Date: 2026-08-04
Status: approved simplified design

## Purpose

Add one Colab notebook that evaluates Qwen on all 9,862 unique FLenQA prompts
and plots binary reasoning accuracy by prompt length. Match the paper's
behavioral method: greedy generation, nominal length buckets of 250, 500, 1000,
2000, and 3000, and the final standalone case-insensitive `True` or `False` in
the response.

Favor a transparent one-off research notebook over a reusable experiment
framework. The complete generation loop, result assembly, persistence, and
analysis remain visible in notebook cells.

## Goals

- Evaluate every unique final FLenQA prompt once by default.
- Preserve the paper-compatible generated-answer evaluator in shared, tested
  evaluation code.
- Save enough per-prompt information to audit the scores.
- Show the paper-weighted random-placement curve and an all-unique-prompt curve.
- Keep the implementation small and easy to read.

## Non-goals

- Resume after interruption.
- Atomic shards, run manifests, configuration hashes, or reusable runners.
- Jacobian Lens or Logit Lens computation.
- Chain-of-thought prompting or coverage analysis.
- A general analysis API for other notebooks.
- Reproducing the paper's closed models or exact tokenizer.

## Paper-compatible method

The x-axis uses the released dataset's declared `ctx_size`, matching the paper's
five nominal buckets. Exact Qwen tokenizer length is stored as a diagnostic.

The shared evaluator searches raw generated text for standalone `True` and
`False` words without receiving the gold label and selects the final match. A
response with no verdict is incorrect. This evaluator is explicitly for paper
compatibility and does not replace constrained-logit FLenQA scoring or the
front-loaded evaluator used by other experiments.

The paper's main curve filters source rows to random/non-adjacent placement and
weights both padding types, producing 600 observations per length. The notebook
evaluates deduplicated prompts once and stores `paper_weight`, the number of
random-placement source rows represented by that prompt. Weighted aggregation
then reproduces the source-row counts without repeating inference. The second
curve assigns weight one to each unique prompt.

## Notebook workflow

Add `notebooks/flenqa_accuracy.ipynb` with the canonical Drive loader and these
visible stages:

1. initialize Colab with a required GPU and W&B disabled;
2. load imports and fixed settings;
3. load, validate, and normalize all 12,000 source rows;
4. deduplicate and assert exactly 9,862 final prompts;
5. load the local Qwen model and tokenizer;
6. define deterministic generation returning `ModelOutput`;
7. loop over every prompt, tokenize without truncation, generate, evaluate, and
   append one result record;
8. construct one typed Arrow table and write `results.parquet` beneath
   `runs/flenqa-accuracy/`;
9. compute and plot the paper-weighted curve;
10. compute and plot the unique-prompt curve;
11. show task, verdict-frequency, and exact-token-length diagnostics.

The run uses `do_sample=False`, a 64-new-token safety cap, and a 4,096-token
input limit. The committed notebook contains no outputs, execution counts, or
credentials.

## Result table

One row per unique prompt stores:

- prompt ID, problem ID, task, gold label, and prompt text;
- nominal context size, exact input-token count, and `paper_weight`;
- generated token IDs, token pieces, raw generated text, generation status,
  and finish reason;
- extracted verdict and correctness.

The notebook writes the table only after all 9,862 prompts finish. An
interrupted run therefore restarts from the beginning. This is an explicit
tradeoff of the simplified design.

## Analysis

The paper-weighted summary groups by nominal context size and sums
`paper_weight`, `correct * paper_weight`, and
`(verdict is missing) * paper_weight`. It asserts 600 weighted observations per
length.

The unique-prompt summary groups every result row once and asserts counts of
300, 2,368, 2,394, 2,400, and 2,400 across the five lengths.

Additional notebook cells show per-task unique-prompt accuracy, counts of
`True`, `False`, and missing verdicts, and minimum/median/maximum exact Qwen
token length in each nominal bucket. The plots do not add confidence intervals,
matching the paper.

## Failure handling

- Full dataset count or deduplication count mismatches stop before inference.
- Inputs longer than 4,096 tokens fail rather than truncate.
- Generation exceptions stop the run rather than become incorrect answers.
- A generated response reaching the 64-token cap is marked truncated and scored
  from the available text; no verdict is incorrect.
- Result-count and bucket-count assertions run before the Parquet file is
  written or plotted.

## Verification

CPU tests cover the shared paper-compatible evaluator: final occurrence, word
boundaries, case insensitivity, repeated verdicts, missing verdicts, and
truncated responses. Notebook tests enforce the canonical loader, full dataset
and dedup counts, deterministic generation settings, explicit input limit,
paper weighting, unique-prompt aggregation, one Parquet output, no saved
outputs, and no credentials.

Final verification runs formatting, lint, the full CPU suite, and the lockfile
check. The model-backed notebook remains a Colab workflow and is not executed in
CI.
