# FLenQA Accuracy by Prompt Length

Date: 2026-08-04
Status: approved design

## Purpose

Add a Colab notebook that evaluates Qwen on every unique FLenQA prompt and
shows how binary reasoning accuracy changes with prompt length. The headline
analysis follows the paper's behavioral evaluation: deterministic generation,
the paper's five nominal length buckets, and the final case-insensitive
`True`/`False` verdict in the response.

The run evaluates all 9,862 unique final prompts by default. It preserves exact
model-token lengths and source-row provenance so the notebook can show both a
paper-weighted curve and a unique-prompt curve without rerunning inference.

## Goals

- Evaluate each unique final FLenQA prompt exactly once.
- Match the paper's direct-prompt answer extraction and nominal length buckets.
- Preserve raw generations and enough metadata to audit every score.
- Resume safely after an interrupted multi-hour Colab run.
- Plot the paper-comparable random-placement result and the full unique-prompt
  result from one set of predictions.
- Keep grading and runner behavior in tested package code rather than notebook
  cells.

## Non-goals

- Chain-of-thought prompting or coverage analysis.
- Jacobian Lens or Logit Lens computation.
- Replacing the existing constrained-logit FLenQA scoring policy.
- Reproducing the paper's closed-model results or exact tokenizer.
- Statistical modeling of causal effects from padding type or placement.

## Paper-compatible method

The paper reports accuracy at nominal input sizes 250, 500, 1000, 2000, and
3000. It treats a sample as belonging to nominal size `N` when the GPT-4 token
count is within `N ± 70`, but the released dataset already carries the nominal
`ctx_size` used in its figures. The notebook uses that declared value for its
x-axis and stores Qwen's exact tokenizer length separately.

The paper evaluates direct prompts with deterministic decoding and identifies a
response by taking the last case-insensitive occurrence of `True` or `False`.
An output with no such verdict is incorrect. This repository will implement
that behavior as an explicitly paper-compatible evaluator rather than alter the
existing front-loaded factual evaluator or constrained-logit FLenQA policy.

Figure 1 in the paper uses non-adjacent/random placement and averages across
three tasks and two padding types, giving 600 source rows per nominal length.
The new runner evaluates deduplicated prompts, then analysis rejoins predictions
to source provenance. Filtering the expanded rows to `dispersion == "random"`
reproduces the paper's sample weighting without generating an identical prompt
more than once. At 250 tokens this intentionally restores duplicate source-row
weighting for the paper curve; the unique-prompt curve remains available to
show the deduplicated analysis.

## Architecture

### Shared evaluation

Extend `jlens_reasoning.evaluation_utils` with a gold-blind binary-verdict
extractor. It searches standalone `true` and `false` words without receiving a
gold label and returns the final match in response order. Substrings such as
`untrue` do not count.

Extend `jlens_reasoning.evaluation` with a typed paper-compatible binary result
and a public evaluation entry point. The result retains the raw `ModelOutput`,
the extracted verdict or `None`, the expected verdict, and correctness. A
complete or truncated model response with no verdict is a scored incorrect
response. A generation exception is not converted to model error data by the
runner; it aborts the current shard so resume can retry it.

Update `docs/llm-answer-evaluation.md` with a separate paper-compatibility
section. The current constrained-logit score remains the primary FLenQA policy;
the new evaluator exists only for behavioral comparison with the published
paper.

### Accuracy runner

Add a focused module under `jlens_reasoning.benchmarks.flenqa` that owns:

- immutable run configuration and result contracts;
- deterministic prompt ordering and sharding;
- one generation call per unique `prompt_id`;
- input and output token metadata capture;
- exact schema construction;
- atomic shard completion and validated resume;
- aggregation helpers for unique-prompt and source-row views.

The runner receives a generation callback or protocol so CPU-only tests can use
fakes. The Colab notebook owns Transformers model loading and defines the real
callback visibly. Generation uses greedy decoding (`do_sample=False`) with a
64-new-token safety cap. The callback returns generated token IDs, individually
decoded token pieces, decoded text, completion/truncation status, and finish
reason in `ModelOutput`.

Prompts are processed in canonical deduplication order. The default run asserts
12,000 normalized source rows and exactly 9,862 unique prompts before inference.
It tokenizes without truncation and rejects an input longer than the configured
4,096-token limit.

### Storage and resume

Write results beneath `context.runs_dir / "flenqa-accuracy"` as atomic Parquet
shards plus completion manifests. A shard is reusable only when its prompt
membership, configuration hash, schema, row count, and checksum validate.
Missing, temporary, or corrupt shards are rebuilt; valid completed shards are
never regenerated. A run-level configuration mismatch is rejected before any
shard is changed.

Each prompt result stores:

- `prompt_id`, canonical index, problem ID, task, and gold label;
- exact prompt text and nominal context size;
- Qwen input IDs and exact input-token count;
- ordered source provenance;
- generated token IDs, token pieces, raw generated text, generation status,
  and finish reason;
- extracted paper verdict and correctness.

The immutable run configuration records model name, tokenizer name, code
revision, maximum sequence length, maximum new tokens, shard size, and decoding
mode. Resume rejects a change to any scoring-relevant setting.

### Notebook

Add `notebooks/flenqa_accuracy.ipynb` as a thin Colab driver. It uses the
canonical Drive loader byte-for-byte, initializes Colab with a required GPU and
W&B disabled, loads the local Qwen model and FLenQA dataset, normalizes and
deduplicates all rows, defines the generation callback, invokes the runner, and
loads the completed result table.

The notebook keeps the workflow visible in ordered cells:

1. canonical Drive loader;
2. Colab initialization;
3. imports and immutable run settings;
4. dataset loading, validation, and deduplication counts;
5. model and tokenizer loading;
6. deterministic generation callback;
7. full accuracy run with resume;
8. result loading and integrity summary;
9. paper-compatible aggregation and plot;
10. all-unique-prompt aggregation and plot;
11. task and verdict-frequency diagnostics.

The committed notebook has no outputs or execution counts and contains no
credentials.

## Analysis outputs

The headline plot has one point for each nominal context size and uses the
paper's random-placement source-row weighting. A companion table reports
correct count, total count, accuracy, and no-verdict count for every point.

The second plot aggregates every unique prompt once. Its expected observation
counts are 300, 2,368, 2,394, 2,400, and 2,400 at nominal sizes 250, 500, 1000,
2000, and 3000. This plot prevents the shortest condition and incidental prompt
collisions from receiving duplicate weight.

Additional diagnostics show:

- per-task accuracy curves;
- counts and rates of `True`, `False`, and no-verdict responses by length;
- minimum, median, and maximum exact Qwen token count within each nominal
  bucket.

The notebook does not add confidence intervals that were absent from the paper.

## Failure handling

- Dataset schema, full-count, and dedup-count failures stop before model load or
  inference.
- Input truncation is prohibited; over-limit prompts fail explicitly.
- A model response that reaches the 64-token cap is stored as truncated and is
  scored from its final available `True`/`False` match, matching the behavioral
  extraction rule. If no verdict exists, it is incorrect.
- An exception during inference leaves the shard incomplete and stops the run.
  Resume rebuilds that shard so infrastructure failures do not become incorrect
  model answers.
- Invalid or mismatched completed shards fail validation and are rebuilt only
  within the known accuracy-run paths.
- Aggregation rejects missing prompt IDs, duplicate predictions, unknown source
  rows, mixed nominal lengths within one deduplicated prompt, and unexpected
  category values.

## Verification

CPU-only tests cover:

- final-occurrence extraction, case insensitivity, word boundaries, repeated
  verdicts, empty outputs, and outputs without a verdict;
- typed paper-compatible evaluation for complete and truncated responses;
- one generation per unique prompt and deterministic ordering;
- exact input-length capture and over-limit rejection;
- output-token and finish metadata preservation;
- atomic shard commit, interrupted-shard rebuild, valid resume, and
  configuration mismatch handling;
- nominal bucket attribution and source provenance round trips;
- paper-style random-placement source-row weighting;
- all-unique-prompt aggregation and expected full-run counts;
- notebook discovery, canonical loader reuse, visible full-run workflow, no
  saved outputs, and absence of credentials.

Final verification runs formatting, lint, the full CPU test suite, and the lock
check required by `CLAUDE.md`. The model-backed notebook itself remains a Colab
workflow and is not executed in CI.
