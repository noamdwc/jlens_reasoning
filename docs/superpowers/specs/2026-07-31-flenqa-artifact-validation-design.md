# FLenQA Artifact Validation Design

## Goal

Add a fail-fast acceptance gate for the artifact produced by
`notebooks/flenqa_full_run.ipynb`. The gate establishes that the saved run is
complete, internally consistent, and scientifically sane enough to serve as
input to later research analysis.

The validator does not perform exploratory analysis, generate plots, interpret
scientific results, or rerun either lens. It does not change the README.

## Architecture

Add a reusable validation module at:

```text
src/jlens_reasoning/benchmarks/flenqa/validation.py
```

and a thin CPU-only Colab driver at:

```text
notebooks/flenqa_validate.ipynb
```

The module owns all acceptance policy and returns a typed validation report.
The notebook only initializes the environment, loads the pinned tokenizer from
local artifacts, invokes the validator for `runs/flenqa-full-run`, and displays
the report. It never loads the language model or Jacobian Lens and does not
require a GPU.

## Inputs and Output

The validator accepts the run directory and tokenizer. It reads
`run-meta.json`, `run-manifest.json`, every shard manifest, and the `prompts`,
`positions`, and `topk` Parquet tables named by those manifests.

Validation produces an immutable typed report with named checks and relevant
counts. Successful validation returns the report. Failed validation raises a
dedicated validation error containing the failed check and diagnostic context;
the notebook prints the compact report or error so the artifact has an
unambiguous pass/fail outcome.

The validator is read-only. It never repairs, deletes, or rewrites run files.

## Acceptance Checks

### Artifact integrity

- Both run-level JSON files exist and contain the required values.
- The run metadata and completion manifest use the same configuration hash.
- Recomputing the canonical hash of the saved configuration reproduces that
  configuration hash.
- Shard IDs are unique and ordered.
- Every shard passes the existing schema, row-count, path, and checksum
  validation against the declared FLenQA table schemas.
- Concatenated shard prompt membership exactly matches the run manifest.
- The manifest and table directories contain exactly the committed shard files;
  no expected file is missing and no unreferenced committed shard is present.

### Prompt and position consistency

- Each manifest prompt appears exactly once in the prompts tables.
- Prompt IDs and canonical indices are unique, with canonical indices strictly
  increasing in manifest order.
- Prompt task, Boolean label, bridge value, text, provenance, and input IDs
  satisfy their declared schema and basic nonempty/range invariants.
- Tokenizing each saved prompt text with the pinned tokenizer reproduces its
  saved input IDs exactly.
- Every positions row references a known prompt and an in-range token index.
- Each prompt contains exactly the task-appropriate required label set from the
  benchmark position policy: all applicable fact ends, the question end, the
  final prompt token, and all applicable bridge anchors. Padding anchors remain
  optional because some prompts have no eligible padding content. Unknown
  labels are rejected.
- Multiple semantic labels may share a token index, but duplicate
  `(prompt_id, position, label)` rows are rejected.
- The final-prompt anchor equals the final saved input-ID index. Decoding every
  selected position succeeds, providing bounded diagnostics on failure without
  treating a particular display string as a scientific result.

### Paired-lens consistency

- Every top-k row references a known prompt and one of its unique selected
  positions.
- Lens kind is exactly `jacobian` or `logit`; layer, rank, token ID, and logit
  values are in range and every logit is finite.
- Within each `(prompt_id, lens_kind, layer, position)` group, ranks are
  contiguous from one, token IDs are unique, and logits are monotonically
  non-increasing by rank.
- Each prompt has the same nonempty layer set for both lens kinds, and that set
  matches the run manifest's returned layers.
- For every prompt, layer, and selected position, Jacobian and Logit Lens groups
  have identical rank cardinality. The cardinality matches the configured
  `top_k` unless the tokenizer vocabulary is smaller, in which case it matches
  the vocabulary size.
- Every `max_abs_logit_diff` is finite and non-negative, and their maximum
  matches the run manifest summary. The validator confirms that the run records
  finite, non-negative comparison tolerances, but it cannot replay the original
  `allclose` decision because the complete model-logit tensors are intentionally
  absent from the artifact.
- Decoding saved top-k token IDs succeeds. Decoded values are diagnostic only;
  the gate imposes no semantic hypothesis about which token should rank first.

## Notebook Workflow

The notebook uses the repository's byte-identical Drive loader cell, then:

1. initializes Colab with W&B disabled and `require_cuda=False`;
2. loads the tokenizer from the pinned local model path;
3. sets the input directory to `context.runs_dir / "flenqa-full-run"`;
4. runs the reusable validator; and
5. displays a compact acceptance report.

The notebook is committed with stable cell IDs, no outputs, and no execution
counts. It contains no analysis plots or discretionary sampling controls.

## Testing

Unit tests under `tests/benchmarks/flenqa/` construct small valid committed
artifacts and mutate one invariant at a time. Coverage includes successful
validation plus failures for corrupt manifests, prompt membership, tokenizer
mismatch, missing or invalid anchors, unknown top-k positions, incomplete lens
pairs, noncontiguous ranks, duplicate token IDs, non-finite or misordered
logits, layer disagreement, and run-summary disagreement.

Notebook tests add `flenqa_validate.ipynb` to the exact tracked notebook set and
verify the shared loader cell, empty outputs and execution counts, Colab
environment initialization, CPU-only behavior, absence of model/lens loading,
and delegation to the reusable validator.

## Scope Boundaries

- No README or other user documentation changes.
- No model or lens execution.
- No source-dataset reconstruction.
- No plots, aggregate scientific findings, hypothesis tests, or exploratory
  summaries.
- No artifact repair or mutation.
- No claims that the artifact supports a research conclusion; passing means
  only that it is acceptable as input to the next research stage.
