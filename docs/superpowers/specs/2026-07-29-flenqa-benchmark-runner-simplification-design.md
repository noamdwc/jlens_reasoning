# FLenQA Benchmark Runner Simplification Design

> **Status: superseded.** The current runner writes directly to final Parquet
> shards, requires empty output table directories, and does not support
> manifests or resume. This document is retained as historical design context.

## Goal

Replace `experiments.flenqa_length_drift` with a small benchmark runner that
loads FLenQA, finds validated meaningful positions, runs both Jacobian Lens and
Logit Lens at those positions, and saves compact resumable outputs.

There are no compatibility shims for the old experiment import path.

## Package

```text
src/jlens_reasoning/benchmarks/flenqa/
├── __init__.py
├── dataset.py
├── positions.py
├── runner.py
└── storage.py
```

- `dataset.py`: validate and normalize source rows, construct prompts, and
  deduplicate them.
- `positions.py`: tokenize each prompt once, extract bridges, resolve fact,
  bridge, question, and padding-content spans, validate the results, and select
  labeled anchors.
- `runner.py`: run both lens modes at the selected positions, validate their
  outputs, reduce them to deterministic top-k rows, and orchestrate shards.
- `storage.py`: define the three Arrow tables and atomically write, validate,
  and resume shards.

## Meaningful Positions

Lens output is requested only at:

- the end of each key fact;
- each bridge mention in the key facts when the task has a bridge;
- the end of the question;
- the final prompt token; and
- up to four positions sampled from tokens overlapping known padding-content
  spans.

Every paragraph containing a declared key fact is a meaningful fact span.
Repeated matching paragraphs retain the same logical fact label, and all are
excluded from padding-content sampling. A fact is unresolved only when it has
no matching paragraph.

Padding sampling is deterministic. The random generator seed is derived from
`prompt_id` and a fixed sampling seed recorded in the run configuration.
Sampling is without replacement. An eligible token must overlap a declared
padding-content span. Special tokens and tokens that overlap only whitespace
are excluded. Punctuation tokens remain eligible when they are part of actual
padding content.

Multiple labels may resolve to the same token position. Every
`(prompt_id, position, label)` row is saved in `positions`, while lens execution
receives the sorted unique positions only once. Expected `topk` row counts are
therefore calculated from unique positions, not position-label rows.

The former 48-position summary set and its entropy/top-1 summaries are removed.

## Validation

The runner fails rather than silently saving partial or misaligned data:

- full-dataset schema and count invariants are checked;
- prompt IDs and canonical indices must be unique and deterministic;
- each applicable PIR and MonoRel problem must have one consistent bridge that
  does not appear in its question;
- required fact, bridge, and question spans, plus declared padding-content
  spans when present, must resolve consistently;
- anchor positions must be in range and carry the expected labels;
- both lens passes must return the prepared input IDs;
- both lens passes must return the same layer keys;
- layer tensors must match the unique requested positions; and
- saved table membership, schemas, and row counts must match the shard plan.

The two passes may differ slightly because of numerical execution. Their model
logits are compared with:

```python
torch.allclose(jacobian_logits, logit_logits, rtol=1e-5, atol=1e-6)
```

These tolerances are part of the recorded run configuration. The maximum
absolute difference is defined exactly as:

```python
max_abs_logit_diff = (
    jacobian_logits - logit_logits
).abs().max().item()
```

It is stored in the `prompts` table and summarized in the run manifest.

The saved `layer` value is the exact integer layer key returned by
`JacobianLens.apply`. It identifies the transformer readout layer and is not
renumbered by the runner. The requested and returned layer keys are recorded in
run metadata.

Smoke or subset runs supply explicit expected dataset and bridge counts; they
do not disable validation.

## Saved Data

Each shard contains three typed Parquet tables:

- `prompts`: prompt identity, task and label, exact text and input IDs,
  extracted bridge, maximum model-logit difference, and complete source
  provenance.
- `positions`: every semantic label attached to each selected position.
- `topk`: prompt ID, lens kind, returned layer index, unique position, rank,
  token ID, and logit.

Source provenance is ordered by `source_row_id` and stored directly in Arrow as
a typed field, not JSON or a serialized string:

```text
provenance: list<struct<
    source_row_id: int32,
    ctx_size: int32,
    padding_type: string,
    dispersion: string
>>
```

`top_k` is configurable and defaults to 25. Both lens modes use identical
unique positions.

## Atomic Shards and Resume

Each shard is written to shard-specific temporary files. After all three files
are closed and validated, they are atomically moved to their final paths. The
completion manifest is also written through a temporary file and atomically
replaced last.

A shard without a valid completion manifest is incomplete. On resume, the
runner removes only that shard's known temporary and partial files and rebuilds
the shard from the beginning. A completed shard is reused only when its
configuration hash, prompt membership, schemas, row counts, and checksums
match.

## Removed Scope

Remove generated-answer scoring, synthetic preflight prompts, padding-content
placement classification, bridge-target-specific ranks, entropy summaries, the
48-position summary budget, and the old experiment package.

Both Jacobian Lens and Logit Lens outputs remain.

## Notebooks and Tests

Move and rename the tracked drivers to:

```text
notebooks/flenqa_smoke.ipynb
notebooks/flenqa_full_run.ipynb
```

They remain thin drivers over the package. The untracked
`experiments/flenqa_length_drift/flenqa_smoke_output.ipynb` is user-owned and
must not be edited or moved.

Tests move under `tests/benchmarks/flenqa/` and cover dataset preparation,
bridge/span/anchor gates, unique-position paired-lens execution, tolerant logit
comparison, deterministic top-k output, provenance records, atomic shard
recovery, and resume validation. Notebook and README references are updated to
the new benchmark-runner names and locations.
