# FLenQA Benchmark Runner Simplification Design

## Goal

Turn `flenqa_length_drift` from an experiment package into a small FLenQA
benchmark runner that:

1. loads and validates the FLenQA dataset;
2. finds and validates meaningful token positions;
3. runs both Jacobian Lens and Logit Lens at only those positions; and
4. saves compact, resumable outputs.

The two tracked FLenQA driver notebooks move from
`experiments/flenqa_length_drift/` to `notebooks/`. The old
`experiments.flenqa_length_drift` Python package is removed without
compatibility shims.

## Package Layout

The implementation lives under:

```text
src/jlens_reasoning/benchmarks/flenqa/
├── __init__.py
├── dataset.py
├── positions.py
├── runner.py
└── storage.py
```

### `dataset.py`

Owns the FLenQA data model and prompt construction:

- source-row schema and count validation;
- row normalization;
- prompt text and stable prompt ID construction;
- deduplication with source-row provenance.

### `positions.py`

Owns all logic required to choose trustworthy JLens positions:

- one untruncated tokenization of each final prompt;
- key-fact, bridge, question, and paragraph span resolution;
- task-specific bridge extraction;
- padding-token identification;
- deterministic semantic-anchor selection; and
- dataset-level and per-prompt validation gates.

The public result is a prepared prompt with exact input IDs and labeled anchor
positions. Bridge extraction, span resolution, and gates are implementation
details of position selection rather than separate analysis subsystems.

### `runner.py`

Owns lens execution:

- adapters around `JacobianLens.apply`;
- one Jacobian Lens pass and one Logit Lens pass using identical anchors;
- strict agreement checks for prepared input IDs and model logits;
- layer/position tensor-shape validation;
- deterministic top-k reduction at anchor positions only; and
- shard orchestration.

`top_k` remains configurable and defaults to 25.

### `storage.py`

Owns output persistence:

- typed Arrow schemas;
- immutable Parquet shard writing;
- completion manifests written last;
- configuration and prompt-membership validation on resume; and
- run-level completion validation.

## Position Selection

The runner requests lens output only for labeled meaningful positions:

- the end of each key fact;
- each bridge mention in the key facts when the task has a bridge;
- the end of the question;
- the final prompt token; and
- up to four deterministically sampled padding positions.

Positions must be unique, sorted for lens execution, within the tokenized
prompt, and traceable to one or more semantic labels. The saved position table
retains the labels even when multiple labels resolve to the same token.

The former 48-position summary set is removed. In particular, the runner no
longer adds fact-tail tokens, the final four-token window, or padding positions
merely to fill a fixed summary budget. It also no longer computes entropy,
maximum-logit, or top-1 summaries for those positions.

## Validation Gates

Validation must prevent a run from quietly producing incomplete or
semantically misaligned results.

Before lens execution:

- source schema and full-dataset count invariants are checked when a full
  dataset is supplied;
- prompt IDs and canonical indices must be unique and deterministic;
- every applicable PIR and MonoRel problem must have one consistent bridge;
- the bridge must not appear in the question;
- required fact, bridge, and question spans must resolve unambiguously;
- anchors must have the expected labels and valid token indices; and
- prompts must not exceed the configured lens sequence limit.

After each pair of lens passes:

- both passes must use the exact prepared input IDs;
- both passes must return identical model logits;
- requested layers and positions must have compatible tensor shapes; and
- top-k results must contain the expected number of deterministic rows.

During persistence:

- every prompt in a shard must appear in all required tables;
- shard files must match their declared Arrow schemas;
- a completion manifest is written only after all shard files close
  successfully; and
- resume is rejected when the configuration or prompt membership differs.

Smoke or subset runs may supply explicit expected dataset and bridge counts.
They do not silently disable validation.

## Saved Data

Each shard contains three typed tables.

### `prompts`

One row per deduplicated prompt:

- prompt ID and canonical index;
- problem ID, task, and label;
- final prompt text and exact input IDs;
- token count;
- source-row IDs;
- declared context-size, padding-type, and dispersion provenance; and
- extracted bridge when applicable.

List-valued provenance stays on the prompt row instead of using a separate
source-row table.

### `positions`

One row per prompt, token position, and semantic label:

- prompt ID;
- token position; and
- label such as `fact_a_end`, `bridge_fact_a`, `question_end`,
  `final_prompt`, or `sampled_padding`.

### `topk`

One row per retained value:

- prompt ID;
- lens kind (`jacobian` or `logit`);
- layer;
- token position;
- top-k rank;
- token ID; and
- logit.

Top-k output is produced only for positions present in `positions`.

Run metadata and shard manifests record model, lens, tokenizer, code revision,
configuration hash, table checksums, and prompt membership.

## Removed Scope

The refactor removes:

- generated True/False answers and scoring;
- answer-token ranks and correctness fields;
- synthetic exact-length preflight prompts;
- the rule requiring Jacobian Lens to beat Logit Lens in preflight;
- padding-condition classification and fractional placement analysis;
- bridge-target-specific ranks and logits;
- entropy, maximum-logit, and top-1 summary tables;
- the 48-position summary budget; and
- the `experiments.flenqa_length_drift` import path.

Both lens modes remain because their saved outputs are useful for later
experiments.

## Notebook and Documentation Migration

Move the tracked notebooks to:

```text
notebooks/flenqa_smoke.ipynb
notebooks/flenqa_length_drift.ipynb
```

They remain thin drivers that load the dataset and model/lens assets, construct
the runners and configuration, invoke the benchmark runner, and report the
saved run location. Preflight, scoring, and direct data-reduction code do not
remain in notebook cells.

The untracked
`experiments/flenqa_length_drift/flenqa_smoke_output.ipynb` is user-owned and
must not be edited or moved.

README and notebook-location tests are updated to describe FLenQA as a
benchmark runner rather than an experiment. Historical design and plan
documents remain historical records unless a live link or instruction would
otherwise point users to a removed path.

## Testing

Tests move under `tests/benchmarks/flenqa/` and are consolidated around the
four responsibilities:

- dataset validation, normalization, prompt construction, and deduplication;
- bridge/span/padding/anchor selection and failure gates;
- paired lens execution, agreement checks, and deterministic top-k output; and
- typed shard writing, interruption behavior, resume checks, and final
  manifests.

Notebook tests assert the new locations, thin-driver imports, absence of saved
outputs and credentials, and removal of scoring/preflight calls.

The migration is complete when no runtime code or tests import
`experiments.flenqa_length_drift`, both tracked notebooks use the new runner,
the focused and full test suites pass, and Ruff reports no violations.
