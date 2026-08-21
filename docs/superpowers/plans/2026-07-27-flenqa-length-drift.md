# FLenQA Length-Drift Readout (Phase 1) Implementation Plan

> **For agentic workers:** Use `superpowers:executing-plans` to implement this
> plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a complete, deduplicated, token-aligned FLenQA readout whose
Parquet shards are safe to resume and sufficient for offline drift analysis.

**Architecture:** Preserve the approved prompt templates and deduplicate their
final text in first-occurrence order. Prepare each unique prompt once with the
model tokenizer, derive spans, conditions, padding positions, and anchors from
those token IDs, then stream typed Arrow batches through tested `run_prompt`,
`run_shard`, and `run_experiment` functions. A shard is visible only after all
of its tables are committed and its manifest is written last.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace Transformers/Datasets,
PyArrow/Parquet, pinned `jlens`, pytest, ruff.

---

## Non-negotiable contracts

- The prompt is the unit of analysis. `prompt_id = SHA-256(final_prompt_text)`;
  deduplication preserves the first source-row occurrence. Source rows are
  provenance only.
- Do not edit the approved prompt strings, whitespace, task branching, or
  RuleTaker rule injection. Add a golden-text regression test before moving
  their existing definitions into `flenqa_prompts.py`.
- Tokenize the complete prompt once with `truncation=False` and offsets enabled.
  Assert `len(input_ids) <= 4096`. Both lens calls receive `max_seq_len=4096`
  and identical selected positions; their consumed IDs must equal the prepared
  IDs and each other.
- PIR/MonoRel key spans are their complete fact paragraphs. For RuleTaker,
  resolve each bare statement to the unique enclosing paragraph in `mixin`;
  the paragraph is key material, never padding.
- Parse `mixin` into newline-delimited paragraph payloads and structural
  separators. `padding_positions` is the explicit sorted set of model tokens
  overlapping non-key paragraph payloads; blank lines, newline delimiters, and
  whitespace between key paragraphs are excluded. If any key paragraph is
  unresolved, the set is empty. Placement fractions, `sampled_padding`, and
  summary padding fill use only this set.
- Every logical fact, question, and bridge span has its own `span_status` and
  `span_match_count`. Do not collapse them to a prompt-level status.
- Every stored target rank, including bridge and True/False ranks, calls
  `jlens_reasoning.evaluation_utils.best_token_rank`. Top-k uses the same
  lower-token-ID tie rule through one threshold selection and a stable
  `(-logit, token_id)` ordering of the selected IDs, not one
  `best_token_rank` call per output token.
- Never build table rows as `list[dict]`. Producers return schema-checked
  `pyarrow.RecordBatch` objects and `ParquetWriter` appends them per prompt.
- Shard assignment is `canonical_prompt_index // SHARD_SIZE`, computed from the
  original deduplicated order before resume filtering.

## Artifact contract

Required per-shard tables:

| Table | Required fields beyond `prompt_id` |
| --- | --- |
| `prompts` | canonical index, problem/task/label, final-text hash, token hash and count, declared condition lists, effective conditions, three token-placement fractions, padding count, bridge |
| `source_rows` | source-row ID and every declared source condition |
| `spans` | kind, ordinal, fact ordinal, surface, status, match count, nullable char/token bounds |
| `anchors` | label, token position |
| `topk` | layer, position/anchor label, lens kind, deterministic rank, token ID, logit |
| `bridge` | layer, position/anchor label, lens kind, candidate surface/token ID, exact rank, logit |
| `summary` | layer, position/provenance, lens kind, entropy, max logit, top-1 token ID |
| `scoring` | True/False logits and ranks, constrained verdict/correctness, generated text, extracted verdict/correctness, agreement |

Run-global typed tables are `vocab(token_id, token_text)` and
`run_meta(config_hash, model/lens/tokenizer/template coordinates, budgets,
schema version, code revision)`. A shard manifest records its immutable prompt
IDs plus row count and checksum for every required table. The run-completion
manifest is written last after all shard manifests, `vocab`, and `run_meta`
validate.

## Task 1: Dataset model, immutable prompts, and ordered deduplication

**Files**

- Create: `src/jlens_reasoning/benchmarks/flenqa.py`
- Create: `src/jlens_reasoning/benchmarks/flenqa_prompts.py`
- Create: `experiments/flenqa_length_drift/constants.py`
- Test: `tests/benchmarks/test_flenqa.py`
- Test: `tests/benchmarks/test_flenqa_prompts.py`

**Interfaces:** `FlenqaRow`, `FlenqaPrompt(canonical_index, source_row_ids, ...)`,
`verify_schema`, `normalize_rows`, `build_prompt_text`, `compute_prompt_id`,
`deduplicate`.

- [ ] Write failing tests for the published categorical/count invariants,
  sequential `source_row_id`, and RuleTaker using `statement` plus injected
  `rule`.
- [ ] Add golden expected strings for PIR/MonoRel and RuleTaker, copied
  byte-for-byte from the already-approved templates. This test must fail before
  the template-owning module is created; never “improve” the strings.
- [ ] Test that deduplication hashes final prompt text, collapses identical
  prompts, rejects mixed problem/label/context size, and preserves
  first-occurrence order in `canonical_index`.
- [ ] Run:
  `uv run pytest tests/benchmarks/test_flenqa.py tests/benchmarks/test_flenqa_prompts.py -v`
  and confirm the intended missing-module/assertion failures.
- [ ] Implement the smallest typed row/prompt pipeline that passes. It may use
  dataclasses for the 12,000-row preparation set; table serialization remains
  columnar.
- [ ] Re-run the two test files and commit.

## Task 2: Task-aware spans and one untruncated tokenization

**Files**

- Create: `src/jlens_reasoning/experiments_utils/spans.py`
- Create: `src/jlens_reasoning/benchmarks/flenqa_preparation.py`
- Test: `tests/experiments_utils/test_spans.py`
- Test: `tests/benchmarks/test_flenqa_preparation.py`

**Interfaces:** `SpanDiagnostic`, `PreparedPrompt`, `find_all_spans`,
`char_span_to_token_span`, `resolve_key_paragraphs`,
`prepare_prompt(prompt, tokenizer, max_seq_len=4096)`.

- [ ] Write failing span tests for exact, missing, and multiple matches and for
  offset-mapping boundaries without substring re-tokenization.
- [ ] Write a RuleTaker fixture where a short statement sits inside a longer
  paragraph. Assert the diagnostic/token span covers the full paragraph and
  the paragraph remainder is not padding. Add missing- and multiple-enclosing-
  paragraph cases. Paragraphs are newline-delimited non-empty blocks in
  `mixin`; preserve their original character offsets.
- [ ] With a recording fake tokenizer, assert `prepare_prompt` calls it with
  `truncation=False`, retains the returned IDs and offsets, accepts exactly
  4096 IDs, and rejects 4097. A test using a truncating fake must prove the
  length guard cannot be fooled by a truncated result.
- [ ] Assert separate diagnostic rows and match counts for every fact and
  question. For each applicable bridge-in-fact target, record its own match
  count and the last resolved occurrence; zero occurrences is unresolved.
- [ ] Run the two test files and observe the expected failures.
- [ ] Implement one tokenization path and task-aware resolution; no helper may
  tokenize a substring or invoke the prompt template again.
- [ ] Re-run the tests and commit.

## Task 3: Padding set, token placement, bridges, and anchors

**Files**

- Create: `src/jlens_reasoning/benchmarks/flenqa_conditions.py`
- Create: `experiments/flenqa_length_drift/bridges.py`
- Create: `experiments/flenqa_length_drift/anchors.py`
- Create: `experiments/flenqa_length_drift/gate.py`
- Test: `tests/benchmarks/test_flenqa_conditions.py`
- Test: `tests/experiments/flenqa_length_drift/test_bridges.py`
- Test: `tests/experiments/flenqa_length_drift/test_anchors.py`
- Test: `tests/experiments/flenqa_length_drift/test_gate.py`

**Interfaces:** `build_padding_positions`, `derive_conditions`,
`extract_bridge`, `bridge_gate`, `select_anchors`,
`select_summary_positions`.

- [ ] Write failing tests showing `padding_positions` contains only tokens
  overlapping non-key paragraph payloads. Explicitly test that key paragraphs,
  structural separators/blank lines, question, rule, BOS/EOS, and template
  tokens are never candidates.
- [ ] Add the dataset invariant test: all 300 content-verified unpadded prompts
  have exactly zero padding positions, including fixtures where separators
  receive their own model token IDs.
- [ ] Test placement fractions by token count, including a fixture whose
  character fractions imply a different classification. Fractions must sum to
  one when padding exists; unresolved keys yield effective dispersion
  `unresolved`, no padding anchors, and no padding summary fill.
- [ ] Test that content determines only `none` versus padded. For padding,
  require all source rows to agree on one declared `books` or `same` value and
  copy that value to `padding_type_effective`; reject disagreement rather than
  inferring the subtype from text. Retain declared conditions on provenance.
- [ ] Test both seeded sampling functions with adversarial non-padding gaps and
  assert every `sampled_padding`/padding-fill position belongs to the explicit
  set.
- [ ] Test the task-specific bridge extractor and run `bridge_gate` over exactly
  200 distinct applicable problems: 100 PIR plus 100 MonoRel. Assert RuleTaker
  is excluded and that resolving only one applicable task fails the count gate.
- [ ] Run the four test files, implement minimally, re-run, and commit.

## Task 4: Deterministic readout, scoring, and token-count preflight

**Files**

- Create: `experiments/flenqa_length_drift/readout.py`
- Create: `experiments/flenqa_length_drift/scoring.py`
- Create: `experiments/flenqa_length_drift/preflight.py`
- Modify: `docs/llm-answer-evaluation.md`
- Test: `tests/experiments/flenqa_length_drift/test_readout.py`
- Test: `tests/experiments/flenqa_length_drift/test_scoring.py`
- Test: `tests/experiments/flenqa_length_drift/test_preflight.py`

**Interfaces:** `reduce_readout`, `score_binary_answer`,
`pad_to_token_count`, `run_preflight`.

- [ ] Write tied-logit tests proving bridge and True/False target ranks call
  `best_token_rank`. For top-k, assert lower-token-ID ordering at ties and spy
  that `best_token_rank` is not called once per selected token; implement one
  top-k threshold selection plus stable `(-logit, token_id)` ordering.
- [ ] Test exact bridge rank below top-k, labelled anchors, source layers taken
  from the lens output, and identical positions for both lens kinds.
- [ ] Test constrained True/False scoring and generated verdict extraction
  through existing shared evaluation primitives; update the policy document
  with the binary-verdict contract before implementing scoring.
- [ ] Test preflight construction at exact measured token counts 250, 1000, and
  3000 with a fake tokenizer. The API accepts `target_tokens`, never
  `target_words`, and asserts the achieved untruncated count.
- [ ] Run the three test files, implement minimally, re-run, and commit.

## Task 5: Typed Arrow schemas and transactional shard writers

**Files**

- Create: `experiments/flenqa_length_drift/tables.py`
- Create: `src/jlens_reasoning/experiments_utils/storage.py`
- Modify: `pyproject.toml`
- Test: `tests/experiments/flenqa_length_drift/test_tables.py`
- Test: `tests/experiments_utils/test_storage.py`

**Interfaces:** `TABLE_SCHEMAS`, `GLOBAL_SCHEMAS`, batch builders,
`ShardWriter.append`, `ShardWriter.commit`, `validate_shard_manifest`,
`is_shard_complete`.

- [ ] Write failing schema tests covering every table and every declared/
  effective condition, provenance, anchor, vocabulary, and span-diagnostic
  field listed in the artifact contract.
- [ ] Assert each batch builder returns a `pyarrow.RecordBatch` with the exact
  schema. Add a regression test that rejects `list[dict]` input.
- [ ] With a tiny batch-size threshold, stream several batches through one open
  `ParquetWriter` per table and prove peak buffered rows never exceeds one
  prompt batch.
- [ ] Simulate failure after each table commit. Until all eight required table
  files validate and the manifest is atomically renamed last,
  `is_shard_complete` must be false. Validate manifest prompt IDs, row counts,
  schemas, and checksums on read.
- [ ] Run the two test files, add PyArrow, implement, re-run, and commit.

## Task 6: Tested `run_prompt`

**Files**

- Create: `experiments/flenqa_length_drift/experiment.py`
- Test: `tests/experiments/flenqa_length_drift/test_experiment.py`

**Interface:**

```python
def run_prompt(
    prepared: PreparedPrompt,
    *,
    jacobian_runner: LensRunner,
    logit_runner: LensRunner,
    tokenizer: Tokenizer,
) -> tuple[tuple[str, pyarrow.RecordBatch], ...]: ...
```

- [ ] First write a fake `LensRunner` test proving both calls receive the same
  final prompt, explicit positions, and `max_seq_len=4096`.
- [ ] Assert each runner returns the token IDs it actually consumed.
  `run_prompt` must raise if either differs from `prepared.input_ids` or from
  the other runner, before emitting any batch.
- [ ] Assert the successful result includes schema-valid batches for all eight
  per-shard tables, including empty typed `bridge`/`topk` batches when a prompt
  legitimately has no rows.
- [ ] Assert sampled positions come only from the prepared explicit padding set
  and every emitted anchor is traceable through `anchors`.
- [ ] Run the focused tests, implement minimal orchestration over Tasks 2–5,
  re-run, and commit.

## Task 7: Stable, all-or-nothing `run_shard`

**Files**

- Modify: `experiments/flenqa_length_drift/experiment.py`
- Test: `tests/experiments/flenqa_length_drift/test_experiment.py`

**Interfaces:** `PromptShard(shard_id, prompt_indices)`, `plan_shards`,
`run_shard`.

```python
def run_shard(
    shard: PromptShard,
    prepared_prompts: Sequence[PreparedPrompt],
    *,
    output_dir: Path,
    runners: LensRunners,
    tokenizer: Tokenizer,
) -> ShardManifest: ...
```

- [ ] Test `plan_shards` against canonical indices before creating any pending
  list. Complete shard 0, resume, and assert every later shard retains exactly
  its original ID and prompt membership.
- [ ] Inject a crash after each required table. Assert the whole shard remains
  incomplete, its temporary state is ignored/cleaned on restart, and all
  prompts in that shard rerun.
- [ ] Assert `run_shard` keeps only open writers plus one prompt’s batches,
  closes and atomically renames every table, validates them, then writes the
  completion manifest last.
- [ ] Run the focused tests, implement, re-run, and commit.

## Task 8: Complete `run_experiment` and thin notebooks

**Files**

- Modify: `experiments/flenqa_length_drift/experiment.py`
- Create: `experiments/flenqa_length_drift/flenqa_smoke.ipynb`
- Create: `experiments/flenqa_length_drift/flenqa_length_drift.ipynb`
- Modify: `tests/test_notebooks.py`
- Modify: `README.md`
- Test: `tests/experiments/flenqa_length_drift/test_experiment.py`

```python
def run_experiment(
    rows: Sequence[FlenqaRow],
    *,
    output_dir: Path,
    tokenizer: Tokenizer,
    runners: LensRunners,
    config: RunConfig,
) -> RunManifest: ...
```

- [ ] Write an end-to-end fake-run test: final-text dedup, canonical shard
  plan, bridge gate, untruncated preparation, global `vocab`/`run_meta`, all
  required shard tables, safe resume, and final run manifest.
- [ ] Assert config mismatch aborts before a shard opens. Assert missing or
  corrupt global tables/shard manifests prevent run completion.
- [ ] Assert `vocab` maps each tokenizer ID once and `source_rows` preserves all
  12,000-to-prompt links without affecting seeds, joins, or shard assignment.
- [ ] Implement `run_experiment`: validate/dedup once, plan immutable shards,
  write/validate global typed tables, skip only validated complete shards, call
  `run_shard`, validate the whole run, and atomically save the final manifest
  last.
- [ ] Keep both notebooks to bootstrap/load, gate, preflight,
  `run_experiment`, and report only. No loop, schema, span logic, readout
  reduction, or shard bookkeeping may live in a cell. Update the notebook
  discovery tripwire.
- [ ] Run the fake end-to-end tests and notebook tests, then commit.

## Task 9: Final verification

- [x] Real published `alonj/FLenQA` eval gate (2026-07-28):
  `rows=12000, applicable=200, resolved=200`. Do not start GPU work unless the
  same gate passes in the execution environment.
- [ ] Run:

```bash
uv run ruff format .
uv run ruff check .
uv run pytest
uv lock --check
```

- [ ] Inspect every notebook for empty outputs/execution counts and confirm the
  loader cell matches repository policy.
- [ ] Inspect one synthetic completed run: all schemas match, each source row
  maps to one final prompt, every prompt belongs to exactly one canonical shard,
  all sampled padding positions belong to its prepared padding set, each lens
  pair has identical token hashes/positions, every shard has all eight tables,
  and the run-completion manifest is the last artifact.
- [ ] Commit only after all verification output is clean.
