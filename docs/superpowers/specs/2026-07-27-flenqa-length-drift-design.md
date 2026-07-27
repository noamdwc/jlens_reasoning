# FLenQA Length-Drift Readout (Phase 1)

Date: 2026-07-27
Status: design (revised after dataset verification)

## Purpose

Produce the J-Lens readout data needed to ask how a model's internal concept
workspace changes when the *same reasoning task* is given progressively longer
inputs.

FLenQA holds the task and the two key facts fixed and varies only padding
length, padding type, and key-fact placement. Any readout difference across
those variants is therefore attributable to the input, not to the question.
Averaged over the models in the original paper, accuracy falls from 0.92 to
0.68 between the ~250-token and 3000-token conditions. This experiment collects
the measurements that a mechanism for that drop would have to explain.

## Scope

Phase 1 is **data production and scoring only**. It runs the FLenQA eval split
through the J-Lens, scores the model's answers, and persists the readout in a
form that supports offline analysis.

### Non-goals

- **Phase 2 — J-Lens length sanity control.** Validating that the lens behaves
  correctly as prompt length grows. The design leaves a seam for it, and Phase 1
  includes only a minimal go/no-go pre-flight gate (below), not the full control.
- **Phase 3 — concept amplification/suppression.** The target concept and layer
  are outputs of Phase 1's analysis and cannot be chosen now.
- The drift analysis itself. Phase 1 produces tables; interpreting them must not
  require re-running the GPU job.

## Verified dataset facts

All figures below were verified directly against the published parquet
(`alonj/FLenQA`, `eval` split, 12,000 rows, 18 MB) rather than taken from the
paper or the dataset card. Several contradict the previous revision of this
document.

### Identifiers — `global_sample_id` is not unique

| Field | Unique values | Meaning |
| --- | --- | --- |
| `global_sample_id` | **300** — each appears **exactly 40 times** | the base reasoning problem |
| `sample_id` | **100** | index of the base problem *within its task* |
| `(dataset, sample_id)` | 300 | equivalent to `global_sample_id` |

`global_sample_id` identifies the base problem, **not the row**. Using it as a
storage grain, join key, shard key, seed, or resume key would collide 40 ways.
The previous revision did exactly that and was wrong.

**`(global_sample_id, ctx_size, padding_type, dispersion)` is unique: verified
12,000 of 12,000.**

This design therefore uses:

- **`base_id`** — the underlying reasoning problem. Equals `global_sample_id`.
- **`variant_id`** — the specific length/padding/dispersion condition. A stable
  identifier derived from the verified composite key. Used for **storage grain,
  joins, seeds, sharding, and resume**.
- **`prompt_id`** — content hash of `(mixin, assertion/question)`. See dedup.

### The grid is 2 padding types, not 3

Verified value counts:

| Field | Values | Counts |
| --- | --- | --- |
| `dataset` | `PIR`, `MonoRel`, `Simplified RuleTaker` | 4000 each |
| `ctx_size` | 250, 500, 1000, 2000, 3000 | 2400 each |
| `padding_type` | **`books`, `same` — 2 values only** | 6000 each |
| `dispersion` | `first`, `middle`, `last`, `random` | 3000 each |
| `label` | `True`, `False` (strings) | 6000 each — balanced |

The paper describes three padding types (`duplicate`, `similar`, `different`);
the **published dataset has two**, and uses different names. `same` is
same-task text, `books` is Books Corpus. The paper's `duplicate`/`similar`
distinction is not recoverable from the released fields.

Grid: 3 tasks × 100 base instances × 5 lengths × 2 padding × 4 dispersions
= **12,000**. Confirmed. So there are **40 variants per base problem**, not 60.

### `facts` is null for RuleTaker

| Task | `facts` | `statement` | `rule` |
| --- | --- | --- | --- |
| PIR | present | null | null |
| MonoRel | present | null | null |
| Simplified RuleTaker | **null (all 4000)** | present | present |

Key-span extraction must be task-specific. RuleTaker is structurally different
in the data as well as in the reasoning, reinforcing its role as a contrast
condition rather than a primary target.

### Fact spans occur exactly once

Across all 16,000 (row × fact) checks for PIR and MonoRel, each full fact
paragraph occurs **exactly once** in `mixin` — in every padding type and every
length. Paragraph-level matching is unambiguous in practice.

However, `facts` entries are ~600-character **paragraphs** in which the key
sentence is restated roughly five times. So the *bridge entity* occurs many
times **within** a single fact span, even though the span itself is unique.

### ctx=250 is 8× redundant — dedup required

| `ctx_size` | Rows | Unique `(mixin, question)` |
| --- | --- | --- |
| 250 | 2400 | **300** |
| 500 | 2400 | 2368 |
| 1000 | 2400 | 2394 |
| 2000 | 2400 | 2400 |
| 3000 | 2400 | 2400 |
| **Total** | **12,000** | **9,862** |

At 250 tokens there is essentially no padding, so all 8 padding × dispersion
combinations produce identical text. Running them separately would waste ~18% of
the GPU budget and, worse, **inflate apparent statistical power at the shortest
length by 8×** through pseudo-replication — a genuine analysis error, since a
deterministic forward pass returns identical readouts.

The run therefore executes **9,862 unique `prompt_id`s**, and a small
`variant_map` table maps all 12,000 `variant_id`s to their `prompt_id`. Analysis
joins through it and can weight or deduplicate explicitly.

### Token counts closely track `ctx_size`

Measured with the Qwen3-4B tokenizer (vocab 151,669) as a proxy:

| `ctx_size` | mean tokens | p95 | max |
| --- | --- | --- | --- |
| 250 | 276.6 | 344 | 368 |
| 500 | 501.6 | 519 | 539 |
| 1000 | 1000.4 | 1023 | 1034 |
| 2000 | 2002.5 | 2024 | 2066 |
| 3000 | 3002.0 | 3023 | 3077 |

The previous revision warned that GPT-4-token labels would diverge substantially
from Qwen counts. **That was overstated** — divergence is ~10% at the shortest
setting and under 1% elsewhere. `ctx_size` is a sound grouping label.
`n_tokens_actual` remains the correct quantitative variable, and both are
stored, but this is a minor correction rather than a confound.

Observed max is ~3077 tokens, so **`max_seq_len = 4096`** is a safe constant
with headroom for chat templating.

## Compute budget

One A100 40GB for 24 hours. Mean context across the grid ≈ 1357 tokens. At
batch 1 with HuggingFace overhead, ~1.5–3 s per prompt:

**9,862 unique prompts → 4–8 hours.** Comfortable, with room for a re-run.

Peak VRAM ≈ 11 GB (8 GB bf16 weights + 0.55 GB recorded activations + 0.44 GB
KV cache + ~2 GB workspace) against 40 GB available.

**Compute is not the binding constraint; storage and host RAM are.**
`apply()` returns `[n_positions, vocab]` fp32 **on CPU, per layer** — at 3000
positions that is 1.8 GB per layer and ~65 GB across all layers for one row.
Position subsampling is a hard precondition.

Because compute is not binding, **batch 1 throughout**. `JacobianLens.apply()`
takes a single `prompt: str` and has no batched path; writing one is not
justified.

## Architecture

```
experiments/flenqa_length_drift/
  constants.py    model/lens coordinates, grid, position budget, top-k, shard size
  bridges.py      task-specific bridge-entity extraction
  readout.py      per-prompt lens pass -> persisted records
  experiment.py   run loop, checkpoint/resume, Drive sync
  flenqa_length_drift.ipynb   Colab A100 driver (full run)
  flenqa_smoke.ipynb          Colab L4 driver (stratified subset)
```

Two shared additions under `src/jlens_reasoning/`:

- **`benchmarks/flenqa.py`** — download, schema verification, normalization,
  identifier construction, dedup, prompt construction, key-span location.
  In `src` because Phase 2 and Phase 3 both need it. This does not conflict with
  the environment-setup rule that the environment API stays benchmark-agnostic;
  that rule governs `environments`, which is untouched.
- **`experiments_utils/storage.py`** — Parquet shard writer/reader with atomic
  writes. `artifacts.py` provides only JSON `write_results`, which fails at this
  scale.

`readout.py` takes the prompt, anchor positions, and readout targets **from its
caller** rather than deriving them from a FLenQA row. This is the seam that lets
Phase 2 reuse the identical readout path with synthetic prompts.

## Data flow

Per unique `prompt_id`:

1. Construct the prompt from `mixin` and `assertion/question` using the task's
   prompt format, pinned from the authors' published analysis notebook.
2. Tokenize with `return_offsets_mapping`. Record `n_tokens_actual`. Assert it
   does not exceed the explicitly passed `max_seq_len`.
3. Locate key spans (task-specific) and map character spans to token spans via
   the offset mapping.
4. Extract bridge-entity candidates and locate their occurrences.
5. Select anchor positions and summary positions; record an **anchor label** for
   each.
6. Call `lens.apply()` twice — Jacobian and logit-lens — with explicit
   `positions` and explicit `max_seq_len`.
7. Reduce to `topk`, `bridge`, and `summary` records.
8. Score the answer deterministically (below).
9. Append to the current Parquet shard.

## Key-span and bridge extraction

### Span matching is find-all-and-verify, never first-match

For each key span the implementation finds **all** occurrences, not the first.
Given the verified result that each fact paragraph occurs exactly once, the
expected count is 1; the code asserts this and records
`span_match_count` plus a `span_status` of `ok` / `ambiguous` / `unresolved`.
Rows that are ambiguous or unresolved are **retained** with null spans and
excluded at analysis time, not silently dropped or silently mismatched.

Character spans are converted to token spans through the tokenizer's
`return_offsets_mapping`, never by re-tokenizing the substring — re-tokenization
does not respect the surrounding context and produces off-by-token errors at
span boundaries.

### Bridge extraction is task-specific and must be validated before the run

The bridge is the entity appearing in **both** key facts but **not** in the
question — `John's living room` in PIR, the middle person in MonoRel.

**A longest-common-substring rule was tested and rejected.** On PIR it returns
ragged spans (`"John's living room is"`, `"mention of John's grand ballroom i"`);
on MonoRel it returns boilerplate filler
(`"This is a fact that has been established and is well known among their circle
of friends"`) rather than any entity. It is recorded here as tested-and-rejected
so it is not retried.

The extractor is therefore **task-specific** — a possessive room phrase for PIR,
a person name for MonoRel — operating on candidate entity spans and selecting
those present in both facts and absent from the question. RuleTaker has no
entity bridge and is recorded with a null bridge.

**Required gate:** the extractor runs over all 300 base instances offline and
must resolve every PIR and MonoRel instance before any GPU time is spent. This
is a cheap CPU check and it protects the most fragile part of the design.

Because the bridge recurs ~5× within a fact paragraph, the anchor uses the
**last occurrence within the span** (the most recent binding, nearest to use),
and `bridge_occurrence_count` is recorded.

## Readout targets

### Exact bridge measurements — not top-k truncated

Top-25 is insufficient: a bridge token may rank far below 25, and at long
contexts that is precisely the outcome the experiment expects to find. A
top-k-only design would silently censor the primary signal.

The run therefore computes, **on GPU during the forward pass**, the **exact rank
and exact logit** of every reasonable bridge-token candidate at every anchor
position and every layer, for both lens kinds. Rank is computed against the full
vocabulary. Top-k is retained separately for exploration only.

### Anchor positions, with labels

Anchor **meaning** is persisted, not just position — a bare integer is not
joinable or interpretable across rows of differing length.

| `anchor_label` | Rationale |
| --- | --- |
| `fact_a_end` | was fact A encoded locally |
| `fact_b_end` | was fact B encoded locally |
| `bridge_fact_a` | first binding site |
| `bridge_fact_b` | second binding site; drift between sites is directly measurable |
| `question_end` | where the query is composed |
| `final_prompt` | where the next-token distribution, and thus the answer, forms |
| `sampled_padding` | baseline contrast; deterministic by seed, several per row |

≈10 anchors per prompt. Unlocatable anchors are recorded as absent; the prompt
is not dropped.

### Summary positions

≈40 positions receiving scalar summaries only, selected deterministically:
all anchors; every token of both key spans capped at 12 per span taken from the
span end backwards; the final 4 prompt positions; the remainder sampled from
padding on a seed derived from **`variant_id`**. The budget is a constant and is
asserted.

### Layer coverage

Layers come from `lens.source_layers`, not assumed. Sizing figures below cite 36
layers as an estimate for a Qwen3-4B-class model; the implementation derives the
count from the loaded lens and `run_meta` records it.

### Both lenses at the same positions

The Jacobian and logit lenses are read at the **same** anchor and summary
positions. Restricting the logit lens to the final position would make the
central comparison unmatched, and symmetric coverage is also simpler to
implement and describe than an asymmetric rule.

## Storage schema

Parquet throughout, keyed on `variant_id`/`prompt_id`. Token **strings are never
repeated** across rows: `topk` and `bridge` store `token_id` only, and a single
small `vocab` table maps `token_id` to its string. At ~178 M `topk` rows this
saves several gigabytes on its own.

| Table | Grain | Approx. rows | Approx. size |
| --- | --- | --- | --- |
| `topk` | `(prompt_id, layer, anchor_label, lens_kind, rank)`, top-25 | 9862 × 36 × 10 × 2 × 25 ≈ 178 M | ~2.5–3.5 GB |
| `bridge` | `(prompt_id, layer, anchor_label, lens_kind, candidate_token_id)` — exact rank + logit | ≈ 28 M | ~0.5 GB |
| `summary` | `(prompt_id, layer, position, lens_kind)` — provenance, entropy, max_logit, top1_token_id | ≈ 28 M | ~1 GB |
| `anchors` | `(prompt_id, anchor_label)` — position, span status, occurrence count | ≈ 100 K | trivial |
| `scoring` | one row per `prompt_id` | 9.9 K | trivial |
| `variant_map` | `variant_id` → `prompt_id`, plus grid fields | 12 K | trivial |
| `vocab` | `token_id` → token string | ~152 K | trivial |
| `run_meta` | model/lens revision, git commit, config hash, layer count, timings | — | trivial |

**Total ≈ 4–5 GB.** If this proves tight the trim order is: reduce top-k from 25
to 10, then restrict `topk` to the workspace layer band. Neither touches the
`bridge` table, which holds the primary signal.

`scoring` columns: `n_tokens_actual`, `ctx_size_reported`, `generated_text`,
`extracted_answer`, `logit_score_true`, `logit_score_false`, `predicted`,
`label`, `correct`, `bridge_resolved`, `span_status`.

## Deterministic scoring — no LLM grader

`label` is exactly `"True"` or `"False"`, balanced 6000/6000. An LLM grader would
add cost, nondeterminism, and a dependency for a binary comparison. Scoring is
therefore deterministic, with two independent measures:

1. **Constrained logit score** — compare the model's final-position logits for
   the `True` and `False` token variants. This is **free**: those logits are
   already returned by `apply()` as `model_logits`. Deterministic, immune to
   formatting, and never ambiguous.
2. **Short generation with regex extraction** — retained for the record and for
   comparability with the paper's generative protocol.

Both are stored. Disagreement between them is itself diagnostic. This follows
the existing LLM answer-evaluation policy's separation of raw generation,
extraction, normalization, and scoring; the policy's LLM-grader path is not
needed for a balanced binary label.

## Correctness guards

**Truncation.** `JacobianLens.apply()` defaults to `max_seq_len=512`, which flows
to `model.encode(prompt, max_length=max_seq_len)` and, in `jlens/hf.py:157`, to
`tokenizer(..., truncation=True, max_length=max_length)`. With HuggingFace's
default `truncation_side='right'` this keeps the first 512 tokens and discards
the rest with no warning and no exception.

The damage would be condition-dependent: with `dispersion=first` only padding is
lost; with `dispersion=last` the **key facts are deleted entirely** and the model
answers from padding alone; `middle` and `random` lose partially and
length-dependently. The result would be a strong, clean-looking
`dispersion × length` interaction that is pure tokenizer artifact.
`max_seq_len` is always passed explicitly (4096) and the untruncated token count
is asserted before the call.

**Position subsampling.** `positions` must be non-`None` and within budget.
`positions=None` on a 3000-token prompt accumulates ~65 GB of fp32 logits in
**host** RAM; a high-RAM Colab instance has ~83 GB, so it will not fail fast — it
will thrash and die unpredictably, or half-succeed.

**Schema verification on load.** Assert the exact verified value sets:
`dataset` ∈ {PIR, MonoRel, Simplified RuleTaker}; `ctx_size` ∈
{250, 500, 1000, 2000, 3000}; `padding_type` ∈ {books, same}; `dispersion` ∈
{first, middle, last, random}; `label` ∈ {True, False}; 12,000 rows; 300 unique
`base_id`s at 40 rows each; composite key unique. Fail loudly — the published
data already disagrees with both the paper and the dataset card, so silent
tolerance is not safe.

**Tokenization identity.** The two `apply()` calls must agree on `input_ids` and
baseline logits, as `jlens_readout_sanity` already checks.

**Model dtype.** bf16 asserted; fp32 doubles weights to 16 GB and buys nothing.

## Pre-flight lens-validity gate

The lens is fitted on wikitext at n=1000 prompts, almost certainly short. Whether
the Jacobian remains valid at 3000 tokens is an assumption **confounded with the
exact variable under study**. Full validation is Phase 2, but spending 4–8 A100
hours before any check would be reckless.

Phase 1 therefore runs a minimal go/no-go gate first: take the validated
`jlens_readout_sanity` `spider` case, pad it to ~250, ~1000, and ~3000 tokens
with Books Corpus text, and confirm the J-Lens still surfaces the target concept
and still beats the logit-lens baseline at each length. If the J-Lens advantage
collapses at length, the full run is not worth starting and Phase 2 must come
first. This is a gate, not the Phase 2 control, and it is deliberately small.

## Checkpointing and Drive I/O

Drive over FUSE runs ~10–30 MB/s and degrades badly with many small files.

- The dataset is copied to local `/content` once at startup and never read from
  Drive inside the loop.
- Parquet shards of ~500 prompts are written to **local** disk — few large files.
- **Atomic writes:** each shard is written to a temporary path and renamed on
  completion, so an interrupted write can never be mistaken for a complete shard.
- Each completed shard is synced to Drive on a background thread so GPU work does
  not block on I/O. At ~20 shards this is a few minutes spread across a 4–8 hour
  run, so syncing every shard is nearly free and caps crash loss at one shard.
- **Config-aware resume:** `run_meta` stores a hash of the run configuration
  (model, lens revision, prompt format, anchor/summary budgets, top-k, bridge
  rule). On resume the hash is recomputed and compared; a mismatch **aborts**
  rather than silently blending incompatible shards. On match, completed
  `prompt_id`s are skipped.

## Notebooks

**`flenqa_smoke.ipynb`** — L4, run first. 2 base instances × 40 variants,
deduplicating to ~66 unique prompts, stratified to cover every `padding_type` ×
`dispersion` × `ctx_size` cell and both labels, including 3000-token rows so
truncation and memory are genuinely exercised. It runs the **identical** code
path, asserts the same guards, and writes the same schema.

It must also **measure and report real end-to-end wall-clock per prompt**, broken
down by `ctx_size`, and print an extrapolated full-run estimate. The 1.5–3 s
figure in this document is an estimate; the smoke test replaces it with a
measurement, and the full run should not start if the projection exceeds the
budget. L4 has 24 GB against ~11 GB peak and is ~3–5× slower than A100, so
expect ~10 minutes.

**`flenqa_length_drift.ipynb`** — A100 40GB, 9,862 unique prompts, 4–8 hours.

## Testing

CPU-only and model-free, consistent with existing CI policy:

- schema verification, including each failure mode
- identifier construction; `variant_id` uniqueness; `variant_map` round-trip
- dedup correctness — 12,000 variants collapse to 9,862 prompts
- prompt construction against pinned expected strings
- span location: exact-one, ambiguous, and unresolved paths
- offset-mapping char→token conversion at span boundaries
- bridge extraction over all 300 base instances (the required gate)
- anchor labeling and budget enforcement; unlocatable-anchor path
- deterministic scoring, including logit/generation disagreement
- truncation guard raises; `positions=None` rejected
- Parquet round-trip for all tables; atomic write leaves no partial shard
- resume skips exactly the completed prompts; config-hash mismatch aborts

Model-backed behavior is mocked.

## Assumptions still requiring verification

Stated explicitly so they are checked at implementation time rather than
assumed:

1. **Model identity.** `constants.py` names `Qwen/Qwen3.5-4B`. Token counts and
   vocab size (151,669) here were measured with **Qwen3-4B as a proxy**. Confirm
   the actual tokenizer, layer count, and `d_model` on the real model; all
   storage sizing depends on the layer count.
2. **`lens.source_layers`** — the fitted lens may cover a subset of layers. The
   36-layer figure is an estimate.
3. **Prompt format.** The authors' exact per-task prompts have not yet been
   extracted from their analysis notebook. Divergence weakens comparison to the
   published accuracy curve.
4. **Bridge extractor.** The task-specific rule is specified but not yet built or
   validated; the 300-instance gate is the check.
5. **`padding_type=same` collisions.** Whether same-task padding can contain the
   bridge string itself has not been measured. If it can, `sampled_padding`
   anchors may not be a clean baseline, and padding occurrences of the bridge
   should be recorded.
6. **Drive throughput and L4 availability** in the actual Colab session.
7. **Model capability.** Qwen3.5-4B may floor at long lengths or fail to
   reproduce the paper's accuracy curve. The `scoring` table makes this visible
   immediately rather than after analysis.

## Decisions deferred by design

- Task choice for the drift analysis — all three are collected.
- Bridge token definition — exact per-candidate ranks plus top-k make this
  resolvable offline without re-running.
- Which grid axes to feature — the full grid is collected.
