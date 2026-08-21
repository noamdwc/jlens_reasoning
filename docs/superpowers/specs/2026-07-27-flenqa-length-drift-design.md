# FLenQA Length-Drift Readout (Phase 1)

Date: 2026-07-27
Status: design

## Purpose

Measure how the model's internal concept workspace changes when the *same*
reasoning task is given longer inputs. FLenQA fixes the task and the key facts
and varies only padding, so readout differences across lengths are attributable
to the input. The original paper reports accuracy falling 0.92 → 0.68 from ~250
to 3000 tokens; this experiment collects the data a mechanism would have to
explain.

Phase 1 is **data production and scoring only**. Out of scope: the drift
analysis itself, the Phase 2 lens-length control, and Phase 3 concept steering
(whose target concept is an output of this phase).

## Core principle

**The prompt is the unit of analysis, not the dataset row.** A row is a recipe;
the prompt is the input. Consequently: one unique prompt is one observation;
padding presence and placement are derived from prompt content, while a padded
prompt's `books`/`same` subtype is carried only from unanimous source-row
provenance; row identity is never a key or seed.

The checked-in prompt templates are an immutable input to this work.
Preparation calls them as-is; neither the design nor implementation may rewrite
their text, spacing, task branching, or rule injection.

## Verified dataset facts

Verified against the published parquet (`alonj/FLenQA`, `eval`, 12,000 rows),
not the paper or dataset card — both of which are wrong in places.

**Identifiers don't identify rows.** `global_sample_id` has **300 unique values,
each appearing exactly 40×** — it names the problem, not the row. Using it as a
key would collide 40 ways.

**The grid is 2 padding types, not 3.** `padding_type` ∈ {`books`, `same`}, not
the paper's {duplicate, similar, different}. Grid: 3 tasks × 100 problems × 5
lengths (250/500/1000/2000/3000) × 2 padding × 4 dispersions
(first/middle/last/random) = 12,000. `label` is balanced True/False.

**`ctx_size=250` contains no padding at all.** For all 2,400 rows across all
three tasks, `mixin` is exactly the key paragraphs joined by a newline. Every
problem has **exactly one** unique prompt there — all 8 padding × dispersion
combinations are byte-identical. Those labels describe a distinction that does
not exist in the input.

**Grouping by prompt gives 9,862 observations** (300 / 2368 / 2394 / 2400 / 2400
by length). No group mixes `ctx_size`, problem, or `label`. 300 groups mix
`padding_type` — exactly the unpadded set. 338 mix `dispersion` — those 300 plus
38 incidental collisions at 500/1000, always `random` landing on a named
arrangement.

**RuleTaker is structurally different.** `facts` is null for all 4,000 rows; it
uses `statement` (bare sentences, whose paragraph expansions appear in `mixin`)
plus `rule` — and **the rule never appears in `mixin`**, so the template must
inject it or the task is unanswerable. PIR/MonoRel `facts` are already full
paragraphs, each occurring exactly once in `mixin`, with the key sentence
restated ~5× inside.

Consequently the key span for RuleTaker is **the paragraph in `mixin` that
contains the statement**, not the statement sentence. Locating the bare sentence
and calling the rest of its paragraph padding would count key material as
padding, inflate `n_padding_tokens`, distort every placement fraction, and let
`sampled_padding` anchors land inside the key facts. Key-text resolution is
therefore task-aware: PIR/MonoRel match the fact text directly, RuleTaker
expands each statement to its enclosing paragraph and requires exactly one
enclosing paragraph.

**Token counts track `ctx_size` closely** (measured with Qwen3-4B as proxy):
mean 277/502/1000/2003/3002, max observed 3077. So `max_seq_len = 4096` is safe.

## Identifiers

| Identifier | Grain | Use |
| --- | --- | --- |
| `problem_id` | reasoning problem; 300 values | grouping for paired comparisons |
| `prompt_id` | **hash of the final templated prompt** | storage, joins, seeds, sharding, resume |
| `source_row_id` | one dataset row | **provenance only** |

`prompt_id` is computed over the string the model actually sees, after
templating. Dedup is recomputed on that string and the resulting count
**asserted**, not assumed from the 9,862 measured on `(mixin, question)`.

## Condition variables at prompt grain

Each is stored as `*_declared` (provenance, may be a set) and `*_effective`
(used for analysis).

- **Length** — `n_tokens_actual`, measured. `ctx_size_declared` is an
  unambiguous grouping label.
- **`padding_type_effective`** ∈ {`none`, `books`, `same`}. Prompt content
  determines only whether padding is absent or present: an empty explicit
  padding-position set gives `none`, never an inference from `ctx_size`. When
  padding exists, all source rows for the deduplicated prompt must agree on one
  declared value, and that `books` or `same` value becomes effective. The
  subtype is provenance; it is not inferred from padding text.
- **Placement** — measured in **model token positions**, never characters.
  `frac_padding_before` / `_between` / `_after` are the shares of the explicit
  padding-position set (below) that fall before the first key span, between key
  spans, and after the last one. Characters are not a proxy: token density
  differs systematically between generated key paragraphs and book padding, and
  the readout itself is indexed by token position, so a character-based fraction
  would not describe the same object the anchors do. `dispersion_effective`
  derives from these: `not_applicable` (no padding), `first` (before≈0), `last`
  (after≈0), `middle` (before≈after, between≈0), else `scattered`;
  `unresolved` when any key span failed to resolve, since padding cannot then be
  separated from key material.

`random` is dropped as a category — it is a generation procedure, not a property
of an input. This resolves all 338 ambiguous groups on content and makes those
rows analyzable rather than a junk bucket.

## Statistical interpretation

The deduplicated design is **unbalanced**: 300 / 2368 / 2394 / 2400 / 2400
observations by length.

- The shortest condition has **300 observations, not 2,400**. Treating the rows
  as independent would inflate power 8× at the anchor point of every length
  comparison, since identical prompts give identical readouts.
- At 250 there is exactly one prompt per problem — a clean per-problem unpadded
  baseline.
- Comparisons are **paired within `problem_id`**. Cells are unequal, so
  unweighted averaging across conditions is invalid.
- Padding and placement are undefined at the baseline; analyses crossing them
  with length must treat it as its own level, not drop or impute it.

## Compute

One A100 40GB / 24 h. Mean context ≈1357 tokens; ~1.5–3 s per prompt at batch 1
→ **9,862 prompts in 4–8 hours**. Peak VRAM ≈11 GB (8 GB bf16 weights + 0.55
activations + 0.44 KV + ~2 workspace).

**Storage and host RAM bind, not compute.** `apply()` returns
`[n_positions, vocab]` fp32 **on CPU, per layer** — 1.8 GB per layer at 3000
positions, ~65 GB across layers for one prompt. Position subsampling is a hard
precondition. Batch 1 throughout; `apply()` has no batched path and writing one
is not justified.

## Architecture

```
experiments/flenqa_length_drift/
  constants.py  bridges.py  anchors.py  preparation.py  readout.py
  scoring.py  tables.py  gate.py  preflight.py  experiment.py
  flenqa_length_drift.ipynb   flenqa_smoke.ipynb
```

Shared additions: **`benchmarks/flenqa*.py`** (download, schema checks, prompt
construction, dedup, task-aware key-text resolution, token-based conditions),
**`experiments_utils/spans.py`**, and **`experiments_utils/storage.py`**
(schema-typed Parquet shards with atomic writes; `artifacts.py` is JSON-only and
fails at this scale).

`readout.py` takes prompt, anchors, and targets **from its caller**, not from a
FLenQA row — the seam for Phase 2's synthetic prompts.

**The run loop is library code, not notebook code.** `experiment.py` exposes
`run_prompt` (one prompt → typed column batches), `run_shard` (one shard →
committed Parquet + manifest), and `run_experiment` (plan shards, resume, run,
write the once-only tables). All three are tested on CPU against a fake lens
pass, so the logic that produces the dataset is covered by CI rather than
observed once in a notebook. The notebooks are thin drivers: load, gate,
preflight, call `run_experiment`, print. The smoke and full notebooks therefore
run the identical code path by construction.

## Readout

**Spans.** Find *all* occurrences, never first-match; expected count is 1.
`span_status` ∈ {ok, ambiguous, unresolved} and `span_match_count` are recorded
**per logical span** — each fact, the question, and each bridge-within-fact
target gets its own row in a `spans` table. A bridge row records the number of
occurrences in its resolved fact paragraph and the chosen last occurrence.
One collapsed per-prompt status would hide which span failed, which is exactly
the fact analysis needs to decide whether a prompt is usable for a given
comparison. Ambiguous spans are retained with null offsets and excluded at
analysis, never silently mismatched. Char→token conversion uses
`return_offsets_mapping`, never re-tokenizing the substring.

**Padding positions are an explicit set, not a complement of convenience.**
Parse `mixin` into newline-delimited paragraph payload spans and structural
separator spans. Padding payloads are the non-key paragraph payloads; blank
lines, newline delimiters, and whitespace between key paragraphs are structural,
not padding. A model-token position is padding only when its offset overlaps a
padding payload. Thus the 300 prompts containing only key paragraphs have
exactly zero padding positions even if their separators tokenize separately.
The injected rule, question, and template scaffolding are outside `mixin` and
therefore can never become padding.
`sampled_padding` anchors and the padding fill of the summary positions are
drawn **only** from that set. Sampling from "everything not an anchor" would
place the padding baseline inside key paragraphs or the template scaffolding,
which would blunt exactly the contrast the experiment exists to measure. When
any key span is unresolved the set is empty and no padding is sampled for that
prompt.

**Bridge.** The entity in both facts but not the question — task-specific
(possessive room phrase for PIR, middle person for MonoRel; null for RuleTaker).
A longest-common-substring rule was **tested and rejected**: it returns ragged
spans on PIR and boilerplate filler on MonoRel. The gate runs over the **200
applicable problems** — the 100 PIR and 100 MonoRel ones; RuleTaker's 100 have no
entity bridge and are excluded rather than counted as failures. The gate asserts
that applicable count as well as full resolution, so an extractor that silently
skips a task cannot pass by resolving a smaller set. It must pass offline before
any GPU time is spent. The bridge recurs ~5× per
paragraph; the anchor uses the last occurrence in the span, and the count is
recorded.

**Exact bridge measurements, not top-k truncated.** A bridge token may rank far
below 25 — at long contexts that is the expected finding, so a top-k-only design
would censor the primary signal. Exact rank and logit for every bridge candidate
are computed on GPU against the full vocabulary, at every anchor and layer, for
both lenses. Top-k is kept for exploration only.

Stored target ranks, including bridge and True/False ranks, use
`evaluation_utils.best_token_rank`, whose ties break toward the lower vocabulary
ID. Top-k output applies the same ordering without calling `best_token_rank` for
each token: select the top-k threshold once, include all strictly higher tokens
plus the lowest-ID boundary tokens needed to fill k, then stably order that
small selected set by `(-logit, token_id)`. Thus target ranks and top-k ranks
share one deterministic convention without multiplying full-vocabulary rank
scans.

**Anchors carry labels**, since a bare position is not comparable across
prompts: `fact_a_end`, `fact_b_end`, `bridge_fact_a`, `bridge_fact_b`,
`question_end`, `final_prompt`, `sampled_padding`. ≈10 per prompt; unlocatable
anchors recorded absent, prompt not dropped.

**Summary positions** — ≈40, scalars only: all anchors, key-span tokens (capped
12/span from the end), final 4 positions, remainder sampled from the padding
position set, seeded by `prompt_id`.

**Layers** come from `lens.source_layers`, not assumed. Both lenses are read at
the **same** positions, so the central comparison is matched.

## Storage

Parquet, keyed on `prompt_id`. Token strings are never repeated — `topk` and
`bridge` store `token_id`, with one `vocab` table mapping ids to strings.

| Table | Rows | Size |
| --- | --- | --- |
| `topk` — `(prompt_id, layer, anchor_label, lens_kind, rank)`, top-25 | ≈178 M | 2.5–3.5 GB |
| `bridge` — exact rank + logit per candidate | ≈28 M | ~0.5 GB |
| `summary` — provenance, entropy, max_logit, top1 | ≈28 M | ~1 GB |
| `prompts`, `source_rows`, `anchors`, `spans`, `scoring`, `vocab`, `run_meta` | small | trivial |

**≈4–5 GB.** Trim order if tight: top-k 25→10, then restrict `topk` to the
workspace layer band. Neither touches `bridge`.

Every table in that list is produced by the run — none is deferred to an
ad-hoc notebook cell. `prompts` holds `problem_id`, task, label,
`n_tokens_actual`, the **declared** conditions (`ctx_size_declared`,
`padding_type_declared`, `dispersion_declared`, each possibly a set), the
**effective** conditions and placement fractions, `n_padding_tokens`, the bridge
string, and the token-sequence signature. `anchors` records every
`(prompt_id, anchor_label, position)` actually read, so a label can be traced
back to a position without re-deriving it. `spans` carries the per-span
diagnostics. `source_rows` maps all 12,000 rows to their prompt, for
traceability only. `vocab` is written once from the tokenizer.

The required per-shard tables are `prompts`, `source_rows`, `anchors`, `spans`,
`topk`, `bridge`, `summary`, and `scoring`. `run_meta` and `vocab` are
run-global typed tables. Declared and effective conditions are explicit columns
of `prompts`, not overloaded into one condition column. Each span row has its
own `span_kind`, ordinal, `span_status`, `span_match_count`, and nullable
character/token bounds; fact, question, and bridge diagnostics are never
collapsed to a prompt-level flag.

**Rows are built columnar, not as dictionaries.** `topk` alone is ~178 M rows;
materializing that as Python dicts is tens of gigabytes of interpreter overhead
and would OOM the host long before Parquet size mattered. Each reduction emits
typed column arrays, one Arrow `RecordBatch` per prompt is appended to an open
`ParquetWriter` per table, and the writer is closed and renamed at shard commit.
Peak memory is one prompt's batch, not one shard's.

## Scoring — deterministic, no LLM grader

`label` is exactly `"True"`/`"False"` and balanced, so an LLM grader would add
cost and nondeterminism to a binary comparison. Two measures, both stored:
**constrained logit score** (compare final-position logits for the True/False
tokens — free, since `apply()` already returns `model_logits`) and a **short
generation with regex extraction** for the record and paper comparability.
Disagreement between them is diagnostic.

Both are built on the shared evaluation primitives — `answer_token_variants`
for the True/False surfaces, `best_token_rank` for ranks, `normalize_text` and
`match_reference` for the generated verdict — not on a private reimplementation.
The binary-verdict answer shape is a new case for
`docs/llm-answer-evaluation.md`, so this phase adds a policy version covering it
rather than extending v1 by inference.

## Guards

- **Truncation.** `apply()` defaults to `max_seq_len=512`, reaching
  `tokenizer(..., truncation=True)` at `jlens/hf.py:157`; with HuggingFace's
  default right-side truncation it silently keeps the first 512 tokens. Damage
  would be placement-dependent — padding-before means the **key facts are
  deleted** and the model answers from padding — producing a clean-looking
  placement × length interaction that is pure artifact. The prompt is therefore
  tokenized **once, with `truncation=False`**, and the resulting length asserted
  `≤ 4096` before anything else happens; `max_seq_len=4096` is then passed
  explicitly to every `apply()`. Tokenizing with `truncation=True` first would
  silently satisfy the assertion it is meant to test.
- **Position subsampling.** `positions` must be non-`None` and within budget.
  `positions=None` at 3000 tokens accumulates ~65 GB in **host** RAM against
  Colab's ~83 GB — it will not fail fast.
- **Schema.** Assert the verified value sets and counts; the released data
  already contradicts both the paper and the dataset card.
- **Dedup invariants.** Assert no group mixes `ctx_size`, problem, or `label`;
  that the 300 content-verified unpadded prompts each have exactly zero padding
  positions despite their structural separators; that
  `padding_type_effective = none` is never assigned where padding exists; and
  that a padded prompt's source rows agree on one declared padding subtype.
- **Tokenization identity.** The token ids each lens pass actually consumed are
  compared against the ids measured in preparation, and against each other, per
  prompt. The two-lens comparison is only meaningful if both read the same
  positions of the same sequence; a differing chat template, BOS handling, or
  truncation setting between the calls would misalign every anchor while
  producing perfectly plausible numbers. The signature is stored in `prompts`.
- **bf16** asserted.

## Pre-flight lens-validity gate

The lens is fitted on wikitext at n=1000, almost certainly short prompts.
Whether the Jacobian holds at 3000 tokens is an assumption **confounded with the
variable under study**. Before committing 4–8 A100 hours: take the validated
`spider` case from `jlens_readout_sanity`, pad it to **measured token counts** of
250/1000/3000 — padding is added until the tokenizer reports the target, not
until a word count is reached, since the lengths under study are token lengths
and a word-count proxy drifts by tens of percent — and confirm the J-Lens still
surfaces the target and still beats the logit lens at each length. If the
advantage collapses, Phase 2 comes first. This is a gate, not the Phase 2
control.

## Checkpointing and Drive I/O

Drive over FUSE runs ~10–30 MB/s and degrades on many small files. Dataset
copied to local `/content` once, never read from Drive in the loop. Shards of
~500 prompts written locally, **atomically** (temp path then rename), synced to
Drive on a background thread — a few minutes total across the run, capping crash
loss at one shard.

**Resume is shard-based, and a shard is all-or-nothing.**

- **Shard assignment is fixed before the run**, from the position of each prompt
  in the canonical ordered prompt list (dedup preserves first-occurrence order),
  not from the pending list. Assigning shards from what is left to do would
  renumber every shard after each crash, so shard 7 of the resumed run would
  contain different prompts than shard 7 of the first run — silently duplicating
  some prompts and dropping others across the union of shards.
- **A shard counts as complete only when a manifest exists**, and the manifest
  is written **last**, after every required per-shard table (`prompts`,
  `source_rows`, `spans`, `anchors`, `topk`, `bridge`, `summary`, `scoring`) has
  been committed for it.
  Resume derived from "which prompt_ids appear in `scoring`" would treat a
  prompt whose `scoring` row was written before a crash as done even though its
  `topk` rows were lost, producing a dataset that is internally inconsistent
  in a way nothing downstream can detect. The manifest records per-table row
  counts, which are re-verified on read.
- On restart, any uncommitted temporary files for an incomplete shard are
  discarded and that whole shard is rerun. Recomputation is bounded by one
  shard; partial state is never merged.
- **Config-aware resume:** `run_meta` stores a config hash (model, lens
  revision, template, budgets, top-k, bridge rule, dedup rule); a mismatch
  **aborts** rather than blending incompatible shards.
- `run_experiment` writes a run-completion manifest last, only after all shard
  manifests and both run-global tables are present and validated. Thus neither
  a shard nor the experiment can advertise completion while a required table is
  absent.

## Notebooks

Both notebooks are **thin drivers over `run_experiment`** — load, gate,
preflight, run, report. No loop body, reduction, or schema lives in a cell, so
"the smoke test runs the identical code path" is a structural fact rather than a
promise to keep two cell sequences in sync.

**`flenqa_smoke.ipynb`** (L4, first) — 2 problems × 40 rows deduplicating to ~33
prompts, stratified across every cell and both labels, including 3000-token
prompts so truncation and memory are exercised. Same guards, same schema. It
**measures real wall-clock per prompt** by `ctx_size` and prints an extrapolated
full-run estimate; the 1.5–3 s figure above is an estimate the smoke test
replaces with a measurement. ~10 minutes.

**`flenqa_length_drift.ipynb`** (A100) — 9,862 prompts, 4–8 hours.

## Testing

CPU-only, model mocked, per existing CI policy: schema verification and its
failure modes; prompt construction incl. RuleTaker `rule` injection; stable
content-determined `prompt_id`; dedup invariants and first-occurrence ordering;
task-aware key-text resolution, incl. RuleTaker paragraph expansion and its
ambiguity cases; padding-free detection from content; `none` rejected when
padding present; unanimous declared subtype required for padded prompts;
token-based placement fractions and `dispersion_effective` incl. `scattered`,
`unresolved`, and the 38 collisions; all 300 unpadded prompts have zero padding
positions; the padding position set excludes key paragraphs, structural
separators, question, rule, and template tokens; padding sampling draws only
from it;
`source_rows` many-to-one mapping; span location (exact/ambiguous/unresolved)
recorded per span; offset-mapping boundaries; bridge extraction gated over the
200 applicable problems; anchor labelling and budget; seeds stable from
`prompt_id`; deterministic scoring incl. disagreement, shared target-rank
tie-breaking, and deterministic top-k ties without per-token rank scans;
untruncated tokenization asserted against `MAX_SEQ_LEN`;
mismatched token ids between lens passes raise; `positions=None` rejected;
typed Arrow batch round-trip and atomic partial-shard safety; `run_prompt`,
`run_shard`, and `run_experiment` against a fake lens pass; a shard missing any
required table is not complete; resume reruns exactly the incomplete shards,
keeps shard assignment stable across restarts, and aborts on config mismatch.

## Assumptions still to verify

1. **Model identity** — `constants.py` names `Qwen/Qwen3.5-4B`; token counts and
   vocab (151,669) were measured with **Qwen3-4B as proxy**. Confirm tokenizer,
   layer count, `d_model`; all storage sizing scales with layer count.
2. **`lens.source_layers`** — may be a subset; 36 layers is an estimate.
3. **Dedup count** — the prompt templates are fixed and are not to be changed,
   but the 9,862 figure was measured on `(mixin, question)`; it is recomputed on
   the final templated text and **asserted**, not assumed.
4. **Bridge extractor** — implemented and covered by the 200-problem gate; the
   real cached dataset gate must still pass before GPU work.
5. **`padding_type=same`** — whether same-task padding can contain the bridge
   string is unmeasured. If it can, `sampled_padding` is not a clean baseline.
6. **Drive throughput and L4 availability** in the actual session.
7. **Model capability** — Qwen3.5-4B may floor at long lengths; the `scoring`
   table makes this visible immediately.

## Deferred by design

Task choice, bridge token definition, and which grid axes to feature are all
analysis decisions — the full grid is collected, and exact per-candidate ranks
plus top-k make them resolvable offline without re-running.
