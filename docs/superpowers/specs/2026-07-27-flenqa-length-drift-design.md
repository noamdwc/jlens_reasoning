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
condition labels are derived from prompt content, never copied from row
metadata; row identity is provenance only, never a key or seed.

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

## Condition variables, derived from the prompt

Each is stored as `*_declared` (provenance, may be a set) and `*_effective`
(used for analysis).

- **Length** — `n_tokens_actual`, measured. `ctx_size_declared` is an
  unambiguous grouping label.
- **`padding_type_effective`** ∈ {`none`, `books`, `same`}. `none` only when
  confirmed from content (prompt is the key paragraphs and nothing else) —
  never inferred from `ctx_size`.
- **Placement** — measured as `frac_padding_before` / `_between` / `_after` from
  actual key-span token positions. `dispersion_effective` derives from these:
  `not_applicable` (no padding), `first` (before≈0), `last` (after≈0), `middle`
  (before≈after, between≈0), else `scattered`.

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
  constants.py  bridges.py  readout.py  experiment.py
  flenqa_length_drift.ipynb   flenqa_smoke.ipynb
```

Shared additions: **`benchmarks/flenqa.py`** (download, schema checks, prompt
construction, dedup, content-derived conditions, span location) and
**`experiments_utils/storage.py`** (Parquet shards with atomic writes;
`artifacts.py` is JSON-only and fails at this scale).

`readout.py` takes prompt, anchors, and targets **from its caller**, not from a
FLenQA row — the seam for Phase 2's synthetic prompts.

## Readout

**Spans.** Find *all* occurrences, never first-match; expected count is 1;
record `span_match_count` and `span_status` ∈ {ok, ambiguous, unresolved}.
Ambiguous rows are retained with null spans and excluded at analysis, never
silently mismatched. Char→token conversion uses `return_offsets_mapping`, never
re-tokenizing the substring.

**Bridge.** The entity in both facts but not the question — task-specific
(possessive room phrase for PIR, middle person for MonoRel; null for RuleTaker).
A longest-common-substring rule was **tested and rejected**: it returns ragged
spans on PIR and boilerplate filler on MonoRel. The extractor must resolve all
300 problems offline before any GPU time is spent. The bridge recurs ~5× per
paragraph; the anchor uses the last occurrence in the span, and the count is
recorded.

**Exact bridge measurements, not top-k truncated.** A bridge token may rank far
below 25 — at long contexts that is the expected finding, so a top-k-only design
would censor the primary signal. Exact rank and logit for every bridge candidate
are computed on GPU against the full vocabulary, at every anchor and layer, for
both lenses. Top-k is kept for exploration only.

**Anchors carry labels**, since a bare position is not comparable across
prompts: `fact_a_end`, `fact_b_end`, `bridge_fact_a`, `bridge_fact_b`,
`question_end`, `final_prompt`, `sampled_padding`. ≈10 per prompt; unlocatable
anchors recorded absent, prompt not dropped.

**Summary positions** — ≈40, scalars only: all anchors, key-span tokens (capped
12/span from the end), final 4 positions, remainder sampled from padding seeded
by `prompt_id`.

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
| `anchors`, `scoring`, `prompts`, `source_rows`, `vocab`, `run_meta` | small | trivial |

**≈4–5 GB.** Trim order if tight: top-k 25→10, then restrict `topk` to the
workspace layer band. Neither touches `bridge`.

`prompts` holds `problem_id`, task, `n_tokens_actual`, declared and effective
conditions, placement fractions. `source_rows` maps all 12,000 rows to their
prompt, for traceability only.

## Scoring — deterministic, no LLM grader

`label` is exactly `"True"`/`"False"` and balanced, so an LLM grader would add
cost and nondeterminism to a binary comparison. Two measures, both stored:
**constrained logit score** (compare final-position logits for the True/False
tokens — free, since `apply()` already returns `model_logits`) and a **short
generation with regex extraction** for the record and paper comparability.
Disagreement between them is diagnostic.

## Guards

- **Truncation.** `apply()` defaults to `max_seq_len=512`, reaching
  `tokenizer(..., truncation=True)` at `jlens/hf.py:157`; with HuggingFace's
  default right-side truncation it silently keeps the first 512 tokens. Damage
  would be placement-dependent — padding-before means the **key facts are
  deleted** and the model answers from padding — producing a clean-looking
  placement × length interaction that is pure artifact. Pass 4096 explicitly and
  assert the untruncated count.
- **Position subsampling.** `positions` must be non-`None` and within budget.
  `positions=None` at 3000 tokens accumulates ~65 GB in **host** RAM against
  Colab's ~83 GB — it will not fail fast.
- **Schema.** Assert the verified value sets and counts; the released data
  already contradicts both the paper and the dataset card.
- **Dedup invariants.** Assert no group mixes `ctx_size`, problem, or `label`;
  that the unpadded set is exactly the 300 content-verified prompts; and that
  `padding_type_effective = none` is never assigned where padding exists.
- **Tokenization identity** across the two `apply()` calls; **bf16** asserted.

## Pre-flight lens-validity gate

The lens is fitted on wikitext at n=1000, almost certainly short prompts.
Whether the Jacobian holds at 3000 tokens is an assumption **confounded with the
variable under study**. Before committing 4–8 A100 hours: take the validated
`spider` case from `jlens_readout_sanity`, pad to ~250/~1000/~3000 tokens, and
confirm the J-Lens still surfaces the target and still beats the logit lens at
each length. If the advantage collapses, Phase 2 comes first. This is a gate,
not the Phase 2 control.

## Checkpointing and Drive I/O

Drive over FUSE runs ~10–30 MB/s and degrades on many small files. Dataset
copied to local `/content` once, never read from Drive in the loop. Shards of
~500 prompts written locally, **atomically** (temp path then rename), synced to
Drive on a background thread — a few minutes total across the run, capping crash
loss at one shard. **Config-aware resume:** `run_meta` stores a config hash
(model, lens revision, template, budgets, top-k, bridge rule, dedup rule); a
mismatch **aborts** rather than blending incompatible shards.

## Notebooks

**`flenqa_smoke.ipynb`** (L4, first) — 2 problems × 40 rows deduplicating to ~33
prompts, stratified across every cell and both labels, including 3000-token
prompts so truncation and memory are exercised. Identical code path, same
guards, same schema. It **measures real wall-clock per prompt** by `ctx_size` and
prints an extrapolated full-run estimate; the 1.5–3 s figure above is an
estimate the smoke test replaces with a measurement. ~10 minutes.

**`flenqa_length_drift.ipynb`** (A100) — 9,862 prompts, 4–8 hours.

## Testing

CPU-only, model mocked, per existing CI policy: schema verification and its
failure modes; prompt construction incl. RuleTaker `rule` injection; stable
content-determined `prompt_id`; dedup invariants; padding-free detection from
content; `none` rejected when padding present; placement fractions and
`dispersion_effective` incl. `scattered` and the 38 collisions; `source_rows`
many-to-one mapping; span location (exact/ambiguous/unresolved); offset-mapping
boundaries; bridge extraction over all 300 problems; anchor labelling and budget;
seeds stable from `prompt_id`; deterministic scoring incl. disagreement;
truncation guard raises; `positions=None` rejected; Parquet round-trip and
atomic-write partial-shard safety; resume skips exactly completed prompts and
aborts on config mismatch.

## Assumptions still to verify

1. **Model identity** — `constants.py` names `Qwen/Qwen3.5-4B`; token counts and
   vocab (151,669) were measured with **Qwen3-4B as proxy**. Confirm tokenizer,
   layer count, `d_model`; all storage sizing scales with layer count.
2. **`lens.source_layers`** — may be a subset; 36 layers is an estimate.
3. **Prompt templates** — not yet extracted from the authors' notebook; this also
   affects the final dedup count, which is asserted rather than assumed.
4. **Bridge extractor** — specified but not built; the 300-problem gate is the
   check.
5. **`padding_type=same`** — whether same-task padding can contain the bridge
   string is unmeasured. If it can, `sampled_padding` is not a clean baseline.
6. **Drive throughput and L4 availability** in the actual session.
7. **Model capability** — Qwen3.5-4B may floor at long lengths; the `scoring`
   table makes this visible immediately.

## Deferred by design

Task choice, bridge token definition, and which grid axes to feature are all
analysis decisions — the full grid is collected, and exact per-candidate ranks
plus top-k make them resolvable offline without re-running.
