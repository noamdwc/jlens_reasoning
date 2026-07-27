# FLenQA Length-Drift Readout (Phase 1)

Date: 2026-07-27
Status: design

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

Phase 1 is **data production and grading only**. It runs the full FLenQA eval
split through the J-Lens, grades the model's answers, and persists the readout
in a form that supports offline analysis.

### Non-goals

Explicitly out of scope, to be handled as separate specs:

- **Phase 2 — J-Lens length sanity control.** Validating that the lens itself
  behaves correctly as prompt length grows. This is the next piece of work and
  the design below leaves a seam for it, but it is not specified here.
- **Phase 3 — concept amplification/suppression.** Steering interventions to
  help the model on long prompts. The target concept and layer for any such
  intervention are outputs of Phase 1's analysis and cannot be chosen now.
- The drift analysis itself. Phase 1 produces tables; interpreting them is
  downstream work that must not require re-running the GPU job.

## Background

### FLenQA construction

FLenQA (Levy, Jacoby, Goldberg — *Same Task, More Tokens*) comprises three
reasoning tasks, each with two key sentences that must be combined:

- **MonoRel** — two comparisons on a transitive scale ("Samantha is younger than
  Julie"; "Julie is younger than Julian"). The middle entity bridges them.
- **People In Rooms (PIR)** — one fact places a person in a named location, the
  other gives that location a property. The location bridges them.
- **Simplified RuleTaker** — one rule plus two facts. The chain runs through a
  predicate, not a named entity.

MonoRel and PIR share a structural feature: a **bridge entity that appears in
both key facts but never in the question**. This is the same "unspoken
intermediate" structure the `jlens_readout_sanity` experiment already validated
with its `spider` case. RuleTaker lacks it, which makes it a useful contrast
condition rather than a defect.

Each base instance is expanded along three axes:

| Axis | Values |
| --- | --- |
| `ctx_size` | 250, 500, 1000, 2000, 3000 tokens (GPT-4 tokenizer, ±70 tolerance) |
| `padding_type` | `duplicate` (key paragraphs repeated), `similar` (paragraphs from other instances of the same task), `different` (Books Corpus prose) |
| `dispersion` | `first`, `middle`, `last`, `random` — where the key paragraphs sit |

100 base instances per task, 300 total, expanded to **12,000 rows** in a single
`eval` split on HuggingFace (`alonj/FLenQA`).

`padding_type=similar` is notable: the padding is drawn from other instances of
the same task, so it contains other people and other rooms. It is a built-in
semantic-distractor condition in the same vocabulary space as the bridge.

### Relevant fields

```
global_sample_id, sample_id, label, dataset,
facts, rule, statement, assertion/question,
mixin, padding_type, dispersion, ctx_size
```

Three matter structurally:

- **`facts` is stored separately from `mixin`.** `mixin` is the full padded
  context; `facts` is the key sentences alone. Key-fact token positions can
  therefore be recovered by string matching rather than heuristics.
- **`sample_id` is stable across variants**, so length comparisons are paired
  within base instance.
- **`ctx_size` is measured in GPT-4 tokens**, not Qwen tokens. See guards.

### Compute budget

One A100 40GB for 24 hours. Estimated per-row cost at batch 1, mean context
≈1350 tokens: prefill ~11 TFLOP (~0.09 s), lens projection ~2.2 TFLOP
(~0.02 s), generation ~8 tokens (~0.05 s). Pure compute ~0.16 s; with
HuggingFace and Python overhead at batch 1, realistically **1.5–3 s per row**,
so **5–10 hours for all 12,000 rows**.

Peak VRAM for the subsampled design: ~8 GB weights (bf16) + 0.55 GB recorded
activations + 0.44 GB KV cache + ~2 GB workspace ≈ **11 GB**, comfortable in
40 GB.

**Compute is not the binding constraint. Storage and host RAM are.** In
particular, `apply()` returns `[n_positions, vocab]` in fp32 **on CPU, per
layer**; at 3000 positions that is 1.8 GB per layer and ~65 GB across 36 layers
for a single row. Position subsampling is a hard precondition, not an
optimization.

Because compute is not binding, **batch 1 throughout**. `JacobianLens.apply()`
accepts a single `prompt: str` and has no batched path; writing one is
explicitly not justified here.

## Architecture

A new experiment package following the `jlens_readout_sanity` layout:

```
experiments/flenqa_length_drift/
  constants.py    model/lens coordinates, grid, position budget, top-k, shard size
  readout.py      per-row lens pass -> persisted records
  experiment.py   run loop, checkpoint/resume, Drive sync
  flenqa_length_drift.ipynb   Colab A100 driver (full split)
  flenqa_smoke.ipynb          Colab L4 driver (stratified subset)
```

Two shared additions under `src/jlens_reasoning/`:

- **`benchmarks/flenqa.py`** — download, schema verification, normalization,
  prompt construction, fact-span location. Placed in `src` rather than in the
  experiment package because the Phase 2 control and any Phase 3 intervention
  both need it. This does not conflict with the environment-setup spec's rule
  that the environment API stays benchmark-agnostic: that rule governs
  `environments`, which is untouched.
- **`experiments_utils/storage.py`** — Parquet shard writer and reader.
  `artifacts.py` provides only `write_results`, which is JSON; JSON is
  precisely what fails at this scale.

`readout.py` takes the prompt, the anchor positions, and the readout targets
**from its caller** rather than deriving them from a FLenQA row. This is the
seam that lets Phase 2 reuse the identical readout path with synthetic prompts
whose correct readout is known a priori.

## Data flow

Per row:

1. Construct the prompt from `mixin` and `assertion/question` using the task's
   prompt format, pinned from the authors' published analysis notebook rather
   than improvised.
2. Tokenize. Record the true Qwen token count. Assert it does not exceed the
   explicitly passed `max_seq_len`.
3. Locate key-fact token spans by matching the `facts` strings within the
   prompt. Derive per-position provenance labels.
4. Resolve bridge-entity candidate token ids (see below).
5. Select anchor positions and summary positions.
6. Call `lens.apply()` twice — Jacobian and logit-lens baseline — with explicit
   `positions` and explicit `max_seq_len`.
7. Reduce to `topk` and `summary` records.
8. Generate the answer and grade it against `label` via the existing evaluation
   module, following the LLM answer-evaluation policy.
9. Append to the current Parquet shard.

Grading happens in the same pass because the generation is nearly free
alongside the prefill already being computed, and re-running to grade later
costs another GPU day.

## Readout targets

### Bridge entity — recorded, not committed to

The bridge is the entity appearing in **both** key facts but **not** in the
question. Bridge entities in FLenQA are multi-token (`John's living room`,
`Julie Baker`), whereas the validated sanity readout used single tokens
(`spider`, `ant`). Rather than settle this now and risk a GPU day on the wrong
definition:

- Derive bridge candidates per row from the `facts` and question strings.
- Resolve single-token variants via the existing `concept_token_variants`.
- Record bridge rank and logit where resolvable, and **null where not**.
- Persist full top-k at every anchor position regardless.

Any alternative bridge definition — head noun, full span, first-name token,
multi-token aggregate — can then be evaluated offline against the stored
top-k without re-running. The PIR-versus-MonoRel and single-versus-multi-token
questions become analysis decisions rather than collection decisions.

### Anchor positions (full top-k)

Positions known a priori to matter, per row:

| Anchor | Rationale |
| --- | --- |
| Final prompt position | where the next-token distribution, and thus the answer, forms |
| End of key fact A | was fact A encoded locally |
| End of key fact B | was fact B encoded locally |
| Bridge occurrence within fact A | first binding site |
| Bridge occurrence within fact B | second binding site; drift between the two sites is directly measurable |
| End of question/assertion | where the query is composed |
| ~4 sampled padding positions | baseline contrast — deterministic by seed |

Approximately 10 anchors per row. Anchors that cannot be located (unresolvable
bridge, unmatched fact span) are recorded as absent; the row is not dropped.

### Summary positions

A wider set of approximately 40 positions receives scalar summaries only,
selected deterministically as:

- all anchor positions;
- every token of both key-fact spans, capped at 12 per fact and taken from the
  span end backwards when a fact exceeds the cap;
- the final 4 positions of the prompt;
- the remainder filled with padding positions sampled on a fixed per-row seed
  derived from `global_sample_id`, so the selection is reproducible without
  storing it.

The exact budget is a constant, and the selection is asserted to stay within it.

### Layer coverage

Readout layers are taken from `lens.source_layers`, not assumed. The fitted lens
may cover a subset of the model's layers, and all sizing figures below that cite
36 layers are estimates based on a Qwen3-4B-class model; the implementation must
derive the count from the loaded lens and the stored `run_meta` must record it.

## Storage schema

Parquet throughout. Four tables.

**`topk`** — full top-25 readout.
Grain: `(global_sample_id, layer, position, lens_kind, rank)`.
Columns: `token_id`, `token`, `logit`.
J-Lens at all ~10 anchor positions across all 36 layers; logit-lens at the
final position only.
Approx. 12,000 × 36 × 10 × 25 ≈ **108 M rows ≈ 2.2 GB**.

**`summary`** — scalars.
Grain: `(global_sample_id, layer, position, lens_kind)`.
Columns: `provenance` (`fact_a` / `fact_b` / `question` / `padding` / `other`),
`entropy`, `max_logit`, `top1_token_id`, `bridge_rank` (nullable),
`bridge_logit` (nullable).
J-Lens at all summary positions; logit-lens at the final position only.
Approx. **17 M rows ≈ 600 MB**.

**`generation`** — one row per variant.
Columns: `global_sample_id`, `sample_id`, `dataset`, `padding_type`,
`dispersion`, `ctx_size_reported`, `n_tokens_actual`, `generated_text`,
`extracted_answer`, `label`, `correct`, `bridge_resolved`, `anchors_located`.

**`run_meta`** — model name and revision, lens repo/revision/file, git commit,
seeds, per-shard timings, library versions.

Total ≈ **3 GB**. If this proves tight, the first trim is restricting `topk` to
the workspace layer band (the `jlens_readout_sanity` band is layers ≈0.35–0.80
of depth) rather than reducing anchors.

### Length variables

Both are persisted and they are not interchangeable:

- **`ctx_size_reported`** — FLenQA's nominal label, measured with the GPT-4
  tokenizer. Use for grouping and for comparison with the published results.
- **`n_tokens_actual`** — the true Qwen token count. Use as the length variable
  in any quantitative analysis.

## Correctness guards

Every failure mode here is silent, which is what makes it dangerous. Each is an
assertion, not a convention.

**Truncation.** `JacobianLens.apply()` defaults to `max_seq_len=512`, which
flows to `model.encode(prompt, max_length=max_seq_len)` and, in `jlens/hf.py`,
to `tokenizer(..., truncation=True, max_length=max_length)`. With HuggingFace's
default `truncation_side='right'` this keeps the first 512 tokens and discards
the rest with no warning and no exception.

On this dataset the damage is condition-dependent: with `dispersion=first` only
padding is lost; with `dispersion=last` the **key facts are deleted entirely**
and the model answers from padding alone; with `middle` and `random` the loss is
partial and length-dependent. The result would be a strong, clean-looking
`dispersion × length` interaction that is pure tokenizer artifact. `max_seq_len`
is therefore always passed explicitly and the untruncated token count is
asserted before the call.

**Position subsampling.** `positions` must be non-`None` and within budget.
`positions=None` on a 3000-token prompt accumulates ~65 GB of fp32 logits in
**host** RAM; a high-RAM Colab instance has ~83 GB, so it will not fail fast —
it will thrash and die unpredictably, or half-succeed.

**Schema verification on download.** The published sources disagree: the
HuggingFace dataset card appears to list `first`/`middle`/`last`/`random` under
`padding_type`, whereas the paper assigns those to `dispersion` and gives
`padding_type` as `duplicate`/`similar`/`different`. The observed value sets for
`padding_type`, `dispersion`, and `ctx_size` are asserted against expectations
at load time, failing loudly rather than silently mislabeling all 12,000 rows.

**Tokenization identity.** The Jacobian and logit-lens `apply()` calls must
agree on `input_ids` and on baseline logits, as `jlens_readout_sanity` already
checks.

**Model dtype.** bf16 load asserted; fp32 doubles weights to 16 GB and buys
nothing.

**Lens validity at length.** The lens is fitted on wikitext at n=1000 prompts,
almost certainly short ones. Whether the Jacobian remains valid at 3000 tokens
is an assumption, and it is confounded with the exact variable under study. This
is not resolvable within Phase 1; it is the entire motivation for the Phase 2
control, and it is recorded here as a stated limitation of any Phase 1 result.

## Checkpointing and Drive I/O

Google Drive over FUSE runs at roughly 10–30 MB/s and degrades badly with many
small files.

- The dataset is copied to local `/content` once at startup and never read from
  Drive inside the loop.
- Parquet shards of ~500 rows are written to local disk — few large files, not
  many small ones.
- Each completed shard is synced to Drive on a background thread, so GPU work
  does not block on I/O. At ~100 MB per shard across ~24 shards this is ~4
  minutes of transfer spread over a 5–10 hour run, so syncing every shard costs
  almost nothing and caps crash loss at one shard.
- On startup the run scans existing shards and skips completed
  `global_sample_id`s. Colab will disconnect during a run of this length;
  resume is required, not optional.

## Notebooks

**`flenqa_smoke.ipynb`** — L4 GPU, run first. A stratified subset of ~80 rows
(2 base instances × full grid) covering every `padding_type` × `dispersion` ×
`ctx_size` cell and both labels, including 3000-token rows so that truncation
and memory are genuinely exercised. It runs the **identical** code path, asserts
the same guards, and writes the same Parquet schema — a smoke test that bypasses
the real path is worthless. L4 has 24 GB, against ~11 GB peak, and is ~3–5×
slower than A100, so expect ~10 minutes.

**`flenqa_length_drift.ipynb`** — A100 40GB, full 12,000-row split, 5–10 hours.

## Testing

CPU-only and model-free, consistent with the existing CI policy (no repository,
HuggingFace, W&B, or Drive credentials):

- dataset normalization and schema-verification failure
- prompt construction against pinned expected strings
- fact-span location, including the unmatched-span fallback
- bridge derivation, including the unresolvable-bridge null path
- anchor and summary position selection, and budget enforcement
- provenance labeling
- truncation guard raises rather than truncating
- `positions=None` rejected
- Parquet round-trip for all four tables
- resume logic skips exactly the completed ids

Model-backed behavior is mocked.

## Risks

- **Lens validity at 3000 tokens is unverified.** Mitigated only by Phase 2; any
  Phase 1 finding is provisional until then.
- **Prompt format fidelity.** Divergence from the authors' prompts weakens
  comparison to the published accuracy curve. Mitigated by pinning their format
  and by checking that the accuracy-versus-length trend reproduces directionally.
- **Qwen3.5-4B may be too weak** to show the paper's effect cleanly, or may
  floor at long lengths. The `generation` table makes this visible immediately
  rather than after analysis.
- **Storage growth** if anchors expand. Trim path is the workspace layer band.

## Decisions deferred by design

These are recorded so that they are not silently re-litigated:

- Task choice for the drift analysis (PIR / MonoRel / RuleTaker) — all three are
  collected; the choice is an analysis decision.
- Bridge token definition — full top-k at anchors makes this offline-resolvable.
- Which grid axes to feature in the analysis — the full grid is collected.
