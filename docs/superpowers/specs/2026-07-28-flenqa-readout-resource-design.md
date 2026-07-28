# FLenQA Lens-Readout Resource

Date: 2026-07-28
Status: design

## Purpose

Produce a **reusable resource**: a lens readout of every unique FLenQA prompt,
plus the module that builds it. FLenQA fixes the reasoning task and the key
facts and varies only padding, so it isolates input length from task difficulty.
The first consumer will be a length-drift analysis, but nothing here is specific
to that analysis — this deliverable is the dataset and the code that produces
it.

Out of scope: the drift analysis itself, any statistics, and any experiment
package. Those import this module.

## Core principle

**The prompt is the unit of analysis, not the dataset row.** A row is a recipe;
the prompt is the input. One unique prompt is one observation; condition labels
are derived from prompt content, never copied from row metadata; row identity is
provenance only, never a key or seed.

**Keep the implementation small.** Three modules and one notebook, roughly 320
lines. Where a rule fits in twenty lines inside an existing module, it does not
get its own module.

## Verified dataset facts

Verified directly against the published parquet (`alonj/FLenQA`, `eval`), not
the paper or the dataset card, both of which are wrong in places. Every number
below was re-checked on 2026-07-28.

| Claim | Verified value |
| --- | --- |
| Rows | 12,000 — 4,000 each for PIR, MonoRel, Simplified RuleTaker |
| `global_sample_id` | **300 unique values, each appearing exactly 40×** — it names the problem, not the row |
| `padding_type` | `{books, same}` — 2 values, not the paper's 3 |
| `dispersion` | `{first, middle, last, random}` |
| `ctx_size` | `{250, 500, 1000, 2000, 3000}` |
| `label` | exactly `"True"` / `"False"`, balanced |
| Unique prompts | **9,862** on `(dataset, mixin, question)` |
| Unique prompts by length | 300 / 2368 / 2394 / 2400 / 2400 |

**The prompt templates are the authors' own.** They are not in the dataset repo
and not in the paper; they live in the authors' analysis notebook,
`alonj/Same-Task-More-Tokens` → `FLenQA analysis.ipynb`, as a `prompt_structures`
dict of per-task lambdas (we take the three non-CoT variants). Rendering them
verbatim over all 12,000 rows reproduces exactly **9,862** unique prompts with
the same per-length breakdown, and no prompt group spans more than one
`ctx_size`, `global_sample_id`, or `label` — so `EXPECTED_PROMPTS = 9862` is a
measured fact under the real templates, not an assumption. Two quirks are
reproduced deliberately rather than fixed: the RuleTaker template's unbalanced
quote (`"True or "False"`), and `rule` rendering with its list brackets
(`Rule: ['If X is …']`) because the column stores a list. Both appear in the
published prompts; silently cleaning them would measure a prompt the paper never
ran. All three templates end with a trailing newline, so the True/False answer
token follows `\n` — token-boundary choice is pinned by the smoke run.

**`ctx_size=250` contains no padding.** For all 1,600 PIR and MonoRel rows at
250, `mixin` is exactly the key paragraphs joined by a newline. Every problem
has exactly one unique prompt there, so all 8 padding × dispersion combinations
are byte-identical: those labels describe a distinction that does not exist in
the input.

**RuleTaker is structurally different.** `facts` is null for all 4,000 rows; it
uses `statement` (bare sentences whose paragraph expansions appear in `mixin`)
plus `rule` — and **the rule never appears in `mixin`**, so the template must
inject it or the task is unanswerable. PIR and MonoRel `facts` are already full
paragraphs, each occurring once in `mixin`, with the key sentence restated ~5×.

**Token counts track `ctx_size`** (measured with Qwen3-4B as proxy): mean
277/502/1000/2003/3002, max observed 3077. `max_seq_len = 4096` is safe.

## The bridge

Each PIR and MonoRel problem chains two facts through an entity that appears in
**both facts but not in the question**, and not in the answer either:

- PIR — Fact A "*John's living room* is marble-floored", Fact B "Ethan
  Washington is in *John's living room*", question "Is Ethan Washington in a
  marble-floored room?" → bridge `John's living room`.
- MonoRel — the middle person: "Julie Baker is younger than Julian Barton",
  "Samantha Arnold is younger than Julie Baker", question "Is Samantha Arnold
  younger than Julian Barton?" → bridge `Julie Baker`.
- RuleTaker — **no bridge**. Every entity named in the statements also appears
  in the question.

The bridge is the concept the model must construct internally and never sees in
the question nor emits in the answer. It is the measurement a lens can make that
the output cannot: accuracy says the model was wrong, bridge rank says whether
the linking concept ever formed.

**Extraction rule, verified on all 300 problems.** A generic
longest-common-substring rule fails — the fact paragraphs share filler
boilerplate. Two small task-specific rules resolve every problem:

- **MonoRel** — capitalised name in both facts, minus names in the question.
- **PIR** — anchor on each `X's` possessive, extend across both facts while they
  agree, trim to a whole word, drop candidates present in the question.

Verified outcomes: **300/300 resolve** (100 PIR, 100 MonoRel, 100 RuleTaker
correctly reporting no bridge); **0 bridges leak into the question**; **0 of the
8,000 PIR/MonoRel rows lack the bridge in `mixin`**.

Two further verified facts make the bridge a sound measurement:

- **The bridge never appears in padding** — for both padding types at every
  length, the bridge occurrence count is identical to that problem's 250-token
  count (mean and max delta 0.0). Padding positions are therefore a clean
  control region.
- **Occurrence count is constant across lengths**, so a change in bridge rank
  cannot be explained by the bridge simply being mentioned more or less often.

The bridge recurs 4–11× per prompt. The anchor uses the **last** occurrence
within each fact; the count is recorded.

## Module layout

```
src/jlens_reasoning/flenqa/
  __init__.py    public API
  dataset.py     rows → prompt table: schema checks, templates, dedup,
                 conditions, bridge, key-span offsets
  readout.py     prompt + positions + targets → lens records
  storage.py     parquet shards, resume by completed prompt_id
notebooks/02_build_flenqa_readout.ipynb
```

Output paths come from the existing `config.create_artifact_paths()` — the
prompt table under `datasets/`, readout shards under `runs/`. No new path
constants, and it works unchanged on Mac, Colab, and CI.

### Public API

```python
load_prompts(paths, *, rebuild=False) -> pd.DataFrame
read_prompt(model, lens, prompt, *, positions, targets, layers) -> Records
write_shard(path, records) / completed_prompt_ids(path) -> set[str]
```

`read_prompt` takes prompt, positions, and targets **from its caller**, not from
a FLenQA row. A later experiment supplying synthetic or re-padded prompts reuses
it unchanged; that seam is the reason this is a resource and not an experiment.

`dataset.py` is where the FLenQA knowledge lives — verified schema values, the
three templates, dedup, content-derived conditions, the bridge rules. It is
tokenizer-free and model-free, so the whole build phase is CI-testable.

## Identifiers

| Identifier | Grain | Use |
| --- | --- | --- |
| `problem_id` | reasoning problem; 300 values | grouping for paired comparisons |
| `prompt_id` | hash of the final templated prompt | storage, joins, seeds, resume |
| `source_row_id` | one dataset row | provenance only |

`prompt_id` is computed over the string the model actually sees, after
templating. Dedup is recomputed on that string and the resulting count
**asserted**, not assumed from the 9,862 measured on `(dataset, mixin,
question)` — templating could merge or split groups.

## Condition variables, derived from content

- **Length** — `n_tokens_actual`, measured. `ctx_size` is kept as an unambiguous
  grouping label.
- **`padding_type_effective`** ∈ {`none`, `books`, `same`}. `none` means the
  **minimum-length prompt for its problem**, never an inference from `ctx_size`.
  Verified: this selects exactly 300 prompts, one per problem. For PIR and
  MonoRel it is independently cross-checked against strict content equality
  (`mixin` is the key paragraphs and nothing else) — the two rules agree on all
  200, with **zero** false positives among the other 9,562 prompts. RuleTaker
  cannot use the content check, because its `mixin` holds sentence *expansions*
  rather than the bare `statement` values, so the minimum-length rule is what
  makes the test task-agnostic.
- **Placement** — `frac_padding_before` / `_between` / `_after`, computed from
  character offsets of the key spans. `placement_effective` derives from these:
  `not_applicable` (unpadded), `facts_first` (padding-before ≈ 0), `facts_last`
  (padding-after ≈ 0), else `facts_middle`.

The placement rule was validated against the declared labels on a stratified
sample: declared `first` → `facts_first` 144/144, declared `middle` →
`facts_middle` 144/144, declared `last` → `facts_last` 96 with 48 landing in
`facts_middle` (the two facts separated, with padding following the second), and
declared `random` spreading across all three as expected.

An earlier four-way rule with a `scattered` bucket was rejected: it swallowed
123 of 144 declared-`middle` prompts, describing the threshold rather than the
input. The fractions are the measurement; the categorical label is a
convenience over them.

`random` is dropped as a category — it is a generation procedure, not a property
of an input. Deriving placement from content also resolves the prompt groups
whose declared `dispersion` disagrees across their source rows (the 300 unpadded
prompts, plus incidental collisions where `random` landed on a named
arrangement), making them analysable rather than a junk bucket.

Character offsets, not token offsets, keep the build phase tokenizer-free; the
values are fractions, so the classification is equivalent. Recorded as
`placement_basis="characters"`.

## Positions

Roughly eight per prompt, all locatable by exact substring search because
`facts` and `statement` values appear verbatim in `mixin`:

| Position | Why |
| --- | --- |
| `final_token` | predicts True/False — the primary |
| `question_end` | after the question is read |
| `fact_a_end`, `fact_b_end` | each fact just integrated |
| `bridge_fact_a`, `bridge_fact_b` | last bridge mention within each fact |
| `padding_1`, `padding_2` | sampled from padding, seeded by `prompt_id` |

Padding positions are the control: padding is where nothing should be forming,
so without them a length effect at the final token has no internal comparison.

Positions that do not exist — RuleTaker has no bridge, 250-token prompts have no
padding — are **recorded absent and the prompt is kept**. Char→token conversion
uses `return_offsets_mapping`, never re-tokenising the substring.

Spans are located by **find-all, never first-match**, and the match count is
always recorded, because the two task families differ:

- **PIR / MonoRel** — each fact paragraph occurs **exactly once** in `mixin`
  (verified across 2,000 sampled rows). A count other than 1 means the template
  or the data changed, and is a hard error.
- **RuleTaker** — `statement` values are short sentences that **recur 2–14 times
  per row**, since `mixin` both states and expands them. The first occurrence is
  used, and `span_match_count` records the rest.

`span_status` ∈ {`ok`, `unresolved`} is recorded per span; `unresolved` (count 0)
yields null positions and the prompt is kept.

## Measurements

At each (prompt, position, layer, lens):

- **exact rank and logit of the bridge tokens** — computed on GPU against the
  full vocabulary, not read off a truncated top-k. At long contexts a bridge
  token ranking far below 25 is the expected finding, so a top-k-only design
  would censor the primary signal.
- **exact rank and logit of the True/False tokens** — free from `model_logits`.
- **top-25** ids and logits, for exploration.
- **entropy, max logit, top1 id.**

Both lenses (Jacobian and logit-lens baseline) are read at the **same**
positions, so the comparison is matched. Layers come from `lens.source_layers`,
never assumed. Token strings are never repeated in storage — records hold
`token_id`, with one `vocab` table mapping ids to strings.

## Scoring

`label` is exactly `"True"`/`"False"` and balanced, so scoring is a binary
comparison of the final-position logits for the two answer tokens. This is
deterministic and free, since `apply()` already returns `model_logits`.

**No generation pass.** A short generation with regex extraction was considered
and dropped: it roughly doubles GPU time to produce a noisier version of the
same binary answer.

## Storage and sizing

Parquet shards keyed on `prompt_id`. 9,862 prompts × ~8 positions × ~36 layers ×
2 lenses ≈ 5.7 M rows; at top-25 that is ~142 M top-k entries, ≈1.5–2 GB total.

Holding positions to ~8 rather than ~40 is what keeps the output small enough
that shard handling stays simple. Trim order if tight: top-k 25→10, then
restrict top-k to the workspace layer band. Neither touches the bridge measures.

Shards of ~500 prompts are written locally and **atomically** (temp path then
rename), then synced to Drive — Drive over FUSE runs ~10–30 MB/s and degrades on
many small files, so the dataset is copied to local `/content` once and never
read from Drive in the loop. Crash loss is capped at one shard.

**Resume** scans completed `prompt_id`s in existing shards. A single config hash
(model, lens revision, template, budgets, top-k, bridge rule, dedup rule) is
stored and **aborts on mismatch** rather than blending incompatible runs.

## Guards

Each is cheap and each catches a silent failure:

- **Truncation.** `apply()` defaults to `max_seq_len=512`, reaching
  `tokenizer(..., truncation=True)` at `jlens/hf.py:157`; HuggingFace's default
  right-side truncation would silently keep the first 512 tokens. Damage would
  be placement-dependent — padding-before means the **key facts are deleted** —
  producing a clean-looking placement × length interaction that is pure
  artifact. Pass 4096 explicitly and assert the untruncated count.
- **Position subsampling.** `positions` is never `None`. At 3000 tokens
  `positions=None` accumulates ~65 GB in **host** RAM against Colab's ~83 GB; it
  would not fail fast.
- **Schema.** Assert the verified value sets and counts — the released data
  already contradicts both the paper and the dataset card.
- **Dedup invariants.** No group mixes `ctx_size`, problem, or `label`; the
  unpadded set is exactly the 300 content-verified prompts;
  `padding_type_effective = none` is never assigned where padding exists.
- **Bridge.** All 300 problems resolve; no bridge appears in its own question.
- **Tokenisation identity** across the two `apply()` calls; **bf16** asserted.

## Pre-flight lens-validity gate

The lens is fitted on wikitext at n=1000, almost certainly short prompts.
Whether the Jacobian holds at 3000 tokens is an assumption **confounded with the
variable under study**. Before committing hours of A100 time: take the validated
`spider` case from `jlens_readout_sanity`, pad to ~250/~1000/~3000 tokens, and
confirm the J-Lens still surfaces the target and still beats the logit lens at
each length. If the advantage collapses, a lens-length control comes first.

This is one notebook cell, not a module.

## Notebook

`notebooks/02_build_flenqa_readout.ipynb`, following `_template.ipynb`: the two
bootstrap cells, then build the prompt table on CPU with the count asserts, run
the pre-flight gate, load model and lens, loop with resume, sync shards.

A `LIMIT` constant at the top covers the smoke run — a stratified handful of
prompts across every length including 3000, exercising truncation and memory on
the identical code path — so there is no second notebook. The smoke run
**measures wall-clock per prompt** by `ctx_size` and prints an extrapolated
full-run estimate.

Full run: 9,862 prompts on one A100 40GB. Estimated 4–8 hours at 1.5–3 s per
prompt, batch 1 (`apply()` has no batched path). Peak VRAM ≈11 GB.

## Testing

CPU-only, model mocked, per existing CI policy, in `tests/flenqa/` — one file
per module. Coverage: schema verification and its failure modes; prompt
construction including RuleTaker rule injection; stable content-determined
`prompt_id`; dedup invariants; padding-free detection from content; `none`
rejected when padding is present; the unpadded cross-check failing loudly when
the minimum-length and content rules disagree; placement fractions and
`placement_effective`; bridge extraction across all three tasks plus the
300-problem gate; span location (exact / ambiguous / unresolved) and
offset-mapping boundaries; position selection with missing positions; seeds
stable from `prompt_id`; deterministic scoring; truncation guard raises;
`positions=None` rejected; parquet round-trip, atomic-write partial-shard
safety, resume skipping exactly the completed prompts and aborting on config
mismatch.

## Assumptions still to verify

1. **Model identity** — token counts and vocab (151,669) were measured with
   **Qwen3-4B as proxy**, while the project targets `Qwen/Qwen3.5-4B`. Confirm
   tokenizer, layer count, `d_model`; storage sizing scales with layer count.
   Model and lens coordinates live in the notebook, not in the module — they are
   a property of a run, not of the dataset.
2. **`lens.source_layers`** — may be a subset; 36 layers is an estimate.
3. **Drive throughput and GPU availability** in the actual session.
4. **Model capability** — Qwen3.5-4B may floor at long lengths; the scoring
   column makes this visible immediately.

## Deferred by design

Which tasks, which grid axes, and which layers to feature are analysis
decisions. The full grid is collected, and exact per-candidate ranks plus top-k
make them resolvable offline without re-running the GPU job.
