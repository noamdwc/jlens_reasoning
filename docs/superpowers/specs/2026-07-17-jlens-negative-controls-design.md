# J-Lens Sanity Negative Controls Design

**Date:** 2026-07-17

## Purpose

Add four mandatory negative controls to the existing J-Lens read-and-change
sanity experiment without changing its prompts, five scored cases, metrics,
thresholds, checks, intervention strengths, or existing gates. The controls
must make it harder for an arbitrary or mechanically ineffective intervention
to satisfy the experiment while preserving the current real `alpha=1` and
`alpha=2` results.

The four new hard gates are:

1. identity intervention invariance;
2. matched random-vector superiority;
3. matched-concept superiority over wrong-concept swaps;
4. real-target superiority over random-token targets.

The global result passes only when every existing check and every control gate
passes.

## Existing Experiment Contract

The implementation retains the five existing `READOUT_CASES` and matching
`SWAP_CASES` exactly:

- the spider prompt with `spider` to `ant`, scored on `6`/`six`;
- the four France prompts with `France` to `China`, scored on `Beijing`,
  `Chinese`, `Asia`, and `Yuan`.

Existing clean-baseline, spider-read, three-of-five rank-improvement, and
any-target-top-1 gates remain unchanged. Existing `alpha=1` and `alpha=2`
outputs remain in their current fields and keep their current meaning.

All control comparisons use `alpha=1`. The real comparator for each case is
the existing matched source-to-target intervention at `alpha=1`, never the
best result across strengths.

## Exact Intervention-Path Parity

Every control uses the same execution path as the corresponding real
`alpha=1` swap. A single internal intervention runner owns:

- the case's existing formatting-adjusted scoring input;
- the same workspace-layer selection;
- the same transformer block hook locations;
- patching every activation position in each hooked block;
- the same source and target vector preprocessing;
- `LensCoordinateSwapper` and `coordinate_swap` execution;
- one next-token forward pass at `alpha=1`;
- the existing intended-target token variants and rank calculation.

Real and control calls differ only in the source and target token IDs or in
the deliberately supplied matched-norm random vectors. They do not use raw
embeddings, unembedding rows, or alternate activation patching.

Token-backed vectors, including random-token targets, are always constructed
through the current J-Lens path:

```text
v[token, layer] = J[layer]^T W_U[token]
```

Identity uses all workspace-layer hooks together, exactly as the real swap
does. It does not run one layer at a time and does not return cached clean
outputs in place of an intervened forward.

## Shared Deterministic Definitions

### Seeds

One module-level tuple contains exactly 16 explicitly listed integer seeds.
Both randomized controls use this tuple, and `result.json` records it once at
`controls.seeds`.

Random-vector sub-seeds are derived from the ordered tuple
`(base_seed, layer_index, role)`, where role distinguishes source and target.
The derivation uses a documented stable arithmetic or SHA-256 mapping. It must
not use Python `hash()` or iteration order from sets or dictionaries. Layers
are processed in ascending numeric order.

All base random tensors are generated on CPU in `float32`, making fixed-seed
generation reproducible in CPU-only tests and independent of the model's
device. After deterministic generation and norm matching, every random vector
is converted back to the exact device and dtype of its corresponding real
vector before entering the shared intervention executor. Norm validation is
performed on the converted vector with tolerances appropriate to that dtype.

### Log-rank gain

Every scored case uses natural logarithms:

```text
log_rank_gain = log(clean_rank) - log(intervened_rank)
```

A positive value means the intervention improved the intended real target's
rank. Aggregate scores are the arithmetic mean of the five case-level gains.

### Percentile

The deterministic 95th percentile uses sorted values and linear interpolation
at zero-based position:

```text
h = (n - 1) * 0.95
threshold = values[floor(h)]
          + (h - floor(h)) * (values[ceil(h)] - values[floor(h)])
```

With 16 values, `h = 14.25`. Percentile gates use strict comparison:

```text
real_mean > percentile_95
```

Equality fails.

These 16-sample percentile comparisons are deterministic negative-control
sanity checks. They are not hypothesis tests, confidence intervals, or evidence
of statistical significance, and the result metadata and notebook wording must
not describe them as such.

## Identity Control

For the spider case, run `spider` to `spider`. For every France case, run
`France` to `France`. Each case uses its existing prompt, formatting-adjusted
input, intended real target answer, selected workspace layers, all-position
hooks, and the shared `alpha=1` intervention runner.

For every case require:

- intervened top-1 token ID equals clean top-1 token ID;
- intervened intended-target rank equals clean intended-target rank;
- `torch.allclose(clean_logits, identity_logits, atol=1e-6, rtol=1e-5)`.

The result records the absolute and relative tolerances, active workspace
layers, per-case comparisons, each case's maximum absolute logit difference,
and the maximum observed difference across all cases. The identity gate passes
only when every comparison for every case passes.

## Matched Random-Vector Control

For each real case, seed, workspace layer, and role, generate source and target
random vectors separately. Each random vector has the same shape as its
corresponding real J-Lens vector and is rescaled so its per-layer L2 norm
matches the real vector's norm.

If the real norm is zero, return an exact zero vector of the same shape. If a
generated random vector has zero norm while the real norm is nonzero, use a
documented deterministic basis-vector fallback before scaling. Norm checks use
explicit numerical tolerances suited to CPU `float32` calculations.

Each seed runs the same full-layer hook intervention at `alpha=1` and continues
to score the original real target answer. For each seed, compute the arithmetic
mean of the five log-rank gains. The gate is:

```text
real alpha=1 mean gain > 95th percentile of 16 random-vector mean gains
```

The result records per-layer real and generated norms, device/dtype parity,
per-case ranks and gains, every seed mean, the real per-case gains and mean,
the percentile threshold, the gate outcome, and a statement that this is a
sanity check rather than a statistical-significance claim.

## Wrong-Concept Control

Run deliberately mismatched real token-backed swaps through the shared
`alpha=1` execution path:

- `France` to `China` on the spider case;
- `spider` to `ant` on each France case.

Each prompt continues to score its original real target answer. The matched
and mismatched aggregates are arithmetic means of all five case-level
log-rank gains.

A case is a matched win only when:

```text
matched_gain > mismatched_gain
```

Ties do not count. The gate requires both:

- matched aggregate mean is strictly greater than mismatched aggregate mean;
- matched wins at least four of the five cases.

The result records both per-case gains and ranks, strict comparison outcomes,
aggregate means, the winning-case count, both required conditions, and the
combined gate.

## Random-Target Control

Keep each case's real source (`spider` or `France`) and replace its target with
each of 16 deterministic vocabulary token IDs. A sampled target token always
uses the same J-Lens target-vector construction and preprocessing as real
`ant` and `China` targets. Each sampled token is applied to all five cases,
with each case continuing to score its original real intended target answer.

### Token-ID exclusions

Build exclusions after tokenization. The union contains:

- every ID in every real source concept token sequence;
- every ID in every real target concept token sequence;
- every ID in all clean-answer and intended real target-answer tokenizations
  and accepted answer variants;
- IDs observed in the existing per-case prompt-formatting prefixes;
- vocabulary IDs whose decoded surface is empty or whitespace-only;
- tokenizer special, control, padding, beginning-of-sequence,
  end-of-sequence, unknown, and other reserved IDs;
- every ID rejected by any existing sanity-test token filter.

For a multi-token concept or answer, every member token ID enters the exclusion
set. Exclusions are compared as integer token IDs, never by decoded-string
equality.

Eligible IDs are sorted ascending. Each shared base seed maps
cryptographically or arithmetically to an index in the remaining ordered
candidate list. The selected ID is removed before the next selection, making
targets unique when at least 16 eligible IDs exist. If fewer than 16 IDs are
eligible, deterministic reuse is allowed only after the eligible set has been
exhausted. Selection never depends on unordered collection iteration.

The candidate pool is also intersected with the model output vocabulary:

```text
0 <= token_id < unembedding_weight.shape[0]
```

Tokenizer-only IDs that the causal language-model head cannot score or map
through `W_U` are therefore ineligible. Token strings are decoded everywhere
with the existing convention
`tokenizer.decode([token_id], clean_up_tokenization_spaces=False)`.

For each target, compute the arithmetic mean of five case-level gains. The gate
is:

```text
real alpha=1 mean gain > 95th percentile of 16 random-target mean gains
```

The result records every selected token ID and readable decoded string, model
output-vocabulary size, exclusion categories and IDs, per-case ranks and gains,
target means, the real score, percentile threshold, gate outcome, and the same
non-significance statement used by the random-vector control.

## Exact Case-Set Invariant

All real and control comparisons require the exact ordered keys from the five
existing scored cases. Before aggregation, validate that real matched results,
identity results, every random-vector seed, wrong-concept results, and every
random-target result contain each of those five keys exactly once, with no
missing, duplicate, or additional cases. A malformed case set raises a clear
error instead of calculating a partial mean or percentile.

## Result Schema and Gate Aggregation

Preserve existing top-level fields and add:

```text
controls:
  seeds
  definitions
  thresholds
  tolerances
  identity
  matched_random_vector
  wrong_concept
  random_target
  passed
```

Each control contains configuration, per-case results, repeated-run results
where applicable, aggregate values, thresholds or required conditions, and a
`passed` field.

The two overall states have deliberately different scope:

- `controls["passed"]` is true exactly when all four control gates pass;
- top-level `result["passed"]` is true exactly when every pre-existing sanity
  check and all four control gates pass.

One explicit aggregation function receives the existing checks, existing
failures, and all four control payloads. It iterates one authoritative mapping
from control name to global check key, adds all four booleans to the existing
top-level `checks`, appends a control-specific actionable failure for each
failed gate, derives `controls["passed"]` from the four control payloads only,
and derives global `result["passed"] = all(checks.values())`.

Failure text includes observed values and the strict threshold or failed
condition. There is no second pass/fail assembly path in the notebook or
runner that could omit a control.

## Notebook Reporting

After writing `result.json`, print a concise summary containing:

- identity pass/fail and maximum absolute logit difference;
- real matched mean, matched-random 95th percentile, and pass/fail;
- matched and wrong-concept means, matched win count, and pass/fail;
- real mean, random-target 95th percentile, and pass/fail;
- overall controls pass/fail.

Seed-by-seed arrays remain in JSON. The notebook may print additional repeated
details only when a control fails and they materially aid diagnosis.

## Code Organization

### `sanity_controls.py`

A focused module owns deterministic and directly testable control logic:

- the shared seed tuple and numerical tolerances;
- natural-log rank gain and arithmetic means;
- deterministic linear-interpolation percentile and strict gate;
- stable sub-seed derivation;
- deterministic matched-norm vector generation;
- token-ID exclusion construction and deterministic target selection;
- exact five-case-set validation;
- wrong-concept comparison calculation;
- control/global check aggregation and failure formatting.

### `readout_sanity.py`

The existing experiment module retains case definitions, J-Lens vector
construction, coordinate swapping, hooks, scoring-input preparation, rank
calculation, and result serialization. It gains one shared `alpha=1`
intervention execution helper used by the real comparator and all controls,
plus orchestration that assembles the control payloads.

### Notebook

The notebook continues to load the model and lens, call
`run_readout_sanity`, write `result.json`, render the two existing J-Lens
views, and enforce the final global pass. Only its concise textual reporting is
expanded.

## Testing

CPU-only tests use small tensors, tiny blocks, fake logits, and tokenizer/model
stubs without downloads, network access, or GPU requirements. They cover:

1. fixed-seed random-vector determinism;
2. deterministic sub-seed mapping independent of unordered iteration;
3. deterministic random-target selection and uniqueness;
4. selected random-target IDs bounded by the model output vocabulary;
5. token-ID exclusions for all real source, target, clean-answer,
   intended-target-answer, multi-token answer, formatting, reserved, special,
   and existing-filter IDs, with consistent decoding;
6. per-layer source and target norm matching, including zero norms, plus exact
   device and dtype parity with real vectors;
7. identity execution through the real hooks with unchanged top-1, unchanged
   rank, and logits within `atol=1e-6`, `rtol=1e-5`;
8. hook cleanup when a control forward raises;
9. natural-log rank-gain calculation;
10. deterministic percentile interpolation, strict equality failure, and
    non-significance metadata;
11. wrong-concept arithmetic means, strict case wins, and the four-of-five
   requirement;
12. rejection of missing, duplicate, or extra cases in every comparison;
13. all four controls in global checks;
14. separate control-only and global pass semantics;
15. any failed control forcing global failure and adding an actionable,
    control-specific failure;
16. stable JSON serialization of seeds, definitions, tolerances, thresholds,
    per-case results, repeated-run results, and gate outcomes;
17. notebook presence of the concise control summary.

Tests exercise the real tensor transforms, hook context manager, rank logic,
and aggregation functions. Stubs replace only heavyweight model inference and
tokenization infrastructure.

Repeated control execution retains only the small JSON-ready score payload for
each condition. Logit tensors are consumed immediately to calculate top-1,
rank, identity tolerance diagnostics, and gain, then references are released.
The implementation must not accumulate logits from the roughly 170 additional
control forwards in lists, dictionaries, dataclasses, closures, or the result
artifact.

## Validation

After implementation run:

```bash
.venv/bin/pytest tests/test_sanity_controls.py tests/test_readout_sanity.py -q
.venv/bin/pytest -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
git diff --check
```

Any change-caused failure is fixed. Unrelated pre-existing failures, if any,
are reported separately. The pre-existing uncommitted README modification is
preserved and excluded from this work's commits.
