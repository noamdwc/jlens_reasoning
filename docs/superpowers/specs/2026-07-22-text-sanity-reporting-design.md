# Text-Only Sanity Reporting Design

**Date:** 2026-07-22

## Purpose

Replace the two generated J-Lens HTML explorers with notebook-native textual
reporting that makes the sanity run's evidence easy to interpret. Every check
must show the observed metric, the configured Qwen sanity threshold, the
binary pass/fail outcome, and any directly comparable paper target and gap.

This change improves reporting only. It does not alter model inference,
interventions, controls, configured Qwen gates, or the meaning of
`result["passed"]`.

## Scope

The experiment will stop generating:

- `spider.html`;
- `france_capital.html`;
- embedded notebook iframes or any other graphical replacement.

`result.json` remains the complete persistent artifact. The notebook will
render deterministic plain-text tables from the result and will retain its
final exception when a configured sanity check fails.

## Status and Comparison Semantics

Each check has exactly one status: `PASS` or `FAIL`. The status is copied from
the existing Qwen sanity gate and is never derived from the paper comparison.

Each report row contains:

- check name;
- observed value;
- configured sanity threshold;
- margin to the sanity threshold when a meaningful numeric margin exists;
- binary status;
- paper target;
- paper gap.

The paper comparison is diagnostic. It never affects `result["passed"]`.
Where the paper does not define a directly comparable numeric threshold, the
report uses `N/A - no paper threshold`. It must not invent a paper requirement.

### Spider readout

The Qwen sanity gate remains:

- best Jacobian-lens rank at most 5; and
- best Jacobian-lens rank strictly better than the best ordinary logit-lens
  rank.

The paper-aligned target is Jacobian-lens rank 1. The paper gap is:

```text
best Jacobian-lens rank - 1
```

A rank of 3 therefore reports a paper gap of `+2 ranks`. The report also shows
the logit-lens rank so the reader can evaluate the second half of the Qwen gate.

### Coordinate swaps

The aggregate Qwen gates remain:

- at least three of five swaps strictly improve the target rank at either
  configured intervention strength;
- at least one of five swapped targets reaches rank 1 at either configured
  intervention strength.

The paper does not define equivalent aggregate count thresholds, so the two
aggregate rows report `N/A` for paper target and paper gap.

The per-swap report treats `alpha=1` as the primary paper-style metric. It shows
the clean target rank, `alpha=1` rank, `alpha=2` rank, best intervened rank,
whether the target improved, whether it reached top-1, and the diagnostic
paper gap:

```text
alpha=1 target rank - 1
```

This per-case paper gap does not imply that the paper required every swap to
succeed.

### Baselines and negative controls

Clean-baseline and negative-control checks retain their configured gates and
show their observed value, threshold or required condition, sanity margin,
and binary status. Their paper target and paper gap are
`N/A - no paper threshold`.

## Architecture

### Result metadata

The runner will attach the policy values needed to explain the exact run to
the serialized result. At minimum this includes:

- the spider maximum rank;
- the required minimum number of improved swaps;
- the required number of swaps reaching target top-1;
- the configured intervention strengths;
- existing control thresholds and tolerances.

The metadata is descriptive. Existing check booleans remain the only source of
truth for pass/fail.

### Reporting module

A focused pure module under `experiments/jlens_readout_sanity/` will transform
an assembled result dictionary into deterministic report rows and formatted
plain text. It will not run inference, recompute ranks, or independently decide
whether a check passed.

Separating report construction from the notebook makes the output testable and
prevents notebook formatting logic from drifting away from the result schema.

### Notebook

The existing report cell will call the reporting module and print these
sections in order:

1. overall result and provenance;
2. aggregate capability checks;
3. per-case readout details;
4. per-swap clean and intervened ranks;
5. negative-control checks;
6. failure messages, when present.

The notebook will not import `jlens.vis`, call `compute_slice`, call
`build_page`, create an iframe, or write an HTML file.

## Artifact and Documentation Changes

The documented run directory becomes:

```text
runs/jlens-readout-sanity/
└── result.json
```

Documentation will explain the distinction between:

- the configured Qwen sanity threshold that controls binary status; and
- the diagnostic paper target and gap, where a direct comparison exists.

Generated HTML artifacts remain safe to delete because they are not inputs and
will no longer be regenerated.

## Error Handling

Reporting must tolerate failed checks and print their observed values before
the notebook raises its final summary error. A structurally incomplete result
is a programming error: the reporting module should fail with a clear missing
field or invalid-shape exception rather than silently display misleading
defaults.

Paper comparisons that are not defined are expected data, not errors, and are
rendered explicitly as `N/A`.

## Testing

CPU-only tests will cover:

- exact capability-report rows and binary statuses;
- spider paper-gap calculation and rank direction;
- per-swap `alpha=1` paper-gap calculation;
- aggregate, baseline, and control `N/A` paper comparisons;
- sanity-threshold margins on both passing and failing sides;
- threshold metadata in the JSON-compatible result;
- deterministic plain-text formatting;
- notebook reporting-cell separation;
- removal of all `jlens.vis`, HTML-writing, iframe, and visualization calls;
- documentation listing `result.json` as the only run artifact.

The existing runner, control, notebook-structure, serialization, Ruff, and full
test suites must continue to pass.

## Non-Goals

This change does not:

- alter any Qwen sanity threshold;
- add a third or semi-pass status;
- make paper gaps part of pass/fail;
- infer paper thresholds where none were specified;
- add charts, HTML, rich-display widgets, or another visualization system;
- change model prompts, tokenization, inference, interventions, or controls.
