# Control Module Split Design

**Date:** 2026-07-22

## Purpose

Refactor `experiments/jlens_readout_sanity/controls.py` so that pure analysis,
model-backed execution, and public orchestration are easy to understand in
isolation. The current module is 617 lines, and its main readability problem is
the roughly 289-line `run_negative_controls()` function, which prepares shared
inputs and implements four distinct controls inline.

This is a structure-only refactor. It must not change experiment behavior,
control semantics, serialized results, randomness, thresholds, or public
imports.

## Readability Principle

Line counts are guidance, not acceptance criteria. Prefer cohesive modules and
functions over arbitrary extraction solely to satisfy a size limit.

Expected sizes are approximately:

- `controls.py`: 60-100 lines;
- `control_analysis.py`: 180-240 lines;
- `control_execution.py`: 280-340 lines.

The total line count may stay similar or increase slightly because explicit
module boundaries require imports and interfaces. A focused 250-350-line
module is acceptable. Functions should generally stay below roughly 60-80
lines, but a longer function is acceptable when splitting it would obscure one
cohesive operation.

The primary success criterion is navigability: a reader should know which file
to open for pure result logic, intervention execution, or top-level
orchestration.

## Module Responsibilities

### `control_analysis.py`

This module owns work that does not execute a model forward or intervention:

- extracting the configured `alpha=1` intervention payload;
- validating exact case keys and order;
- deriving control metadata and wrong-direction references from prepared
  intervention contexts;
- constructing real rank-gain case payloads from existing swap results;
- calculating wrong-concept comparisons;
- deciding whether all controls passed;
- formatting actionable control failure messages;
- adding control booleans to the global check set;
- assembling the final control result envelope, definitions, thresholds, and
  tolerances.

Tensor-to-number rank summarization is allowed here when it is deterministic
analysis of already-produced logits and does not invoke the model.

### `control_execution.py`

This module owns operations that execute interventions, model forwards, or
randomized control construction. The current inline orchestration will be
extracted into four named functions:

1. identity control execution and aggregation;
2. matched-random-vector control execution and aggregation;
3. wrong-concept intervention execution;
4. random-target selection, vector construction, execution, and aggregation.

Shared execution helpers, such as conversion of intervened logits to a rank
gain payload, live here. Each control function returns the same dictionary
payload currently assembled inside `run_negative_controls()`.

The four functions may share explicitly passed prepared metadata and real-case
summaries. Do not introduce a service class or inheritance hierarchy merely to
reduce argument counts.

### `controls.py`

This module remains the stable public facade. It owns:

- `run_negative_controls()` as a short orchestration function;
- stable re-exports for helpers currently imported from this module;
- imports used by `runner.py` without requiring caller changes.

The orchestrator validates the required five contexts, prepares shared
analysis inputs, calls the four execution functions in the current order, and
passes their results to the pure final assembler.

## Data Flow

The new flow is:

```text
contexts + swap results
        |
        v
pure preparation and validation
        |
        +--> expected keys, metadata, real alpha=1 cases, real mean
        |
        v
four model-backed control executors, in existing order
        |
        v
pure result-envelope assembly
        |
        v
unchanged controls dictionary
```

Shared values are computed once. In particular:

- real `alpha=1` rank-gain cases and their mean are not recomputed by each
  control;
- wrong-direction references are derived once;
- random-target exclusions are derived once;
- expected case ordering is validated before expensive execution begins.

## Compatibility and Behavior Preservation

The refactor preserves:

- the exact serialized control schema and field names;
- control and top-level pass/fail behavior;
- random seeds, namespaces, target selection, and vector generation;
- model-forward and intervention order;
- `alpha=1` selection behavior;
- thresholds, tolerances, comparisons, and failure messages;
- validation and exception behavior;
- public imports from `experiments.jlens_readout_sanity.controls`.

The public facade will re-export these currently imported functions:

- `analyze_identity_case`;
- `summarize_wrong_concept`;
- `require_exact_cases`;
- `controls_passed`;
- `aggregate_all_checks`;
- `run_negative_controls`.

Private helpers may move without compatibility aliases unless repository code
or tests intentionally exercise them as part of a documented contract.

Tests that monkeypatch intervention execution must patch the symbol in
`control_execution.py`, where it is actually used after the refactor. The
refactor should not add dependency injection solely to retain an obsolete
monkeypatch location.

## Error Handling

Validation remains front-loaded. Missing, duplicated, extra, or reordered
cases fail before expensive controls execute. Missing `alpha=1` payloads,
unavailable wrong directions, invalid random-target vocabularies, and missing
control result payloads retain their current exception types and messages.

Control failures remain data, not exceptions: each control returns its payload
and boolean, then global aggregation appends the same actionable failure text.

## Test Organization

Tests will mirror the new responsibilities:

- `test_control_analysis.py`: metadata, case validation, pure rank summaries,
  wrong-concept comparison, pass aggregation, failure messages, and stable
  serialization;
- `test_control_execution.py`: identity execution, hook/model calls, matched
  random vectors, wrong-direction interventions, deterministic random-target
  selection, and per-control payloads;
- `test_controls.py`: facade exports and complete end-to-end orchestration.

Existing tests should move rather than be duplicated. The integration test
must continue checking the full deterministic control payload, execution call
count and order, selected target IDs, and absence of tensors in serialized
results. Passing equivalent booleans is not sufficient evidence of behavior
preservation.

The refactor must pass the focused control and runner tests, the full test
suite, Ruff formatting, and Ruff linting.

## Non-Goals

This refactor does not:

- change any control algorithm;
- change output schema or naming;
- change thresholds or tolerances;
- add new controls;
- optimize GPU or CPU performance;
- replace dictionaries with a new public typed result model;
- create one module per control;
- impose a strict maximum line count;
- rewrite shared utilities under `jlens_reasoning/experiments_utils`.
