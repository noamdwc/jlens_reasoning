# Notebook-Owned Experiment Cases Design

## Goal

Make the J-Lens Colab notebook the single, visible source of truth for its
readout and swap cases. A reader should be able to understand every prompt,
expected answer, concept, and intervention target without opening a Python
module.

The migration must not duplicate case definitions, add a registry, or change
the experiment's model coordinates, control policy, result schema, or
scientific gates.

## Ownership Boundary

The notebook owns two typed tuples in a dedicated `define-cases` cell:

- `READOUT_CASES`, constructed from `ReadoutCase`;
- `SWAP_CASES`, constructed from `SwapCase`.

The cell imports both dataclasses from
`experiments.jlens_readout_sanity.types`. It contains data declarations only,
so it is fast and safe to rerun while debugging.

`experiments/jlens_readout_sanity/constants.py` retains fixed experiment
policy:

- model and lens coordinates;
- workspace bounds;
- intervention strengths;
- control seeds, thresholds, tolerances, and namespaces;
- `CONTROL_REQUIRED_CASE_COUNT = 5` and the required wrong-concept wins.

It no longer imports case dataclasses or defines `READOUT_CASES`, `SWAP_CASES`,
or `CONTROL_CASE_KEYS`.

## Runner API and Data Flow

`run_readout_sanity` requires `cases` and `swap_cases` as keyword arguments.
There are no module-owned defaults. The notebook's `run-experiment` cell
passes its two tuples explicitly.

Before any lens forward, the runner validates:

1. the configured number of readout and swap cases matches the fixed required
   control-case count of five;
2. readout keys are unique;
3. swap keys are unique;
4. the two tuples contain the same keys in the same order;
5. every configured source and target swap surface encodes as exactly one
   token;
6. the required control alpha remains among the requested strengths.

The existing resolved-case objects then carry all case data into analysis and
control execution. No downstream component looks up notebook data through a
registry or imports it from another module.

## Derived Control Cases

Control metadata is derived from the resolved cases passed by the notebook:

- expected case keys come from the ordered resolved swap cases;
- source and target exclusion surfaces come from their swap definitions;
- clean-answer exclusion surfaces come from their paired readout definitions;
- intended-answer exclusion surfaces come from their swap target answers;
- formatting exclusions continue to come from the prepared contexts.

Wrong-concept references are also derived. A swap direction is the resolved
`(source.token_id, target.token_id)` pair. For each context, controls select the
first context in notebook order whose direction differs. With the current
notebook ordering, this preserves the existing pairing: spider uses the first
France/China case, and every France/China case uses spider/ant.

Control execution raises a clear `ValueError` before its expensive loops if it
does not receive exactly five ordered cases or if the supplied cases do not
contain at least two distinct swap directions.

The fixed policy remains unchanged: sixteen deterministic seeds, alpha 1,
strict 95th-percentile comparisons, four required wrong-concept wins, and the
current numerical tolerances.

## Notebook Layout

The notebook keeps its existing explicit workflow and inserts `define-cases`
after environment initialization and before model/lens loading. The relevant
order is:

1. Colab bootstrap;
2. environment initialization;
3. case definitions;
4. model and lens loading;
5. experiment execution;
6. result saving;
7. reporting;
8. visualization;
9. final gate.

The model-loading cell imports only artifact coordinates from `constants.py`
and notebook helpers from `utils.py`. The case-definition cell imports the two
dataclasses. Reporting and visualization continue to use the notebook-owned
tuples and the generated result.

## Testing

Notebook contract tests execute only the `define-cases` cell and verify the
exact five current readout and swap cases. This tests the notebook as the
authoritative definition without loading a model.

Runner tests verify that:

- `cases` and `swap_cases` are required;
- missing, duplicate, reordered, and mismatched cases fail before forwards;
- strict swap-surface validation remains early;
- the complete five-case integration preserves the current result schema and
  intervention count.

Control tests use synthetic, renamed keys and two distinct swap directions to
prove that expected keys, exclusions, and wrong-concept references are derived
rather than tied to `spider` or `france_capital` literals.

Constants tests verify the absence of notebook-owned cases and
`CONTROL_CASE_KEYS`, while retaining exact assertions for artifact coordinates,
seeds, thresholds, tolerances, and control mappings.

Final verification includes the focused notebook, runner, control, constants,
and package tests; the full pytest suite; Ruff lint and formatting; and the
installed-wheel import check for both package roots.

## Acceptance Criteria

- The notebook visibly defines both case tuples in one rerunnable cell.
- No production Python module duplicates those tuples.
- `run_readout_sanity` receives both tuples explicitly.
- Control keys, exclusions, and mismatch references derive from supplied
  cases.
- Exactly five cases and at least two distinct swap directions are validated
  before expensive control execution.
- Current prompts, answers, swaps, thresholds, result fields, and pass/fail
  behavior remain unchanged.
- The experiment remains registry-free and packaged under
  `experiments.jlens_readout_sanity`.
