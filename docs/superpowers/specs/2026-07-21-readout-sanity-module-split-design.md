# Readout Sanity Module Split Design

**Date:** 2026-07-21

## Purpose

Fix the random-target reserved-token exclusion defect and split the 1,316-line
`readout_sanity.py` module into focused components without changing experiment
behavior, result schemas, public imports, or the reviewed P2 random-target
vector construction path.

## Scope

The behavioral change is limited to random-target exclusions: every token ID
present in `tokenizer.added_tokens_decoder` is treated as a reserved/control
token, including entries whose `AddedToken.special` flag is false. A regression
test covers a Qwen-style control token with `special=False`.

The structural change reorganizes existing code only. All prompts, cases,
thresholds, gates, intervention strengths, scoring rules, hook behavior,
result fields, and notebook imports remain unchanged. The P2 recommendation to
cache random-target J-Lens vectors is explicitly out of scope.

## Module Boundaries

### `readout_sanity.py`

Remain the stable public facade. It owns the top-level experiment orchestration
and readout analysis, and re-exports names that callers and tests currently
import from this module.

### `readout_cases.py`

Own the readout/swap case dataclasses, fixed case definitions, token variants,
surface expansion, and strict case resolution. This keeps experiment inputs
and their tokenization contract together.

### `readout_utils.py`

Own stateless shared utilities: rank and top-token calculation, formatting
prefix preparation, workspace calculations, model/lens validation, capability
check aggregation, and stable JSON serialization.

### `intervention_utils.py`

Own intervention-specific types and mechanics: J-Lens vector construction,
coordinate swapping, hook registration/cleanup, shared intervention execution,
identity analysis, intervention-context preparation, swap summarization, and
swap-case execution.

### `readout_controls.py`

Own negative-control orchestration and result assembly. It composes the pure
calculations in `sanity_controls.py` with the shared intervention functions.
The existing random-target inner loop is moved verbatim so P2 behavior is not
changed in this task.

### `sanity_controls.py`

Retain deterministic control calculations. Update reserved-token exclusion to
include every added-token ID, regardless of its `special` flag.

## Dependency Direction

`readout_cases.py` and `readout_utils.py` are foundational and do not import
the facade. `intervention_utils.py` depends on those modules.
`readout_controls.py` depends on cases, utilities, interventions, and
`sanity_controls.py`. `readout_sanity.py` imports and re-exports the public
surface while coordinating the experiment. No extracted module imports
`readout_sanity.py`, preventing circular dependencies.

## Compatibility

Existing imports from `jlens_reasoning.experiments.readout_sanity` remain
valid through explicit imports/re-exports in that module. Private helpers may
move, but tests that intentionally exercise them keep working through the
facade. Serialized output remains byte-compatible except that
`controls.random_target.exclusions.reserved_special` and `all` now correctly
contain non-special added control-token IDs.

## Testing

Use test-driven development for the behavioral fix: first add a tokenizer
fixture entry with `special=False`, assert its ID is excluded, and observe the
test fail. Then implement the minimal exclusion change and observe it pass.

For the refactor, preserve the existing test suite as the behavior contract.
After each extraction, run the focused readout/control tests. Final validation
runs the full test suite, Ruff lint, Ruff formatting, and `git diff --check`.
Tests also verify representative public imports continue to resolve from
`readout_sanity.py`.

## Success Criteria

- Qwen-style non-special added control tokens cannot be selected as random
  targets.
- `readout_sanity.py` becomes a readable facade rather than a monolith.
- Extracted files each have one cohesive responsibility and no import cycle.
- Existing public imports and experiment behavior remain compatible.
- P2 remains unmodified.
- All tests and repository checks pass.
