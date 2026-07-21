# Readout Constants and Random-Target Caching Design

## Goal

Improve readability inside `src/jlens_reasoning/experiments/` by giving
experiment constants focused ownership, while fixing only the previously
identified P2 performance issue in random-target control vector construction.
All current import paths remain compatible through explicit re-exports.

## Scope

This change is limited to the experiment package and its tests. It does not
apply the separate P3 suggestion to broaden the existing facade export test,
change control definitions, alter thresholds, modify case data, or change the
result schema.

The existing uncommitted `README.md` change remains untouched and outside all
commits for this work.

## Constants Architecture

Create two focused modules:

- `readout_constants.py` owns model and lens artifact coordinates plus readout
  policy defaults such as top-k size, workspace-layer bounds, default
  intervention strengths, and the minimum required swap improvements.
- `sanity_constants.py` owns negative-control seeds, exact case keys,
  tolerances, percentile settings, the fixed control alpha, the required
  wrong-concept case wins, low-precision norm tolerances, and the maximum
  random-vector generation attempts.

Structured experiment fixtures remain in `readout_cases.py`. In particular,
`READOUT_CASES` and `SWAP_CASES` are cohesive case configuration and should not
move into a generic constants module.

Schema keys such as `"passed"`, `"cases"`, and `"target_rank"`, along with
one-off error and reporting strings, remain beside the logic that consumes
them. Turning those values into constants would make the data assembly harder
to read without creating a useful policy boundary.

## Compatibility

Existing consumers continue to work:

- `readout_sanity` explicitly re-exports its current model/lens constants and
  `TOP_K` from `readout_constants`.
- `readout_utils` re-exports `TOP_K` while importing workspace bounds from
  `readout_constants`.
- `sanity_controls` explicitly re-exports its current control constants from
  `sanity_constants`.

Internal modules import constants from their owning module rather than through
a compatibility facade. This keeps dependencies clear and prevents accidental
cycles.

## P2 Random-Target Optimization

The current random-target loop computes both source and target J-Lens vectors
inside the five-context loop for every selected target. The target vector
depends only on the selected token and layer, while each prepared context
already stores its real source vector.

Add a private single-token vector helper in `intervention_utils.py`. The
existing pair-building helper composes two calls to it, preserving its current
behavior. For each selected random target, `readout_controls.py` calls the
single-token helper once across the workspace layers, then pairs those cached
target vectors with each context's existing source vectors.

The resulting intervention vectors are numerically identical to the current
implementation. The loop order, selected targets, forward-pass count, scoring
inputs, result payload, and intentionally fixed control behavior do not change.
Only redundant Jacobian matrix-vector products are removed.

## Testing

Follow test-driven development:

1. Extend the integration test to count J-Lens vector construction and prove
   the current implementation performs redundant work.
2. Implement the single-token helper and cached target-vector assembly until
   the count demonstrates one target-vector construction per selected target
   and layer, with source vectors reused from contexts.
3. Add structural tests that import both constants modules and verify legacy
   constant aliases resolve to the exact same objects or values.
4. Run focused readout/control tests, notebook tests, the full test suite, Ruff
   lint and formatting checks, and `git diff --check`.

## Success Criteria

- Random-target target vectors are built once per selected target and layer,
  not once per selected target, context, and layer.
- Existing source vectors are reused from each `InterventionContext`.
- Results and control schemas are unchanged.
- Constants have focused ownership in two modules.
- Existing import paths continue to work through explicit re-exports.
- The P3 facade-test expansion is not implemented.
- All verification commands pass.
