# Experiment Package Architecture Design

**Date:** 2026-07-21

## Purpose

Reorganize the repository so each experiment is a self-contained, importable
package beside its Colab notebook, while reusable experiment mechanics live in
one focused `jlens_reasoning.experiments_utils` package.

The design should make new experiments additive. A new experiment adds a new
package, notebook, and mirrored tests without modifying an experiment registry,
an existing experiment package, or a central notebook list. The layout should
also keep notebooks readable and independently debuggable rather than hiding
the complete workflow behind one opaque function call.

## Goals

- Replace `jlens_reasoning.experiments` with
  `jlens_reasoning.experiments_utils` through a clean migration.
- Place each experiment in an importable top-level package under
  `experiments/<experiment_name>/`.
- Keep generic experimental mechanics reusable across future experiments.
- Keep experiment policy, result assembly, and notebook reporting local to the
  owning experiment.
- Preserve the existing J-Lens readout sanity experiment's prompts,
  calculations, controls, artifact names, result schema, gates, and observable
  notebook behavior.
- Keep the canonical bootstrap and environment-check notebooks in the shared
  top-level `notebooks/` directory.
- Make notebook discovery automatic so adding an experiment does not require
  editing a central test list.

## Non-Goals

- No experiment registry, plugin protocol, controller, or central discovery
  runtime.
- No backwards-compatibility wrappers for the old
  `jlens_reasoning.experiments.*` imports.
- No change to the scientific meaning or thresholds of the current J-Lens
  sanity experiment.
- No requirement that every experiment contain only three files. Additional
  focused modules are allowed when they keep responsibilities clear.
- No attempt to generalize policy that is specific to one experiment.

## Target Repository Layout

```text
src/jlens_reasoning/
  experiments_utils/
    __init__.py
    artifacts.py
    controls.py
    interventions.py
    tokens.py
    validation.py

experiments/
  __init__.py
  jlens_readout_sanity/
    __init__.py
    constants.py
    types.py
    utils.py
    runner.py
    controls.py
    jlens_readout_sanity.ipynb

notebooks/
  _template.ipynb
  00_environment_check.ipynb

tests/
  experiments_utils/
  experiments/
    jlens_readout_sanity/
```

Both `experiments/` and every experiment directory are Python packages. Code
and notebooks use absolute imports such as:

```python
from experiments.jlens_readout_sanity.constants import MODEL_NAME
from experiments.jlens_readout_sanity.utils import run_readout_sanity
from jlens_reasoning.experiments_utils.artifacts import write_results
```

Notebook-local imports such as `from utils import ...` are not used because
their behavior depends on the active working directory.

## Packaging and Importability

The current setuptools configuration discovers packages only below `src/`.
The migration must update package discovery so the built wheel contains both
package families:

- `jlens_reasoning*` discovered below `src/`;
- `experiments*` discovered from the repository root.

Discovery uses explicit include filters so scanning the repository root cannot
accidentally package tests, documentation, or unrelated directories. It remains
pattern-based rather than listing individual experiment package names; adding
`experiments/<new_experiment>/` therefore does not require changing
`pyproject.toml`.

The Colab bootstrap continues to build and install the project wheel. Notebook
imports must succeed from that installed wheel and must not rely on the current
working directory, an ad hoc `sys.path` insertion, or importing a sibling file
by its bare module name. The notebook itself remains in the cloned repository
for interactive execution and is not required to be wheel package data.

## Shared Utility Boundaries

The shared package owns algorithms and mechanics that are useful to more than
one plausible experiment. Shared functions receive their policy as arguments;
they do not import constants from `jlens_readout_sanity`.

### `artifacts.py`

- Convert tensors, dataclasses, paths, mappings, and sequences into stable
  JSON-ready values.
- Write sorted, deterministic result JSON.
- Contain no experiment-specific result schema or artifact path.

### `tokens.py`

- Generate concept surface variants.
- Resolve strict single-token surfaces and accepted token variants.
- Find token subsequences and positions relative to prompt literals.
- Calculate stable one-based target ranks.
- Build top-token and next-token payloads.
- Prepare formatting-adjusted scoring inputs.

These helpers accept tokenizer-like objects and do not depend on fixed prompts,
answers, or J-Lens sanity case names.

### `interventions.py`

- Construct token-backed J-Lens vectors.
- Apply coordinate swaps.
- Own hook registration and guaranteed hook cleanup.
- Execute an intervention for supplied model blocks, inputs, vectors, and
  strength.

The shared layer knows how to perform an intervention, but it does not know
which source and target concepts an experiment chooses.

### `controls.py`

- Calculate natural-log rank gain and arithmetic means.
- Calculate deterministic interpolated percentiles and strict percentile
  comparisons.
- Derive stable random sub-seeds.
- Generate norm-matched random vectors.
- Build token-ID exclusion categories.
- Select deterministic random target IDs.

Every seed list, quantile, tolerance, expected case set, and gate requirement is
provided by the caller or stored in the experiment's constants. This module
does not contain names such as `wrong_concept_control` or assumptions about the
five J-Lens cases.

### `validation.py`

- Validate model/lens residual width and fitted-layer compatibility.
- Select workspace layers from caller-provided bounds.
- Calculate workspace loading for caller-provided layers and positions.

## J-Lens Experiment Boundaries

The J-Lens package owns all fixed scientific policy and the complete result
contract for this experiment.

### `constants.py`

Contains immutable experiment configuration:

- model and lens artifact coordinates;
- readout and swap prompts, concepts, and accepted answers;
- default intervention strengths and workspace bounds;
- control seeds, tolerances, percentile quantile, and required case wins;
- expected ordered case keys and the mapping from controls to global checks;
- artifact names and reporting labels.

It contains no model execution, tensor operations, file writing, or control
orchestration.

### `types.py`

Contains the experiment-specific dataclasses for readout cases, swap cases,
resolved cases, token variants when they carry experiment meaning, and prepared
intervention contexts. Generic tokenizer or intervention types remain in the
shared utilities when appropriate.

### `runner.py`

Contains the model-independent experiment flow:

- analyze readout cases;
- prepare per-case scoring and intervention contexts;
- execute the real swaps at configured strengths;
- invoke experiment-specific negative controls;
- assemble existing capability checks and control checks;
- return the complete result dictionary.

`run_readout_sanity` accepts already-loaded model, lens, tokenizer,
unembedding weights, and a next-token forward callable. It performs no Colab
initialization and chooses no output directory, so CPU tests can exercise the
full flow with stubs.

### `controls.py`

Contains policy and orchestration for the four required controls:

- identity;
- matched random-vector;
- wrong concept;
- random target.

It validates the exact ordered five-case set, assembles per-control result
payloads, formats experiment-specific failures, and derives the control-only
and global pass states. It delegates generic calculations and random generation
to `jlens_reasoning.experiments_utils.controls`.

### `utils.py`

Provides the small, stable public surface used by the notebook. It re-exports
or wraps only the functions needed for notebook orchestration, such as:

- `run_readout_sanity`;
- model/lens validation;
- result writing;
- concise summary formatting;
- visualization inputs required by the notebook.

It is not a second implementation of the runner and must not become a catch-all
module.

## Notebook Design

The experiment notebook moves to:

```text
experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb
```

It retains the canonical first bootstrap cell from `notebooks/_template.ipynb`.
The notebook remains explicit enough that a Colab user can rerun and debug each
expensive or failure-prone stage independently.

The cells perform these responsibilities in order:

1. Run the canonical project bootstrap.
2. Initialize the Colab environment with CUDA required.
3. Import experiment constants and utilities.
4. Load the causal language model, tokenizer, wrapped model, and J-Lens
   artifact.
5. Validate model/lens compatibility.
6. Define the small next-token forward adapter.
7. Run `run_readout_sanity`.
8. Attach provenance and write `result.json` to the existing run directory.
9. Print concise readout, swap, and control summaries.
10. Render and save the same selected HTML visualizations.
11. Raise the existing final failure when the global result does not pass.

The notebook does not implement rank calculations, tensor transforms, control
generation, gate aggregation, or result schemas. Conversely, those modules do
not hide model downloads, device placement, artifact paths, visualization, or
the final notebook failure behind a single `run_experiment(context)` call.

The shared `notebooks/_template.ipynb` and
`notebooks/00_environment_check.ipynb` remain in place.

## Data Flow

```text
Notebook setup and model loading
  -> experiment runner with loaded dependencies
  -> shared token/intervention/control mechanics
  -> experiment-specific control and gate assembly
  -> complete result dictionary
  -> notebook provenance, serialization, reporting, visualization
  -> final hard-gate exception when result["passed"] is false
```

The runner and control modules retain only JSON-ready summaries for repeated
control executions. Large logits remain transient and are consumed immediately
for ranks, top-1 IDs, and diagnostic values.

## Open/Closed Extension Rule

Adding a new experiment requires only:

```text
experiments/<new_experiment>/
tests/experiments/<new_experiment>/
```

The new package defines its own constants, utilities, runner modules as needed,
and notebook. It imports shared mechanics from `experiments_utils` but does not
modify the J-Lens package.

There is no experiment registry and no central experiment list. Shared
utilities should change only when a genuinely reusable primitive is missing,
not merely because a new experiment has new policy.

## Clean Migration

- Delete `src/jlens_reasoning/experiments` after every owned symbol has moved.
- Update all imports to the new shared or experiment-local package paths.
- Do not leave compatibility modules or re-export wrappers at the old paths.
- Move, rather than copy, the J-Lens notebook.
- Update README references to the new notebook location while preserving the
  pre-existing unrelated README change.
- Preserve public behavior and serialized field names unless a path or import
  necessarily changes under this design.

## Error Handling

- Validate configured cases and strict token surfaces before lens forwards.
- Validate model/lens dimensions and workspace layers before experiment work.
- Guarantee intervention hook cleanup when a forward raises.
- Reject malformed or partial control case sets rather than computing partial
  aggregates.
- Reject empty random-target candidate pools and out-of-output-vocabulary IDs.
- Preserve actionable control failure messages with observed and required
  values.
- Write the result artifact before the notebook enforces the final hard gate,
  preserving diagnostics for failed runs.

## Testing

### Shared utility tests

Tests under `tests/experiments_utils/` cover generic behavior independently of
the J-Lens experiment:

- token variants, ranks, formatting prefixes, and token spans;
- coordinate swap math, all-position hooks, and cleanup on exceptions;
- deterministic rank gain, means, percentile interpolation, and strict gates;
- random-vector determinism, norm matching, device/dtype restoration, and
  zero-norm behavior;
- token exclusions, deterministic target selection, and output-vocabulary
  bounds;
- stable JSON serialization;
- model/lens and workspace validation.

### Experiment tests

Tests under `tests/experiments/jlens_readout_sanity/` cover policy and complete
orchestration:

- exact prompts, cases, artifact coordinates, thresholds, and seeds;
- real readout and swap result compatibility;
- the four negative-control configurations and gates;
- exact five-case enforcement and actionable failures;
- complete CPU-only integration with model, lens, and tokenizer stubs;
- stable result schema and no retained logits;
- concise notebook reporting inputs.

### Notebook contract tests

Notebook tests keep the two shared notebooks explicit and discover experiment
notebooks with a path pattern such as `experiments/*/*.ipynb`. They verify:

- no saved outputs or execution counts;
- the canonical bootstrap cell matches `_template.ipynb`;
- Colab initialization and CUDA requirements are present where required;
- model loading, runner invocation, result writing, visualization, and the
  final failure gate remain visible;
- credentials and generation-based evaluation do not appear.

Adding a new experiment notebook therefore does not require editing a central
notebook list.

## Acceptance Criteria

- `src/jlens_reasoning/experiments` no longer exists.
- `jlens_reasoning.experiments_utils` contains only reusable mechanics with
  caller-provided policy.
- `experiments.jlens_readout_sanity` is importable and owns the current
  experiment's policy and orchestration.
- The built wheel includes both `jlens_reasoning.experiments_utils` and
  `experiments.jlens_readout_sanity`; imports do not depend on the notebook's
  working directory.
- The J-Lens notebook lives beside its experiment code and follows the explicit
  debuggable cell flow above.
- The template and environment-check notebooks remain under `notebooks/`.
- No registry, controller, or old-import compatibility layer is introduced.
- Existing J-Lens scientific behavior and artifact schema remain unchanged.
- Tests mirror the shared and experiment-local package boundaries.
- Experiment notebook tests use discovery rather than a manually maintained
  central list.
- The full test suite, Ruff checks, formatting checks, and patch whitespace
  checks pass.
