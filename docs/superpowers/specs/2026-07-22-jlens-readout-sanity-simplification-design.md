# J-Lens Readout Sanity Simplification Design

**Date:** 2026-07-22

## Purpose

Make the J-Lens readout sanity experiment easy to understand from its notebook
and main runner while preserving its scientific behavior, project-standard LLM
evaluation, reusable intervention mechanics, deterministic controls, and result
artifacts.

The final design should let a reader answer three questions without tracing
parallel collections or nested untyped dictionaries:

1. What cases does the experiment run?
2. Which operations apply to each case?
3. How is each output evaluated and compared?

## Design Principles

- Define each logical experiment case once.
- Declare case behavior explicitly; never infer behavior from a magic key.
- Use full generated-text evaluation for clean model capability.
- Use paper-style next-token ranks for interventions.
- Put all output-evaluation contracts and rank-evaluation helpers in
  `evaluation.py` and `evaluation_utils.py`.
- Keep policy-free intervention math in `experiments_utils` for reuse.
- Keep experiment thresholds, case selection, and control policy local to the
  J-Lens sanity experiment.
- Use typed results internally and convert to JSON-compatible dictionaries only
  at the artifact boundary.
- Give every public function and every function whose body spans 20 or more
  source lines a one-sentence docstring describing its responsibility.

## Approaches Considered

### Selected: one compositional case with optional operation specifications

A single `Case` owns the prompt, clean accepted answers, and optional readout
and intervention specifications. The presence of a specification tells the
runner which operation to perform.

This supports cases that need both readout and intervention work without a
second source of truth such as a `kind` field.

### Rejected: one case with a mutually exclusive kind enum

A `READOUT`, `SWAP`, or `COMPARE` enum makes dispatch simple but cannot naturally
represent the spider case, which needs both readout and swap operations. It
would either duplicate the prompt or add exceptions to the enum model.

### Rejected: separate case subclasses

Subclassing preserves static distinctions but recreates the fragmented case
model that made the experiment hard to read. The experiment has too few cases
and too little variant-specific behavior to justify a class hierarchy.

## Case Model

The notebook defines one ordered `CASES` collection. Each entry contains all
human-readable experimental inputs for that logical case.

```python
@dataclass(frozen=True, slots=True)
class ReadoutSpec:
    concepts: tuple[str, ...]
    literal_argument: str | None = None
    require_capability_gate: bool = False


@dataclass(frozen=True, slots=True)
class InterventionSpec:
    source_surface: str
    target_surface: str
    target_answers: tuple[str, ...]
    alphas: tuple[float, ...] = (1.0, 2.0)


@dataclass(frozen=True, slots=True)
class Case:
    key: str
    prompt: str
    expected_answers: tuple[str, ...]
    readout: ReadoutSpec | None = None
    intervention: InterventionSpec | None = None
```

`Case` is the only case class. `ReadoutSpec` and `InterventionSpec` are operation
configuration values, not alternative case identities.

The runner dispatches from specification presence:

```python
readout = run_readout(...) if case.readout is not None else None
intervention = run_intervention(...) if case.intervention is not None else None
```

No code checks for `case.key == "spider"`. The spider-specific capability gate
is declared by `require_capability_gate=True` in its `ReadoutSpec`.

Intervention comparisons are not a separate case operation. Every intervention
is evaluated against its clean next-token target rank. Negative controls compare
their configured candidate condition with the same real or clean reference
through the shared rank-comparison API.

## Evaluation Model

The project has two explicit evaluation modes.

### Generated-text evaluation

Every case performs one deterministic clean generation with sampling disabled
and a declared 64-token safety limit. The experiment builds a `ModelOutput`
containing raw text, token IDs, token pieces, generation status, finish reason,
and any generation error. It evaluates that output using
`SimpleFactualEvaluator(reasoning_parser=parse_think_tags)` and the case's
`expected_answers`.

This restores the normative project policy: preserve the full response, remove
declared reasoning without consulting the references, extract a front-loaded
answer, and then compare it with predefined references.

The clean generation evaluation is a capability gate. A next-token top-1 check
must not replace it.

### Next-token rank evaluation

Interventions retain the paper-style protocol. The experiment obtains logits at
the fixed formatting-adjusted scoring position for:

- the clean, non-intervened condition;
- every configured intervention strength;
- every negative-control condition.

These logits are evaluated through a public rank-evaluation API in
`evaluation.py`:

```python
@dataclass(frozen=True, slots=True)
class RankedToken:
    token_id: int
    token: str
    logit: float


@dataclass(frozen=True, slots=True)
class NextTokenEvaluation:
    accepted_references: tuple[str, ...]
    accepted_token_ids: tuple[int, ...]
    top1_id: int
    top1_token: str
    target_rank: int
    top_tokens: tuple[RankedToken, ...]


@dataclass(frozen=True, slots=True)
class RankComparison:
    baseline_rank: int
    candidate_rank: int
    rank_gain: int
    log_rank_gain: float
    improved: bool
    reached_top1: bool
```

The public functions are:

```python
def evaluate_next_token(...) -> NextTokenEvaluation:
    """Evaluate a next-token distribution against predefined accepted answers."""


def compare_token_ranks(...) -> RankComparison:
    """Compare a candidate target rank with its clean or real reference rank."""
```

`evaluation_utils.py` owns the reusable implementation helpers for:

- resolving accepted answer strings to deterministic single-token variants;
- ranking one or more accepted token IDs with deterministic tie handling;
- extracting the top-ranked tokens;
- computing absolute and logarithmic rank gains.

The evaluation layer reports measurements and pairwise comparisons. It does not
own experiment policy such as the required number of improved cases or control
percentiles.

Rank gains use the existing sign convention:

- `rank_gain = baseline_rank - candidate_rank`;
- `log_rank_gain = log(baseline_rank) - log(candidate_rank)`.

Positive values therefore mean improvement.

## Intervention Mechanics

Policy-free intervention mechanics remain under
`src/jlens_reasoning/experiments_utils/interventions.py` because future
experiments will reuse them. This module owns:

- J-Lens vector construction from a Jacobian and unembedding row;
- coordinate-swap tensor math;
- exception-safe model hook installation and removal;
- intervention execution;
- per-layer source and target vector construction.

The module must not contain J-Lens sanity case names, fixed alphas, pass
thresholds, control seeds, or reporting policy.

Token-position discovery and model/lens compatibility checks may remain in
other focused `experiments_utils` modules when they are genuinely reusable.

## Execution Flow

`run_case` is the primary human-readable execution unit:

```python
def run_case(case: Case, runtime: Runtime) -> CaseResult:
    """Run the clean evaluation and every operation configured for one case."""
    baseline = generate_and_evaluate(case, runtime)
    readout = run_readout(case, case.readout, runtime) if case.readout else None
    intervention = (
        run_intervention(case, case.intervention, runtime)
        if case.intervention
        else None
    )
    return CaseResult(case, baseline, readout, intervention)
```

`run_intervention` performs one linear sequence:

1. Prepare the fixed scoring input.
2. Resolve source, target, and accepted-answer token variants.
3. Evaluate clean target-answer logits with `evaluate_next_token`.
4. Construct reusable intervention vectors.
5. Execute each configured alpha.
6. Evaluate each intervened logits vector with `evaluate_next_token`.
7. Compare every intervention with clean using `compare_token_ranks`.

`run_experiment` validates the case collection, runs `run_case` once per case,
runs aggregate negative controls, applies experiment-specific capability gates,
and returns one typed `ExperimentResult`.

The runner must not create parallel readout and swap result lists, join them by
string key, or mutate a previously assembled baseline.

## Controls

This refactor preserves all four existing negative controls:

- identity;
- matched random vector;
- wrong concept;
- random target.

Controls reuse the same intervention execution and next-token evaluation APIs
as real swaps. Experiment-local code remains responsible for generating control
directions or targets and applying aggregate gates.

Control execution and analysis are colocated in one experiment-local
`controls.py`. A facade that re-exports private functions from execution and
analysis modules is not part of the target design.

## Results and Reporting

Typed internal results make the data flow discoverable:

- `CaseResult` contains clean generated-text evaluation, optional readout
  results, and optional intervention results.
- `InterventionResult` contains the clean target-rank evaluation and one result
  per alpha, including its `RankComparison`.
- `ExperimentResult` contains cases, controls, policy checks, failures, and
  provenance.

Dataclasses are converted recursively only when writing `result.json`.

The human report consumes typed summary fields rather than reconstructing
semantics from deeply nested dictionaries. It shows:

- clean generated answer and evaluation status;
- readout ranks for configured readout cases;
- clean and intervened target ranks;
- rank improvements and top-1 hits;
- aggregate control outcomes;
- overall failures.

Full per-layer and per-position diagnostics remain available in the JSON
artifact. The standard human report renders summaries only.

## Function Documentation

Every public function and each function whose body spans 20 or more source lines
receives a one-sentence docstring. The sentence describes the function's
responsibility or contract rather than repeating its name.

Examples:

```python
"""Generate and evaluate one case's clean visible response."""

"""Measure configured concepts across fitted layers and scored positions."""

"""Run deterministic negative controls and apply their aggregate gates."""
```

Small private formatting or arithmetic helpers do not require docstrings when
their names and types fully communicate their behavior.

## Validation and Error Handling

The experiment fails before inference when:

- case keys are empty or duplicated;
- a case has no readout or intervention operation;
- accepted answer collections are empty;
- configured swap surfaces are not single tokens;
- intervention alphas are empty or duplicated;
- model and lens dimensions or fitted layers are incompatible;
- no fitted layer falls inside the configured workspace.

Generation errors become `ModelOutput` and `EvaluationResult` data rather than
discarding the raw failure. Invalid experiment configuration raises a clear
exception because no meaningful run can proceed.

## Testing Strategy

Tests focus on behavior and stable contracts rather than module facades or exact
incidental dictionary nesting.

1. Add a regression test proving every configured case performs clean full-text
   generation through `SimpleFactualEvaluator`.
2. Remove the notebook assertion that `SimpleFactualEvaluator` and generation
   must be absent.
3. Extend evaluation tests for answer-token resolution, deterministic ranks,
   top-token extraction, ties, and clean-versus-candidate comparisons.
4. Parameterize case-dispatch tests for rejecting operation-free cases and for
   readout-only, intervention-only, and combined readout-plus-intervention
   behavior.
5. Retain focused tensor tests for coordinate-swap math, dtype preservation,
   tuple outputs, and hook cleanup after success or failure.
6. Test that real swaps and controls call the same next-token evaluator.
7. Test aggregate experiment gates from small typed fixtures.
8. Keep a notebook contract test showing the case table, pinned artifacts,
   generation callback, experiment invocation, result serialization, and final
   failure gate.

## Target Package Shape

```text
experiments/jlens_readout_sanity/
├── constants.py
├── experiment.py       # spec types, typed results, run_case, run_experiment
├── controls.py         # experiment-specific negative-control policy
├── reporting.py        # compact human report
└── jlens_readout_sanity.ipynb # the concrete ordered CASES collection

src/jlens_reasoning/
├── evaluation.py       # text and next-token public evaluation contracts
├── evaluation_utils.py # text and rank evaluation primitives
└── experiments_utils/
    ├── artifacts.py
    ├── interventions.py
    ├── tokens.py        # experiment token-position utilities only
    └── validation.py
```

## Migration Sequence

1. Add failing regression tests for clean generation evaluation.
2. Add typed next-token evaluation and comparison APIs.
3. Replace parallel readout and swap cases with the unified case collection.
4. Introduce typed case and intervention results.
5. Rewrite the runner around `run_case` without post-assembly mutation.
6. Move controls onto the shared intervention and rank-evaluation path.
7. Simplify reporting and JSON conversion.
8. Remove obsolete facades, duplicated rank helpers, and tests for superseded
   structures only after behavioral equivalence is verified.

This migration preserves the current scientific thresholds and raw diagnostics.
