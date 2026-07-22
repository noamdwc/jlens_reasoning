# J-Lens Readout Sanity Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fragmented J-Lens sanity runner with one declarative case model and a linear per-case execution path while restoring full-output baseline evaluation and preserving paper-style next-token intervention metrics.

**Architecture:** Add typed next-token evaluation beside the existing factual evaluator, then build a new `experiment.py` path around `Case`, `ReadoutSpec`, and `InterventionSpec`. Migrate controls, reporting, and the notebook only after the new path is tested; reusable intervention tensor mechanics remain in `experiments_utils`, while evaluation policy and rank helpers move to `evaluation.py` and `evaluation_utils.py`.

**Tech Stack:** Python 3.11, dataclasses, PyTorch, Transformers, J-Lens, nbformat, pytest, ruff.

---

## Working-tree constraint

`README.md` already contains an unrelated user modification. Do not stage,
restore, or edit it during this implementation. Every commit command below
names its files explicitly.

## Target file ownership

- `src/jlens_reasoning/evaluation.py`: public generated-text and next-token
  evaluation contracts.
- `src/jlens_reasoning/evaluation_utils.py`: text extraction plus policy-free
  answer-token resolution and rank arithmetic.
- `src/jlens_reasoning/experiments_utils/interventions.py`: reusable J-Lens
  vector, hook, and coordinate-swap mechanics; behavior stays unchanged.
- `src/jlens_reasoning/experiments_utils/artifacts.py`: recursive serialization
  for typed experiment results and existing mapping artifacts.
- `src/jlens_reasoning/experiments_utils/tokens.py`: prompt-position and
  formatting-input helpers only.
- `src/jlens_reasoning/experiments_utils/controls.py`: reusable random-control
  generation and percentile helpers only.
- `experiments/jlens_readout_sanity/experiment.py`: case specs, typed results,
  case execution, and experiment aggregation.
- `experiments/jlens_readout_sanity/controls.py`: all experiment-specific
  negative-control execution and analysis.
- `experiments/jlens_readout_sanity/reporting.py`: compact typed-result report.
- `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`: concrete case
  declarations, model callbacks, execution, serialization, and report cells.

### Task 1: Add project-standard next-token evaluation

**Files:**
- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `src/jlens_reasoning/evaluation_utils.py`
- Modify: `tests/test_evaluation.py`
- Modify: `tests/experiments_utils/test_tokens.py`
- Modify: `tests/experiments_utils/test_controls.py`

- [ ] **Step 1: Add failing public-API tests**

Append imports for `NextTokenEvaluation`, `RankComparison`, `RankedToken`,
`compare_token_ranks`, and `evaluate_next_token` to `tests/test_evaluation.py`,
then add these tests:

```python
class RankTokenizer:
    pieces = {
        "Paris": [2],
        " Paris": [4],
        "paris": [2],
        " paris": [4],
        "PARIS": [8, 9],
        " PARIS": [8, 9],
    }

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self.pieces.get(text, [10, 11])

    def decode(
        self, token_ids: list[int], *, clean_up_tokenization_spaces: bool = False
    ) -> str:
        assert clean_up_tokenization_spaces is False
        return {0: "zero", 1: "one", 2: "Paris", 3: "three", 4: " Paris"}.get(
            token_ids[0], f"token-{token_ids[0]}"
        )


def test_next_token_evaluation_resolves_variants_and_stable_ranks() -> None:
    result = evaluate_next_token(
        torch.tensor([0.0, 4.0, 3.0, 1.0, 2.0]),
        ("Paris",),
        RankTokenizer(),
        top_k=3,
    )

    assert result.accepted_references == ("Paris",)
    assert result.accepted_token_ids == (2, 4)
    assert result.target_rank == 2
    assert result.top1_id == 1
    assert result.top1_token == "one"
    assert result.top_tokens == (
        RankedToken(1, "one", 4.0),
        RankedToken(2, "Paris", 3.0),
        RankedToken(4, " Paris", 2.0),
    )


def test_rank_comparison_uses_positive_improvement_convention() -> None:
    baseline = NextTokenEvaluation(
        accepted_references=("Paris",),
        accepted_token_ids=(2,),
        top1_id=1,
        top1_token="one",
        target_rank=10,
        top_tokens=(),
    )
    candidate = replace(
        baseline,
        top1_id=2,
        top1_token="Paris",
        target_rank=1,
    )

    comparison = compare_token_ranks(baseline, candidate)

    assert comparison == RankComparison(
        baseline_rank=10,
        candidate_rank=1,
        rank_gain=9,
        log_rank_gain=pytest.approx(math.log(10)),
        improved=True,
        reached_top1=True,
    )


@pytest.mark.parametrize("references", [(), ("",), ("two tokens",)])
def test_next_token_evaluation_rejects_unscorable_answers(
    references: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="accepted|single-token"):
        evaluate_next_token(
            torch.tensor([0.0, 1.0, 2.0]),
            references,
            RankTokenizer(),
        )
```

Add `import math`, `import torch`, and `replace` where required.

- [ ] **Step 2: Run the new tests and verify the API is absent**

Run:

```bash
uv run pytest tests/test_evaluation.py -k 'next_token or rank_comparison' -v
```

Expected: collection fails because the new evaluation symbols cannot be
imported.

- [ ] **Step 3: Implement rank helpers in `evaluation_utils.py`**

Move the behavior of `best_target_rank` and `top_tokens` out of
`experiments_utils/tokens.py`, and move `log_rank_gain` out of
`experiments_utils/controls.py`. Implement these exact public helpers:

```python
def answer_token_variants(
    tokenizer: object,
    accepted_references: Sequence[str],
) -> tuple[tuple[int, str], ...]:
    """Resolve accepted answers to unique case-and-space single-token variants."""
    variants: list[tuple[int, str]] = []
    seen_ids: set[int] = set()
    for reference in accepted_references:
        if not isinstance(reference, str) or not normalize_text(reference):
            raise ValueError("Accepted references must normalize to non-empty text")
        bases = (
            reference,
            reference.lower(),
            reference.capitalize(),
            reference.upper(),
        )
        for base in dict.fromkeys(bases):
            for surface in (base, f" {base}"):
                token_ids = tokenizer.encode(surface, add_special_tokens=False)
                if len(token_ids) == 1 and int(token_ids[0]) not in seen_ids:
                    token_id = int(token_ids[0])
                    seen_ids.add(token_id)
                    variants.append((token_id, surface))
    if not variants:
        raise ValueError("Accepted references have no single-token variants")
    return tuple(variants)


def best_token_rank(logits: torch.Tensor, target_ids: Sequence[int]) -> int:
    """Return the best deterministic one-based rank among target token IDs."""
    if logits.ndim != 1:
        raise ValueError("best_token_rank expects one logits vector")
    if not target_ids:
        raise ValueError("best_token_rank needs at least one target token")
    token_ids = torch.arange(logits.numel(), device=logits.device)
    ranks = []
    for target_id in target_ids:
        target_logit = logits[target_id]
        higher = (logits > target_logit).sum()
        earlier_ties = ((logits == target_logit) & (token_ids < target_id)).sum()
        ranks.append(1 + int(higher.item()) + int(earlier_ties.item()))
    return min(ranks)


def top_token_values(
    logits: torch.Tensor,
    tokenizer: object,
    *,
    k: int,
) -> tuple[tuple[int, str, float], ...]:
    """Return deterministic top-token IDs, decoded surfaces, and logits."""
    if k < 0:
        raise ValueError("top-token count must be non-negative")
    values, indices = torch.topk(logits, k=min(k, logits.numel()))
    return tuple(
        (
            int(token_id),
            tokenizer.decode(
                [int(token_id)],
                clean_up_tokenization_spaces=False,
            ),
            float(value),
        )
        for value, token_id in zip(values.tolist(), indices.tolist(), strict=True)
    )


def log_rank_gain(baseline_rank: int, candidate_rank: int) -> float:
    """Return positive natural-log gain when a candidate rank improves."""
    if baseline_rank < 1 or candidate_rank < 1:
        raise ValueError("Ranks must be positive one-based integers")
    return math.log(baseline_rank) - math.log(candidate_rank)
```

Import `math`, `Sequence`, `torch`, and the required typing names directly in
this module.

- [ ] **Step 4: Implement typed next-token evaluation in `evaluation.py`**

Add these frozen contracts:

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

    def __post_init__(self) -> None:
        """Reject incomplete or invalid next-token evaluation records."""
        if not self.accepted_references or not self.accepted_token_ids:
            raise ValueError("Next-token evaluations require accepted answers")
        if self.target_rank < 1:
            raise ValueError("Target rank must be a positive one-based integer")


@dataclass(frozen=True, slots=True)
class RankComparison:
    baseline_rank: int
    candidate_rank: int
    rank_gain: int
    log_rank_gain: float
    improved: bool
    reached_top1: bool
```

Then implement:

```python
def evaluate_next_token(
    logits: torch.Tensor,
    accepted_references: str | Sequence[str],
    tokenizer: object,
    *,
    top_k: int = 25,
) -> NextTokenEvaluation:
    """Evaluate a next-token distribution against predefined accepted answers."""
    references = (
        (accepted_references,)
        if isinstance(accepted_references, str)
        else tuple(accepted_references)
    )
    normalized_logits = logits.detach().float().cpu()
    variants = answer_token_variants(tokenizer, references)
    accepted_ids = tuple(token_id for token_id, _ in variants)
    top1_id = int(normalized_logits.argmax().item())
    return NextTokenEvaluation(
        accepted_references=references,
        accepted_token_ids=accepted_ids,
        top1_id=top1_id,
        top1_token=tokenizer.decode(
            [top1_id], clean_up_tokenization_spaces=False
        ),
        target_rank=best_token_rank(normalized_logits, accepted_ids),
        top_tokens=tuple(
            RankedToken(token_id, token, logit)
            for token_id, token, logit in top_token_values(
                normalized_logits, tokenizer, k=top_k
            )
        ),
    )


def compare_token_ranks(
    baseline: NextTokenEvaluation,
    candidate: NextTokenEvaluation,
) -> RankComparison:
    """Compare a candidate target rank with its clean or real reference rank."""
    if baseline.accepted_token_ids != candidate.accepted_token_ids:
        raise ValueError("Rank comparisons require the same accepted token IDs")
    return RankComparison(
        baseline_rank=baseline.target_rank,
        candidate_rank=candidate.target_rank,
        rank_gain=baseline.target_rank - candidate.target_rank,
        log_rank_gain=log_rank_gain(
            baseline.target_rank,
            candidate.target_rank,
        ),
        improved=candidate.target_rank < baseline.target_rank,
        reached_top1=candidate.target_rank == 1,
    )
```

- [ ] **Step 5: Update callers and remove duplicate helper tests**

Until the experiment cutover, preserve import compatibility by making
`experiments_utils.tokens.best_target_rank` and `top_tokens` thin aliases around
the new evaluation helpers, and make `experiments_utils.controls.log_rank_gain`
an imported alias. Update their tests to assert the evaluation helpers directly;
Task 7 removes the aliases after every experiment caller has migrated.

- [ ] **Step 6: Run focused evaluation and compatibility tests**

Run:

```bash
uv run pytest tests/test_evaluation.py tests/experiments_utils/test_tokens.py tests/experiments_utils/test_controls.py -v
uv run ruff check src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py
```

Expected: all tests pass and ruff reports no errors.

- [ ] **Step 7: Commit the evaluation API**

```bash
git add src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py src/jlens_reasoning/experiments_utils/tokens.py src/jlens_reasoning/experiments_utils/controls.py tests/test_evaluation.py tests/experiments_utils/test_tokens.py tests/experiments_utils/test_controls.py
git commit -m "feat: add next-token evaluation contract"
```

### Task 2: Introduce the unified case and typed result foundation

**Files:**
- Create: `experiments/jlens_readout_sanity/experiment.py`
- Create: `tests/experiments/jlens_readout_sanity/test_experiment.py`

- [ ] **Step 1: Write failing case-contract tests**

Create `test_experiment.py` with:

```python
from dataclasses import FrozenInstanceError

import pytest

from experiments.jlens_readout_sanity.experiment import (
    Case,
    InterventionSpec,
    ReadoutSpec,
    validate_cases,
)


def spider_case() -> Case:
    return Case(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        readout=ReadoutSpec(
            concepts=("spider",),
            require_capability_gate=True,
        ),
        intervention=InterventionSpec(
            source_surface=" spider",
            target_surface=" ant",
            target_answers=("6", "six"),
        ),
    )


def test_case_composes_readout_and_intervention_without_kind_dispatch() -> None:
    case = spider_case()

    assert case.readout is not None
    assert case.readout.require_capability_gate
    assert case.intervention is not None
    assert case.intervention.alphas == (1.0, 2.0)
    with pytest.raises(FrozenInstanceError):
        case.key = "ant"  # type: ignore[misc]


@pytest.mark.parametrize(
    "cases",
    [
        (),
        (Case("", "prompt", ("answer",), readout=ReadoutSpec(("x",))),),
        (Case("x", "", ("answer",), readout=ReadoutSpec(("x",))),),
        (Case("x", "prompt", (), readout=ReadoutSpec(("x",))),),
        (Case("x", "prompt", ("answer",)),),
        (
            Case("x", "prompt", ("answer",), readout=ReadoutSpec(("x",))),
            Case("x", "other", ("answer",), readout=ReadoutSpec(("x",))),
        ),
    ],
)
def test_validate_cases_rejects_invalid_collections(cases: tuple[Case, ...]) -> None:
    with pytest.raises(ValueError):
        validate_cases(cases)


def test_validate_cases_accepts_readout_swap_and_combined_cases() -> None:
    cases = (
        Case("read", "p1", ("a",), readout=ReadoutSpec(("concept",))),
        Case(
            "swap",
            "p2",
            ("b",),
            intervention=InterventionSpec(" source", " target", ("c",)),
        ),
        spider_case(),
    )

    validate_cases(cases)
```

- [ ] **Step 2: Run the tests and verify the module is absent**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_experiment.py -v
```

Expected: collection fails because `experiment.py` does not exist.

- [ ] **Step 3: Add immutable operation and case specs**

Create `experiment.py` with:

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

Add this validation:

```python
def validate_cases(cases: Sequence[Case]) -> None:
    """Validate the complete caller-owned experiment case collection."""
    if not cases:
        raise ValueError("At least one experiment case is required")
    keys: set[str] = set()
    for case in cases:
        if not case.key.strip() or case.key in keys:
            raise ValueError("Case keys must be non-empty and unique")
        keys.add(case.key)
        if not case.prompt.strip():
            raise ValueError(f"Case {case.key!r} has an empty prompt")
        if not case.expected_answers or any(
            not answer.strip() for answer in case.expected_answers
        ):
            raise ValueError(f"Case {case.key!r} needs non-empty expected answers")
        if case.readout is None and case.intervention is None:
            raise ValueError(f"Case {case.key!r} has no configured operation")
        if case.readout is not None and not case.readout.concepts:
            raise ValueError(f"Case {case.key!r} has no readout concepts")
        if case.intervention is not None:
            spec = case.intervention
            if not spec.source_surface or not spec.target_surface:
                raise ValueError(f"Case {case.key!r} has an empty swap surface")
            if not spec.target_answers:
                raise ValueError(f"Case {case.key!r} has no target answers")
            if not spec.alphas or len(set(spec.alphas)) != len(spec.alphas):
                raise ValueError(f"Case {case.key!r} needs unique intervention alphas")
```

- [ ] **Step 4: Add typed result dataclasses without execution behavior**

Define these exact result shapes in the same module:

```python
@dataclass(frozen=True, slots=True)
class LensLocation:
    best_rank: int
    layer: int
    position: int


@dataclass(slots=True)
class ReadoutResult:
    jacobian_lens: LensLocation
    logit_lens: LensLocation
    workspace_layers: tuple[int, ...]
    scored_positions: tuple[int, ...]
    workspace_loading: float | None
    paper_top1_hit: bool | None
    capability_passed: bool | None
    raw_readouts: dict[str, object]


@dataclass(frozen=True, slots=True)
class InterventionCondition:
    alpha: float
    evaluation: NextTokenEvaluation
    comparison: RankComparison


@dataclass(slots=True)
class InterventionResult:
    source: TokenVariant
    target: TokenVariant
    formatting_prefix: tuple[dict[str, object], ...]
    workspace_layers: tuple[int, ...]
    clean_target: NextTokenEvaluation
    conditions: tuple[InterventionCondition, ...]


@dataclass(slots=True)
class CaseResult:
    case: Case
    baseline: EvaluationResult
    readout: ReadoutResult | None
    intervention: InterventionResult | None


@dataclass(slots=True)
class ExperimentResult:
    cases: tuple[CaseResult, ...]
    controls: dict[str, object]
    checks: dict[str, bool]
    failures: tuple[str, ...]
    policy: dict[str, object]
    metadata: dict[str, object]
    provenance: dict[str, object]

    @property
    def passed(self) -> bool:
        """Return whether every configured experiment check passed."""
        return bool(self.checks) and all(self.checks.values())
```

Import existing `EvaluationResult`, `NextTokenEvaluation`, `RankComparison`, and
`TokenVariant` from their owning modules. Keep tensors and model objects out of
serialized result dataclasses.

- [ ] **Step 5: Run and commit the foundation**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_experiment.py -v
uv run ruff check experiments/jlens_readout_sanity/experiment.py tests/experiments/jlens_readout_sanity/test_experiment.py
git add experiments/jlens_readout_sanity/experiment.py tests/experiments/jlens_readout_sanity/test_experiment.py
git commit -m "refactor: define unified sanity cases"
```

Expected: tests pass, ruff is clean, and the old runner remains untouched.

### Task 3: Restore full-output clean generation evaluation

**Files:**
- Modify: `experiments/jlens_readout_sanity/experiment.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_experiment.py`

- [ ] **Step 1: Add a failing baseline-evaluation test**

Add:

```python
from jlens_reasoning.evaluation import (
    AnswerStatus,
    GenerationStatus,
    ModelOutput,
)


def test_generate_and_evaluate_uses_full_raw_output_and_think_parser() -> None:
    case = spider_case()
    generated = ModelOutput(
        text="<think>A spider has eight legs.</think>\n 8.",
        token_ids=(1, 2, 3),
        token_pieces=("<think>reason</think>", " ", "8."),
        generation_status=GenerationStatus.COMPLETE,
        finish_reason="eos",
    )
    seen: list[str] = []

    def generate_output(prompt: str) -> ModelOutput:
        seen.append(prompt)
        return generated

    result = generate_and_evaluate(case, generate_output)

    assert seen == [case.prompt]
    assert result.raw_output is generated
    assert result.extracted_answer == "8"
    assert result.answer_status is AnswerStatus.CORRECT
    assert result.passed
```

- [ ] **Step 2: Verify the helper is missing**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_experiment.py::test_generate_and_evaluate_uses_full_raw_output_and_think_parser -v
```

Expected: FAIL because `generate_and_evaluate` is not defined.

- [ ] **Step 3: Implement the clean baseline evaluator**

Add:

```python
GenerateOutput = Callable[[str], ModelOutput]


def generate_and_evaluate(
    case: Case,
    generate_output: GenerateOutput,
) -> EvaluationResult:
    """Generate and evaluate one case's clean visible response."""
    output = generate_output(case.prompt)
    return evaluate(
        output,
        case.expected_answers,
        evaluator=SimpleFactualEvaluator(reasoning_parser=parse_think_tags),
    )
```

The experiment consumes a callback so model-specific generation stays visible
in the notebook and tests can supply exact raw output.

- [ ] **Step 4: Run the evaluation regression suite**

```bash
uv run pytest tests/test_evaluation.py tests/experiments/jlens_readout_sanity/test_experiment.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the restored integration primitive**

```bash
git add experiments/jlens_readout_sanity/experiment.py tests/experiments/jlens_readout_sanity/test_experiment.py
git commit -m "fix: restore sanity output evaluation"
```

### Task 4: Build the linear readout and intervention operations

**Files:**
- Modify: `experiments/jlens_readout_sanity/experiment.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_experiment.py`
- Modify: `src/jlens_reasoning/experiments_utils/tokens.py`

- [ ] **Step 1: Add a failing next-token intervention test**

Move the existing `RunnerTokenizer`, `SwapTokenizer`, `TensorBlock`, and
`TinySwapModel` fixtures verbatim from `test_runner.py` into
`test_experiment.py`. Add a `fake_runtime()` helper using the same identity
Jacobian, unembedding rows, and deterministic forward function as the existing
`test_analyze_swap_case_runs_clean_and_both_strengths` test. The assertions are:

```python
def test_run_intervention_evaluates_clean_and_each_alpha() -> None:
    result, prepared = run_intervention(
        case=spider_case(),
        spec=spider_case().intervention,
        runtime=fake_runtime(),
        layers=(2,),
    )

    assert prepared.case.key == "spider"
    assert result.clean_target.accepted_references == ("6", "six")
    assert [condition.alpha for condition in result.conditions] == [1.0, 2.0]
    assert all(
        condition.comparison.baseline_rank == result.clean_target.target_rank
        for condition in result.conditions
    )
    assert [condition.comparison.candidate_rank for condition in result.conditions] == [
        condition.evaluation.target_rank for condition in result.conditions
    ]
```

`fake_runtime()` returns an `ExperimentRuntime` with `TinySwapModel`, the
single-layer identity-Jacobian lens from the old test, `SwapTokenizer`, its
six-row unembedding matrix, the old deterministic `forward_next_token`, and a
`generate_output` callback returning `ModelOutput("8")`.

- [ ] **Step 2: Verify the new operation API is absent**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_experiment.py::test_run_intervention_evaluates_clean_and_each_alpha -v
```

Expected: FAIL because `ExperimentRuntime` and `run_intervention` are absent.

- [ ] **Step 3: Add the runtime and prepared-context types**

Add:

```python
@dataclass(frozen=True, slots=True)
class ExperimentRuntime:
    model: object
    lens: object
    tokenizer: object
    unembedding_weight: torch.Tensor
    forward_next_token: Callable[[torch.Tensor], torch.Tensor]
    generate_output: GenerateOutput


@dataclass(slots=True)
class PreparedIntervention:
    case: Case
    source: TokenVariant
    target: TokenVariant
    scoring_input: torch.Tensor
    formatting_prefix: tuple[dict[str, object], ...]
    clean_logits: torch.Tensor
    vectors_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    workspace_loading: float | None
```

`PreparedIntervention` is runtime-only and is never placed inside
`ExperimentResult`.

- [ ] **Step 4: Move existing readout behavior into `run_readout`**

Move `_summarize_lens`, `_readout_payload`, and the lens-application portion of
`analyze_case` from `runner.py` into `experiment.py`. Change the function to:

```python
def run_readout(
    case: Case,
    spec: ReadoutSpec,
    runtime: ExperimentRuntime,
    *,
    layers: Sequence[int],
    top_k: int,
) -> ReadoutResult:
    """Measure configured concepts across fitted layers and scored positions."""
```

Use `spec.concepts`, `spec.literal_argument`, and
`spec.require_capability_gate`; remove every `case.key == "spider"` condition.
Use `best_token_rank` and `top_token_values` from `evaluation_utils` for raw lens
diagnostics. Return `LensLocation` values instead of nested summary dictionaries.

- [ ] **Step 5: Move context preparation and swap execution into one path**

Move `_prepare_intervention_context` and `analyze_swap_case` behavior from
`runner.py`, then implement:

```python
def run_intervention(
    case: Case,
    spec: InterventionSpec,
    runtime: ExperimentRuntime,
    *,
    layers: Sequence[int],
    top_k: int = TOP_K,
) -> tuple[InterventionResult, PreparedIntervention]:
    """Run and compare every configured coordinate-swap strength for one case."""
```

The implementation must:

1. resolve the configured source and target with `single_token_surface`;
2. call `prepare_scoring_input` once;
3. obtain clean logits once;
4. call `evaluate_next_token(clean_logits, spec.target_answers, tokenizer)`;
5. construct real vectors once with `token_vectors_by_layer`;
6. call `execute_intervention` once per alpha;
7. evaluate each intervened logits vector with `evaluate_next_token`;
8. compare each condition with `compare_token_ranks`;
9. return the serialized `InterventionResult` and runtime-only prepared context.

- [ ] **Step 6: Run focused mechanics and operation tests**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_experiment.py tests/experiments_utils/test_interventions.py tests/experiments_utils/test_tokens.py -v
uv run ruff check experiments/jlens_readout_sanity/experiment.py
```

Expected: all tests pass; reusable intervention mechanics are unchanged.

- [ ] **Step 7: Commit the linear operations**

```bash
git add experiments/jlens_readout_sanity/experiment.py src/jlens_reasoning/experiments_utils/tokens.py tests/experiments/jlens_readout_sanity/test_experiment.py
git commit -m "refactor: linearize sanity case execution"
```

### Task 5: Migrate all negative controls to shared rank evaluation

**Files:**
- Modify: `experiments/jlens_readout_sanity/controls.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_controls.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_control_analysis.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_control_execution.py`

- [ ] **Step 1: Add a failing shared-evaluator control test**

Add a test that monkeypatches the control module's imported evaluator:

```python
def test_control_condition_uses_shared_next_token_evaluation(monkeypatch) -> None:
    calls: list[tuple[str, ...]] = []
    actual = evaluate_next_token

    class ControlRankTokenizer:
        def encode(
            self, text: str, *, add_special_tokens: bool = False
        ) -> list[int]:
            assert add_special_tokens is False
            return [2] if text.strip().casefold() == "target" else [0, 1]

        def decode(
            self,
            token_ids: list[int],
            *,
            clean_up_tokenization_spaces: bool = False,
        ) -> str:
            assert clean_up_tokenization_spaces is False
            return "target" if token_ids[0] == 2 else f"token-{token_ids[0]}"

    def recording_evaluator(logits, references, tokenizer, *, top_k=25):
        calls.append(tuple(references))
        return actual(logits, references, tokenizer, top_k=top_k)

    monkeypatch.setattr(controls_module, "evaluate_next_token", recording_evaluator)

    clean = NextTokenEvaluation(
        accepted_references=("target",),
        accepted_token_ids=(2,),
        top1_id=0,
        top1_token="token-0",
        target_rank=8,
        top_tokens=(),
    )

    result = evaluate_control_condition(
        clean=clean,
        intervened_logits=torch.tensor([0.0, 3.0, 1.0]),
        accepted_references=("target",),
        tokenizer=ControlRankTokenizer(),
        top_k=3,
    )

    assert calls == [("target",)]
    assert result.comparison.baseline_rank == 8
    assert result.comparison.candidate_rank == result.evaluation.target_rank
```

Use a small tokenizer fixture whose `"target"` variants resolve to token ID 2.

- [ ] **Step 2: Verify the control adapter is absent**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_controls.py::test_control_condition_uses_shared_next_token_evaluation -v
```

Expected: FAIL because `evaluate_control_condition` does not exist.

- [ ] **Step 3: Add one typed control-condition adapter**

Define in `controls.py`:

```python
@dataclass(frozen=True, slots=True)
class ControlConditionResult:
    evaluation: NextTokenEvaluation
    comparison: RankComparison


def evaluate_control_condition(
    *,
    clean: NextTokenEvaluation,
    intervened_logits: torch.Tensor,
    accepted_references: Sequence[str],
    tokenizer: object,
    top_k: int,
) -> ControlConditionResult:
    """Evaluate one control intervention against its reference condition."""
    candidate = evaluate_next_token(
        intervened_logits,
        accepted_references,
        tokenizer,
        top_k=top_k,
    )
    return ControlConditionResult(
        evaluation=candidate,
        comparison=compare_token_ranks(clean, candidate),
    )
```

- [ ] **Step 4: Route every real and control rank through typed evaluations**

Refactor identity, matched-random-vector, wrong-concept, and random-target
execution to accept `PreparedIntervention` values. Replace direct calls to
`best_target_rank` and `log_rank_gain` with `evaluate_control_condition` and
fields from `RankComparison`.

Keep these existing policies exactly unchanged:

```python
CONTROL_ALPHA = 1.0
PERCENTILE_QUANTILE = 0.95
WRONG_CONCEPT_REQUIRED_CASE_WINS = 4
```

Identity still additionally checks full-logit `torch.allclose`. Random controls
still use the existing deterministic seeds, namespaces, exclusions, and
norm-preservation helpers.

- [ ] **Step 5: Add the new suite entry point while preserving old imports**

Add:

```python
def run_control_suite(
    *,
    prepared: Sequence[PreparedIntervention],
    interventions: Sequence[InterventionResult],
    runtime: ExperimentRuntime,
    layers: Sequence[int],
    top_k: int,
) -> dict[str, object]:
    """Run deterministic negative controls and apply their aggregate gates."""
```

During this task keep the existing `run_negative_controls` facade working so
the old runner remains green. Task 7 deletes it after the notebook cutover.

- [ ] **Step 6: Run the complete control suites**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_controls.py tests/experiments/jlens_readout_sanity/test_control_analysis.py tests/experiments/jlens_readout_sanity/test_control_execution.py tests/experiments_utils/test_controls.py -v
```

Expected: all existing scientific-control regressions and the new shared-
evaluator test pass.

- [ ] **Step 7: Commit the control migration**

```bash
git add experiments/jlens_readout_sanity/controls.py tests/experiments/jlens_readout_sanity/test_controls.py tests/experiments/jlens_readout_sanity/test_control_analysis.py tests/experiments/jlens_readout_sanity/test_control_execution.py
git commit -m "refactor: share sanity rank evaluation"
```

### Task 6: Cut over orchestration and the notebook

**Files:**
- Modify: `experiments/jlens_readout_sanity/experiment.py`
- Modify: `experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb`
- Modify: `experiments/jlens_readout_sanity/reporting.py`
- Modify: `experiments/jlens_readout_sanity/utils.py`
- Modify: `src/jlens_reasoning/experiments_utils/artifacts.py`
- Modify: `tests/experiments/jlens_readout_sanity/case_fixtures.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_experiment.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_reporting.py`
- Modify: `tests/experiments_utils/test_artifacts.py`
- Modify: `tests/test_notebooks.py`

- [ ] **Step 1: Replace notebook contract expectations first**

Change `case_fixtures.py` to return one collection:

```python
def _load_notebook_cases() -> tuple[Any, ...]:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    case_cell = next(cell for cell in notebook.cells if cell.id == "define-cases")
    namespace: dict[str, Any] = {}
    exec(compile(case_cell.source, f"{NOTEBOOK}:define-cases", "exec"), namespace)
    return namespace["CASES"]


CASES = _load_notebook_cases()
```

Update `test_notebooks.py` to require:

```python
assert "CASES = (" in cells_by_id["define-cases"]
assert "READOUT_CASES" not in source
assert "SWAP_CASES" not in source
assert "generate_output" in cells_by_id["run-experiment"]
assert "causal_lm.generate" in cells_by_id["run-experiment"]
assert "ModelOutput" in source
assert "GenerationStatus" in source
assert "cases=CASES" in cells_by_id["run-experiment"]
```

- [ ] **Step 2: Run notebook tests and verify the old cells fail**

```bash
uv run pytest tests/test_notebooks.py tests/experiments/jlens_readout_sanity/test_experiment.py -v
```

Expected: FAIL because the notebook still defines parallel case collections and
does not generate full outputs.

- [ ] **Step 3: Implement `run_case` and `run_experiment`**

Add:

```python
def run_case(
    case: Case,
    runtime: ExperimentRuntime,
    *,
    layers: Sequence[int],
    top_k: int,
) -> tuple[CaseResult, PreparedIntervention | None]:
    """Run the clean evaluation and every operation configured for one case."""
    baseline = generate_and_evaluate(case, runtime.generate_output)
    readout = (
        run_readout(case, case.readout, runtime, layers=layers, top_k=top_k)
        if case.readout is not None
        else None
    )
    if case.intervention is None:
        intervention = None
        prepared = None
    else:
        intervention, prepared = run_intervention(
            case,
            case.intervention,
            runtime,
            layers=layers,
            top_k=top_k,
        )
    return CaseResult(case, baseline, readout, intervention), prepared
```

Use this public orchestration signature:

```python
def run_experiment(
    *,
    cases: Sequence[Case],
    runtime: ExperimentRuntime,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
    top_k: int = TOP_K,
) -> ExperimentResult:
    """Run the complete configured J-Lens readout sanity experiment."""
```

The body validates cases and model/lens compatibility, derives workspace layers
once, resolves all swap surfaces before model forwards, runs `run_case` once per
case, passes intervention-bearing results and prepared contexts to
`run_control_suite`, and aggregates the existing capability gates. The clean-
baseline gate is `all(case.baseline.passed for case in results)`. Require every
intervention spec to contain `CONTROL_ALPHA` because the control suite compares
against that real condition.

- [ ] **Step 4: Replace the notebook case cell with one ordered collection**

Use `Case`, `ReadoutSpec`, and `InterventionSpec`. Preserve the existing five
keys, prompts, answers, concepts, literals, swap surfaces, targets, and ordering.
The spider declaration is:

```python
Case(
    key="spider",
    prompt="The number of legs on the animal that spins webs is",
    expected_answers=("8", "eight"),
    readout=ReadoutSpec(
        concepts=("spider",),
        require_capability_gate=True,
    ),
    intervention=InterventionSpec(
        source_surface=" spider",
        target_surface=" ant",
        target_answers=("6", "six"),
    ),
)
```

Each France case uses `ReadoutSpec(("France",), literal_argument="France")` and
`InterventionSpec(" France", " China", target_answers)`, where the ordered
target-answer tuples are `("Beijing",)`, `("Chinese",)`, `("Asia",)`, and
`("Yuan",)` for capital, language, continent, and currency respectively.

- [ ] **Step 5: Restore deterministic generation visibly in the run cell**

Define `generate_output(prompt: str) -> ModelOutput` with sampling disabled and
`max_new_tokens=64`. Preserve generated token IDs and individually decoded token
pieces, remove the terminal EOS token only from decoded text, and set:

```python
generation_status = (
    GenerationStatus.COMPLETE
    if generated_ids and generated_ids[-1] in eos_token_ids
    else GenerationStatus.TRUNCATED
)
finish_reason = "eos" if generation_status is GenerationStatus.COMPLETE else "length"
```

Pass `generate_output` through `ExperimentRuntime`, call
`run_experiment(cases=CASES, runtime=runtime)`, and keep execution, save, report,
and final failure checks in their existing separate cells. Update the save cell
to assign `result.provenance = {...}`, and update the final cell to read
`result.passed` and `result.failures` instead of subscripting the result.

- [ ] **Step 6: Export the new entry point directly**

Change `utils.py` to import `run_experiment` from `experiment.py`; do not re-export
the old `run_readout_sanity` name. Update notebook imports and package tests to
use the new name.

- [ ] **Step 7: Rewrite reporting around typed results**

Replace dictionary fixtures in `test_reporting.py` with `ExperimentResult`,
`CaseResult`, `ReadoutResult`, and `InterventionResult` fixtures. Preserve the
section-order assertions and require the report to contain clean extracted
answers, clean `answer_status`, alpha-1 and alpha-2 ranks, control outcomes, and
failure details.

- [ ] **Step 8: Verify the old reporter rejects typed results**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_reporting.py -v
```

Expected: FAIL because the reporter still indexes nested dictionaries.

- [ ] **Step 9: Implement typed reporting**

Keep `_format_table`, remove the generic seven-field `ReportRow`, and implement
these typed row builders:

```python
def capability_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build compact rows for configured capability checks."""


def readout_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build per-case clean-answer and readout summary rows."""


def intervention_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build per-case clean and intervened target-rank rows."""


def control_rows(result: ExperimentResult) -> tuple[tuple[str, ...], ...]:
    """Build aggregate negative-control result rows."""
```

`render_sanity_report(result: ExperimentResult)` reads `result.passed`,
`result.failures`, and `result.provenance` directly. It must not use a next-token
top-1 value as the clean generated-answer status.

- [ ] **Step 10: Add a failing typed-artifact regression**

Extend `test_artifacts.py` with a frozen dataclass containing a tensor and assert
that `write_results` produces the same JSON primitives as a mapping. Also assert
that `get_type_hints(write_results)["result"] is object` so the existing
`Mapping` annotation produces the intended red test.

```bash
uv run pytest tests/experiments_utils/test_artifacts.py -v
```

Expected: FAIL because `write_results` still declares `Mapping[str, Any]`.

- [ ] **Step 11: Widen artifact serialization to typed results**

Change the public signature without changing recursive behavior:

```python
def write_results(path: Path, result: object) -> None:
    """Serialize a typed experiment result or mapping as stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
```

- [ ] **Step 12: Run notebook and typed-result integration tests**

```bash
uv run pytest tests/test_notebooks.py tests/experiments/jlens_readout_sanity/test_experiment.py tests/experiments/jlens_readout_sanity/test_package.py tests/experiments/jlens_readout_sanity/test_reporting.py tests/experiments_utils/test_artifacts.py -v
```

Expected: all tests pass, the evaluator-absence assertions are gone, and case
definitions are visible once. The typed result serializes and renders without a
legacy dictionary adapter.

- [ ] **Step 13: Commit the cutover**

```bash
git add experiments/jlens_readout_sanity/experiment.py experiments/jlens_readout_sanity/jlens_readout_sanity.ipynb experiments/jlens_readout_sanity/reporting.py experiments/jlens_readout_sanity/utils.py src/jlens_reasoning/experiments_utils/artifacts.py tests/experiments/jlens_readout_sanity/case_fixtures.py tests/experiments/jlens_readout_sanity/test_experiment.py tests/experiments/jlens_readout_sanity/test_package.py tests/experiments/jlens_readout_sanity/test_reporting.py tests/experiments_utils/test_artifacts.py tests/test_notebooks.py
git commit -m "refactor: run unified sanity cases"
```

### Task 7: Remove superseded architecture and compatibility aliases

**Files:**
- Modify: `experiments/jlens_readout_sanity/controls.py`
- Modify: `src/jlens_reasoning/experiments_utils/tokens.py`
- Modify: `src/jlens_reasoning/experiments_utils/controls.py`
- Modify: `tests/experiments_utils/test_tokens.py`
- Modify: `tests/experiments_utils/test_controls.py`
- Delete: `experiments/jlens_readout_sanity/runner.py`
- Delete: `experiments/jlens_readout_sanity/types.py`
- Delete: `experiments/jlens_readout_sanity/control_analysis.py`
- Delete: `experiments/jlens_readout_sanity/control_execution.py`
- Delete: `tests/experiments/jlens_readout_sanity/test_runner.py`
- Delete: `tests/experiments/jlens_readout_sanity/test_control_analysis.py`
- Delete: `tests/experiments/jlens_readout_sanity/test_control_execution.py`

- [ ] **Step 1: Preserve every scientific invariant in the new tests**

Before deleting old tests, confirm `test_experiment.py` or `test_controls.py`
contains explicit coverage for artifact coordinates, strict single-token swap
surfaces, literal-position selection, workspace-layer bounds, readout rank and
paper diagnostics, clean-generation capability gates, three-of-five swap
improvement, any-target-top-1, control-alpha presence, pre-forward swap
validation, identity tolerances, random-vector norm preservation, deterministic
seeds and random targets, wrong-concept wins, and all four global control gates.

- [ ] **Step 2: Delete old modules and compatibility aliases**

Move any still-needed pure analysis functions into the final `controls.py`, then
delete `runner.py`, `types.py`, `control_analysis.py`, and
`control_execution.py`. Delete their superseded tests after copying every
scientific invariant into `test_experiment.py` or `test_controls.py`.

Remove `best_target_rank` and `top_tokens` compatibility aliases from
`experiments_utils.tokens`, and remove `log_rank_gain` from
`experiments_utils.controls`. Update all imports to their evaluation owners.

- [ ] **Step 3: Verify no superseded imports remain**

```bash
rg -n 'ReadoutCase|SwapCase|ResolvedSwapCase|run_readout_sanity|control_analysis|control_execution|best_target_rank|experiments_utils\.controls import log_rank_gain' experiments src tests
```

Expected: no matches.

- [ ] **Step 4: Run experiment, control, reporting, and artifact tests**

```bash
uv run pytest tests/test_evaluation.py tests/experiments/jlens_readout_sanity tests/experiments_utils -v
```

Expected: all tests pass with fewer, behavior-focused test modules.

- [ ] **Step 5: Commit the architectural cleanup**

```bash
git add experiments/jlens_readout_sanity src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py src/jlens_reasoning/experiments_utils tests/experiments tests/experiments_utils
git commit -m "refactor: remove fragmented sanity runner"
```

Before committing, run `git diff --cached --name-only` and confirm `README.md`
is absent.

### Task 8: Enforce one-line function documentation and verify everything

**Files:**
- Modify: `experiments/jlens_readout_sanity/experiment.py`
- Modify: `experiments/jlens_readout_sanity/controls.py`
- Modify: `experiments/jlens_readout_sanity/reporting.py`
- Modify: `src/jlens_reasoning/evaluation.py`
- Modify: `src/jlens_reasoning/evaluation_utils.py`
- Modify: `src/jlens_reasoning/experiments_utils/interventions.py`
- Modify: `tests/experiments/jlens_readout_sanity/test_package.py`

- [ ] **Step 1: Add an AST-based documentation regression**

Add this test to `test_package.py`:

```python
import ast


DOCUMENTED_MODULES = (
    Path("experiments/jlens_readout_sanity/experiment.py"),
    Path("experiments/jlens_readout_sanity/controls.py"),
    Path("experiments/jlens_readout_sanity/reporting.py"),
    Path("src/jlens_reasoning/evaluation.py"),
    Path("src/jlens_reasoning/evaluation_utils.py"),
    Path("src/jlens_reasoning/experiments_utils/interventions.py"),
)


def test_public_and_large_functions_have_one_line_docstrings() -> None:
    missing: list[str] = []
    multiline: list[str] = []
    for path in DOCUMENTED_MODULES:
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(module):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            body_start = node.body[0].lineno
            body_lines = node.end_lineno - body_start + 1
            requires_doc = not node.name.startswith("_") or body_lines >= 20
            if not requires_doc:
                continue
            docstring = ast.get_docstring(node, clean=False)
            label = f"{path}:{node.lineno}:{node.name}"
            if docstring is None:
                missing.append(label)
            elif "\n" in docstring:
                multiline.append(label)

    assert missing == []
    assert multiline == []
```

- [ ] **Step 2: Run the regression and collect exact missing functions**

```bash
uv run pytest tests/experiments/jlens_readout_sanity/test_package.py::test_public_and_large_functions_have_one_line_docstrings -v
```

Expected: FAIL listing every public or 20-line function still missing a
one-sentence docstring.

- [ ] **Step 3: Add concise responsibility docstrings**

Add one sentence to each listed function. Use responsibility-focused wording,
including these required sentences:

```python
"""Run the clean evaluation and every operation configured for one case."""
"""Run deterministic negative controls and apply their aggregate gates."""
"""Render the complete sanity result as stable, copyable plain text."""
"""Evaluate a next-token distribution against predefined accepted answers."""
"""Patch configured model layers while evaluating one intervention condition."""
```

Do not add docstrings to tiny private formatting or arithmetic helpers unless
the AST rule requires them.

- [ ] **Step 4: Run formatting, lint, and the complete test suite**

```bash
uv run ruff format experiments/jlens_readout_sanity src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py src/jlens_reasoning/experiments_utils tests/experiments/jlens_readout_sanity tests/experiments_utils tests/test_evaluation.py tests/test_notebooks.py
uv run ruff format --check .
uv run ruff check .
uv run pytest -v
```

Expected: formatting and lint are clean; the complete suite passes without
network or GPU access.

- [ ] **Step 5: Inspect the final scope and readability**

```bash
git diff --stat b6e7912..HEAD
wc -l experiments/jlens_readout_sanity/*.py src/jlens_reasoning/evaluation*.py
rg -n '^(def|class) ' experiments/jlens_readout_sanity src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py
git status --short
```

Expected:

- one case collection and one `run_case` path;
- no parallel read/swap joins or post-assembly baseline mutation;
- clean generation uses `SimpleFactualEvaluator`;
- interventions and controls use `evaluate_next_token`;
- intervention math remains under `experiments_utils`;
- every public or large function has a one-line docstring;
- `README.md` remains modified but unstaged and untouched.

- [ ] **Step 6: Commit documentation and final cleanup**

```bash
git add experiments/jlens_readout_sanity/experiment.py experiments/jlens_readout_sanity/controls.py experiments/jlens_readout_sanity/reporting.py src/jlens_reasoning/evaluation.py src/jlens_reasoning/evaluation_utils.py src/jlens_reasoning/experiments_utils/interventions.py tests/experiments/jlens_readout_sanity/test_package.py
git commit -m "docs: document sanity execution functions"
```

Run `git status --short --branch` afterward. The only remaining worktree change
should be the pre-existing `README.md` modification.
