"""Typed configuration and execution contracts for J-Lens readout sanity."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jlens_reasoning.evaluation import (
    EvaluationResult,
    ModelOutput,
    NextTokenEvaluation,
    RankComparison,
    SimpleFactualEvaluator,
    evaluate,
)
from jlens_reasoning.evaluation_utils import parse_think_tags
from jlens_reasoning.experiments_utils.tokens import TokenVariant


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


GenerateOutput = Callable[[str], ModelOutput]


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
