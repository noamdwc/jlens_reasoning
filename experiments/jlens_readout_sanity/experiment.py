"""Typed configuration and execution contracts for J-Lens readout sanity."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from jlens.hooks import ActivationRecorder

from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    DEFAULT_MAX_FORMATTING_TOKENS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    SPIDER_READ_MAX_RANK,
    SWAP_TARGET_TOP1_REQUIRED_COUNT,
    TOP_K,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
)
from jlens_reasoning.evaluation import (
    EvaluationResult,
    ModelOutput,
    NextTokenEvaluation,
    RankComparison,
    SimpleFactualEvaluator,
    compare_token_ranks,
    evaluate,
    evaluate_next_token,
)
from jlens_reasoning.evaluation_utils import (
    best_token_rank,
    parse_think_tags,
    top_token_values,
)
from jlens_reasoning.experiments_utils.interventions import (
    execute_intervention,
    token_vectors_by_layer,
)
from jlens_reasoning.experiments_utils.tokens import (
    TokenVariant,
    concept_token_variants,
    positions_after_literal,
    positions_from_literal,
    prepare_scoring_input,
    single_token_surface,
)
from jlens_reasoning.experiments_utils.validation import (
    validate_model_lens,
    workspace_layers,
    workspace_loading,
)


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


OutputGenerator = Callable[[str], ModelOutput]


@dataclass(frozen=True, slots=True)
class ExperimentRuntime:
    model: Any
    lens: Any
    tokenizer: Any
    unembedding_weight: torch.Tensor
    forward_next_token: Callable[[torch.Tensor], torch.Tensor]
    generate_output: OutputGenerator


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
    generate_output: OutputGenerator,
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
class PreparedIntervention:
    case: Case
    source: TokenVariant
    target: TokenVariant
    scoring_input: torch.Tensor
    formatting_prefix: tuple[dict[str, object], ...]
    clean_logits: torch.Tensor
    vectors_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    workspace_loading: float | None


def _lens_location(
    logits_by_layer: dict[int, torch.Tensor],
    layers: Sequence[int],
    positions: Sequence[int],
    target_ids: Sequence[int],
) -> LensLocation:
    rank, layer, position = min(
        (best_token_rank(logits_by_layer[layer][position], target_ids), layer, position)
        for layer in layers
        for position in positions
    )
    return LensLocation(rank, layer, position)


def _readout_diagnostics(
    logits_by_layer: dict[int, torch.Tensor],
    tokenizer: Any,
    target_ids: Sequence[int],
    *,
    top_k: int,
) -> dict[str, object]:
    return {
        str(layer): tuple(
            {
                "position": position,
                "target_rank": best_token_rank(position_logits, target_ids),
                "top_tokens": tuple(
                    {
                        "token_id": token_id,
                        "token": token,
                        "logit": logit,
                    }
                    for token_id, token, logit in top_token_values(
                        position_logits,
                        tokenizer,
                        k=top_k,
                    )
                ),
            }
            for position, position_logits in enumerate(layer_logits)
        )
        for layer, layer_logits in sorted(logits_by_layer.items())
    }


def run_readout(
    case: Case,
    spec: ReadoutSpec,
    runtime: ExperimentRuntime,
    *,
    layers: Sequence[int],
    top_k: int = TOP_K,
) -> ReadoutResult:
    """Measure one case with the Jacobian and logit lenses."""
    jacobian_logits, model_logits, input_ids = runtime.lens.apply(
        runtime.model,
        case.prompt,
        positions=None,
    )
    logit_logits, repeated_logits, repeated_input_ids = runtime.lens.apply(
        runtime.model,
        case.prompt,
        positions=None,
        use_jacobian=False,
    )
    if not torch.equal(input_ids, repeated_input_ids):
        raise RuntimeError("J-Lens and logit-lens tokenization differed")
    if not torch.equal(model_logits, repeated_logits):
        raise RuntimeError("J-Lens and logit-lens baseline logits differed")

    variants = concept_token_variants(runtime.tokenizer, spec.concepts)
    target_ids = tuple(variant.token_id for variant in variants)
    positions = (
        tuple(range(input_ids.shape[1]))
        if spec.literal_argument is None
        else tuple(
            positions_after_literal(
                runtime.tokenizer,
                input_ids,
                spec.literal_argument,
            )
        )
    )
    workspace = tuple(layers)
    if not workspace:
        raise ValueError("Readout requires at least one workspace layer")
    jacobian_location = _lens_location(
        jacobian_logits,
        workspace,
        positions,
        target_ids,
    )
    logit_location = _lens_location(
        logit_logits,
        workspace,
        positions,
        target_ids,
    )
    paper_top1_hit = None
    capability_passed = None
    if spec.require_capability_gate:
        paper_top1_hit = jacobian_location.best_rank == 1
        capability_passed = (
            jacobian_location.best_rank <= SPIDER_READ_MAX_RANK
            and jacobian_location.best_rank < logit_location.best_rank
        )

    input_token_ids = tuple(int(token_id) for token_id in input_ids[0].tolist())
    baseline_top1_id = int(model_logits[-1].argmax().item())
    return ReadoutResult(
        jacobian_lens=jacobian_location,
        logit_lens=logit_location,
        workspace_layers=workspace,
        scored_positions=positions,
        workspace_loading=None,
        paper_top1_hit=paper_top1_hit,
        capability_passed=capability_passed,
        raw_readouts={
            "input_ids": input_token_ids,
            "input_tokens": tuple(
                runtime.tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                )
                for token_id in input_token_ids
            ),
            "target_variants": tuple(
                {"token_id": variant.token_id, "surface": variant.surface}
                for variant in variants
            ),
            "baseline": {
                "top1_id": baseline_top1_id,
                "top1_token": runtime.tokenizer.decode(
                    [baseline_top1_id],
                    clean_up_tokenization_spaces=False,
                ),
            },
            "jacobian_lens": _readout_diagnostics(
                jacobian_logits,
                runtime.tokenizer,
                target_ids,
                top_k=top_k,
            ),
            "logit_lens": _readout_diagnostics(
                logit_logits,
                runtime.tokenizer,
                target_ids,
                top_k=top_k,
            ),
        },
    )


def run_intervention(
    case: Case,
    spec: InterventionSpec,
    runtime: ExperimentRuntime,
    *,
    layers: Sequence[int],
    top_k: int = TOP_K,
) -> tuple[InterventionResult, PreparedIntervention]:
    """Prepare, execute, and evaluate one configured coordinate swap."""
    workspace = tuple(layers)
    if not workspace:
        raise ValueError("Intervention requires at least one workspace layer")
    source = single_token_surface(runtime.tokenizer, spec.source_surface)
    target = single_token_surface(runtime.tokenizer, spec.target_surface)
    input_ids = runtime.model.encode(case.prompt)
    scoring_input, formatting_prefix = prepare_scoring_input(
        input_ids,
        forward_next_token=runtime.forward_next_token,
        tokenizer=runtime.tokenizer,
        max_formatting_tokens=DEFAULT_MAX_FORMATTING_TOKENS,
    )
    vectors_by_layer = token_vectors_by_layer(
        lens=runtime.lens,
        unembedding_weight=runtime.unembedding_weight,
        layers=workspace,
        source_token_id=source.token_id,
        target_token_id=target.token_id,
    )
    loading = None
    if case.readout is not None and case.readout.literal_argument is not None:
        with (
            torch.inference_mode(),
            ActivationRecorder(runtime.model.layers, at=workspace) as recorder,
        ):
            runtime.forward_next_token(input_ids)
        loading = workspace_loading(
            recorder.activations,
            {layer: vectors_by_layer[layer][0] for layer in workspace},
            positions=positions_from_literal(
                runtime.tokenizer,
                input_ids,
                case.readout.literal_argument,
            ),
        )

    with torch.inference_mode():
        clean_logits = runtime.forward_next_token(scoring_input)
    clean_target = evaluate_next_token(
        clean_logits,
        spec.target_answers,
        runtime.tokenizer,
        top_k=top_k,
    )
    conditions = []
    for alpha in spec.alphas:
        intervened_logits = execute_intervention(
            model=runtime.model,
            forward_next_token=runtime.forward_next_token,
            scoring_input=scoring_input,
            vectors_by_layer=vectors_by_layer,
            alpha=alpha,
        )
        evaluation = evaluate_next_token(
            intervened_logits,
            spec.target_answers,
            runtime.tokenizer,
            top_k=top_k,
        )
        conditions.append(
            InterventionCondition(
                alpha=alpha,
                evaluation=evaluation,
                comparison=compare_token_ranks(clean_target, evaluation),
            )
        )

    prefix = tuple(formatting_prefix)
    prepared = PreparedIntervention(
        case=case,
        source=source,
        target=target,
        scoring_input=scoring_input,
        formatting_prefix=prefix,
        clean_logits=clean_logits,
        vectors_by_layer=vectors_by_layer,
        workspace_loading=loading,
    )
    result = InterventionResult(
        source=source,
        target=target,
        formatting_prefix=prefix,
        workspace_layers=workspace,
        clean_target=clean_target,
        conditions=tuple(conditions),
    )
    return result, prepared


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


def run_case(
    case: Case,
    runtime: ExperimentRuntime,
    *,
    layers: Sequence[int],
    top_k: int,
) -> tuple[CaseResult, PreparedIntervention | None]:
    """Run the clean evaluation and each operation configured for one case."""
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
        if readout is not None:
            readout.workspace_loading = prepared.workspace_loading
    return CaseResult(case, baseline, readout, intervention), prepared


def run_experiment(
    *,
    cases: Sequence[Case],
    runtime: ExperimentRuntime,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
    top_k: int = TOP_K,
) -> ExperimentResult:
    """Run the complete configured J-Lens readout sanity experiment."""
    from experiments.jlens_readout_sanity.control_analysis import (
        aggregate_all_checks,
    )
    from experiments.jlens_readout_sanity.controls import run_control_suite

    validate_cases(cases)
    validate_model_lens(runtime.model, runtime.lens)
    layers = tuple(
        workspace_layers(
            runtime.model.n_layers,
            runtime.lens.source_layers,
            lower_fraction=WORKSPACE_LAYER_LOWER_FRACTION,
            upper_fraction=WORKSPACE_LAYER_UPPER_FRACTION,
        )
    )
    if not layers:
        raise ValueError("No fitted layers fall inside the workspace range")

    intervention_cases = [case for case in cases if case.intervention is not None]
    for case in intervention_cases:
        spec = case.intervention
        assert spec is not None
        single_token_surface(runtime.tokenizer, spec.source_surface)
        single_token_surface(runtime.tokenizer, spec.target_surface)
        if CONTROL_ALPHA not in spec.alphas:
            raise ValueError(
                f"Case {case.key!r} must include control alpha {CONTROL_ALPHA:g}"
            )

    results = []
    prepared = []
    interventions = []
    for case in cases:
        result, context = run_case(case, runtime, layers=layers, top_k=top_k)
        results.append(result)
        if context is not None:
            prepared.append(context)
            assert result.intervention is not None
            interventions.append(result.intervention)

    readout_gates = [
        result.readout
        for result in results
        if result.case.readout is not None
        and result.case.readout.require_capability_gate
    ]
    improved_count = sum(
        any(condition.comparison.improved for condition in intervention.conditions)
        for intervention in interventions
    )
    top1_count = sum(
        any(condition.comparison.reached_top1 for condition in intervention.conditions)
        for intervention in interventions
    )
    checks = {
        "clean_baselines": all(result.baseline.passed for result in results),
        "spider_read": bool(readout_gates)
        and all(readout.capability_passed is True for readout in readout_gates),
        "swap_rank_improvements": improved_count >= minimum_improvements,
        "swap_target_top1": top1_count >= SWAP_TARGET_TOP1_REQUIRED_COUNT,
    }
    failures = []
    if not checks["clean_baselines"]:
        failures.append("one or more clean baseline answers failed evaluation")
    if not checks["spider_read"]:
        failures.append("configured readout capability gate failed")
    if not checks["swap_rank_improvements"]:
        failures.append(
            f"coordinate swaps improved {improved_count}/{len(interventions)} "
            f"target ranks; need at least {minimum_improvements}"
        )
    if not checks["swap_target_top1"]:
        failures.append("no coordinate swap placed its target answer at top-1")

    controls = run_control_suite(
        prepared=prepared,
        interventions=interventions,
        runtime=runtime,
        layers=layers,
        top_k=top_k,
    )
    checks, failures, _ = aggregate_all_checks(checks, failures, controls)
    case_count = len(interventions)
    return ExperimentResult(
        cases=tuple(results),
        controls=controls,
        checks=checks,
        failures=tuple(failures),
        policy={
            "clean_baselines": {"required_count": len(results)},
            "spider_read": {
                "maximum_rank": SPIDER_READ_MAX_RANK,
                "requires_better_than_logit_lens": True,
                "paper_target_rank": 1,
            },
            "swap_rank_improvements": {
                "required_count": minimum_improvements,
                "case_count": case_count,
            },
            "swap_target_top1": {
                "required_count": SWAP_TARGET_TOP1_REQUIRED_COUNT,
                "case_count": case_count,
                "paper_primary_alpha": CONTROL_ALPHA,
                "paper_target_rank": 1,
            },
        },
        metadata={
            "workspace_layers": layers,
            "top_k": top_k,
            "case_count": len(results),
        },
        provenance={},
    )
