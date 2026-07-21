"""Read-and-change sanity checks for the public Qwen Jacobian lens."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from jlens.hooks import ActivationRecorder
from torch import nn

from jlens_reasoning.experiments.readout_cases import (
    READOUT_CASES,
    SWAP_CASES,
    ReadoutCase,
    ResolvedSwapCase,
    SwapCase,
    TokenVariant,
    _concept_surfaces,
    concept_token_variants,
    resolve_swap_cases,
    single_token_surface,
)
from jlens_reasoning.experiments.readout_utils import (
    TOP_K,
    aggregate_capability_checks,
    best_target_rank,
    positions_after_literal,
    positions_from_literal,
    prepare_scoring_input,
    top_tokens,
    validate_model_lens,
    workspace_layers,
    workspace_loading,
)
from jlens_reasoning.experiments.readout_utils import (
    find_last_subsequence as find_last_subsequence,
)
from jlens_reasoning.experiments.readout_utils import (
    write_results as write_results,
)
from jlens_reasoning.experiments.sanity_controls import (
    CONTROL_CASE_KEYS,
    CONTROL_SEEDS,
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    NORM_ATOL,
    NORM_RTOL,
    PERCENTILE_INTERPRETATION,
    PERCENTILE_QUANTILE,
    aggregate_all_checks,
    build_random_target_exclusions,
    controls_passed,
    log_rank_gain,
    matched_random_vectors,
    mean,
    require_exact_cases,
    select_random_targets,
    strict_percentile_gate,
    summarize_wrong_concept,
)

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"


@dataclass(slots=True)
class InterventionContext:
    resolved: ResolvedSwapCase
    input_ids: torch.Tensor
    scoring_input: torch.Tensor
    formatting_prefix: list[dict[str, Any]]
    clean_logits: torch.Tensor
    target_ids: tuple[int, ...]
    real_vectors_by_layer: dict[int, tuple[torch.Tensor, torch.Tensor]]
    workspace_loading: float | None


def jlens_vector(
    lens: Any,
    unembedding_weight: torch.Tensor,
    *,
    layer: int,
    token_id: int,
) -> torch.Tensor:
    jacobian = lens.jacobians[layer].to(
        device=unembedding_weight.device,
        dtype=torch.float32,
    )
    unembedding_row = unembedding_weight[token_id].to(dtype=torch.float32)
    return jacobian.T @ unembedding_row


def coordinate_swap(
    hidden: torch.Tensor,
    source_vector: torch.Tensor,
    target_vector: torch.Tensor,
    *,
    alpha: float,
) -> torch.Tensor:
    if hidden.shape[-1] != source_vector.numel():
        raise ValueError("Source vector width does not match hidden width")
    if source_vector.shape != target_vector.shape:
        raise ValueError("Source and target vectors must have the same shape")

    working = hidden.float()
    vectors = torch.stack(
        (
            source_vector.to(device=hidden.device, dtype=torch.float32),
            target_vector.to(device=hidden.device, dtype=torch.float32),
        ),
        dim=-1,
    )
    coordinates = working @ torch.linalg.pinv(vectors).T
    delta = (coordinates.flip(-1) - coordinates) @ vectors.T
    return (working + float(alpha) * delta).to(dtype=hidden.dtype)


class LensCoordinateSwapper:
    def __init__(
        self,
        blocks: Sequence[nn.Module],
        vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
        *,
        alpha: float,
    ) -> None:
        self._blocks = blocks
        self._vectors_by_layer = dict(vectors_by_layer)
        self._alpha = alpha
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, layer: int):
        source_vector, target_vector = self._vectors_by_layer[layer]

        def patch(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
            del module, inputs
            hidden = output if torch.is_tensor(output) else output[0]
            patched = coordinate_swap(
                hidden,
                source_vector,
                target_vector,
                alpha=self._alpha,
            )
            if torch.is_tensor(output):
                return patched
            return (patched, *output[1:])

        return patch

    def __enter__(self) -> LensCoordinateSwapper:
        try:
            for layer in sorted(self._vectors_by_layer):
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._hook(layer))
                )
        except Exception:
            self._remove()
            raise
        return self

    def _remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __exit__(self, *exc: Any) -> None:
        self._remove()


def execute_intervention(
    *,
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    scoring_input: torch.Tensor,
    vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    alpha: float,
) -> torch.Tensor:
    with (
        torch.inference_mode(),
        LensCoordinateSwapper(model.layers, vectors_by_layer, alpha=alpha),
    ):
        return forward_next_token(scoring_input)


def _next_token_payload(
    logits: torch.Tensor,
    target_ids: Sequence[int],
    tokenizer: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    logits = logits.detach().float().cpu()
    top1_id = int(logits.argmax().item())
    return {
        "top1_id": top1_id,
        "top1_token": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
        "target_rank": best_target_rank(logits, target_ids),
        "top_tokens": top_tokens(logits, tokenizer, k=top_k),
    }


def analyze_identity_case(
    *,
    key: str,
    clean_logits: torch.Tensor,
    scoring_input: torch.Tensor,
    model: Any,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    real_vectors_by_layer: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    target_ids: Sequence[int],
) -> dict[str, Any]:
    identity_vectors = {
        layer: (source_vector, source_vector)
        for layer, (source_vector, _) in sorted(real_vectors_by_layer.items())
    }
    intervened_logits = execute_intervention(
        model=model,
        forward_next_token=forward_next_token,
        scoring_input=scoring_input,
        vectors_by_layer=identity_vectors,
        alpha=1.0,
    )
    clean = clean_logits.detach().float().cpu()
    intervened = intervened_logits.detach().float().cpu()
    clean_top1_id = int(clean.argmax().item())
    intervened_top1_id = int(intervened.argmax().item())
    clean_target_rank = best_target_rank(clean, target_ids)
    intervened_target_rank = best_target_rank(intervened, target_ids)
    maximum_difference = float((clean - intervened).abs().max().item())
    top1_unchanged = clean_top1_id == intervened_top1_id
    target_rank_unchanged = clean_target_rank == intervened_target_rank
    logits_close = bool(
        torch.allclose(
            clean,
            intervened,
            atol=IDENTITY_ATOL,
            rtol=IDENTITY_RTOL,
        )
    )
    return {
        "key": key,
        "workspace_layers": sorted(identity_vectors),
        "alpha": 1.0,
        "atol": IDENTITY_ATOL,
        "rtol": IDENTITY_RTOL,
        "clean_top1_id": clean_top1_id,
        "intervened_top1_id": intervened_top1_id,
        "top1_unchanged": top1_unchanged,
        "clean_target_rank": clean_target_rank,
        "intervened_target_rank": intervened_target_rank,
        "target_rank_unchanged": target_rank_unchanged,
        "logits_close": logits_close,
        "maximum_absolute_logit_difference": maximum_difference,
        "passed": top1_unchanged and target_rank_unchanged and logits_close,
    }


def summarize_swap_logits(
    clean_logits: torch.Tensor,
    intervened_logits: Mapping[float, torch.Tensor],
    *,
    clean_answers: Sequence[str],
    target_answers: Sequence[str],
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    expected_variants = concept_token_variants(tokenizer, clean_answers)
    expected_ids = tuple(variant.token_id for variant in expected_variants)
    target_variants = concept_token_variants(tokenizer, target_answers)
    target_ids = tuple(variant.token_id for variant in target_variants)
    normalized_clean = clean_logits.detach().float().cpu()
    clean = _next_token_payload(normalized_clean, target_ids, tokenizer, top_k=top_k)
    clean["expected_rank"] = best_target_rank(normalized_clean, expected_ids)
    clean["expected_top1"] = clean["expected_rank"] == 1
    interventions = {
        str(alpha): _next_token_payload(logits, target_ids, tokenizer, top_k=top_k)
        for alpha, logits in sorted(intervened_logits.items())
    }
    best_rank = min(item["target_rank"] for item in interventions.values())
    return {
        "clean_answers": list(clean_answers),
        "clean_answer_variants": [asdict(variant) for variant in expected_variants],
        "target_answers": list(target_answers),
        "target_variants": [asdict(variant) for variant in target_variants],
        "clean": clean,
        "interventions": interventions,
        "best_intervened_rank": best_rank,
        "improved": best_rank < clean["target_rank"],
        "target_top1": best_rank == 1,
    }


def _summarize_lens(
    logits_by_layer: Mapping[int, torch.Tensor],
    *,
    layers: Sequence[int],
    positions: Sequence[int],
    target_ids: Sequence[int],
) -> dict[str, int]:
    candidates = [
        (
            best_target_rank(logits_by_layer[layer][position], target_ids),
            layer,
            position,
        )
        for layer in layers
        for position in positions
    ]
    rank, layer, position = min(candidates)
    return {"best_rank": rank, "layer": layer, "position": position}


def _readout_payload(
    logits_by_layer: Mapping[int, torch.Tensor],
    tokenizer: Any,
    target_ids: Sequence[int],
    *,
    top_k: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(layer): [
            {
                "position": position,
                "target_rank": best_target_rank(position_logits, target_ids),
                "top_tokens": top_tokens(position_logits, tokenizer, k=top_k),
            }
            for position, position_logits in enumerate(layer_logits)
        ]
        for layer, layer_logits in sorted(logits_by_layer.items())
    }


def analyze_case(
    case: ReadoutCase,
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    validate_model_lens(model, lens)
    jacobian_logits, model_logits, input_ids = lens.apply(
        model, case.prompt, positions=None
    )
    logit_logits, repeated_model_logits, repeated_input_ids = lens.apply(
        model, case.prompt, positions=None, use_jacobian=False
    )
    if not torch.equal(input_ids, repeated_input_ids):
        raise RuntimeError("J-Lens and logit-lens tokenization differed")
    if not torch.equal(model_logits, repeated_model_logits):
        raise RuntimeError("J-Lens and logit-lens baseline logits differed")

    target_variants = concept_token_variants(tokenizer, case.target_concepts)
    target_ids = tuple(variant.token_id for variant in target_variants)
    baseline_top1_id = int(model_logits[-1].argmax().item())
    scored_positions = (
        list(range(input_ids.shape[1]))
        if case.literal_argument is None
        else positions_after_literal(tokenizer, input_ids, case.literal_argument)
    )
    scored_layers = workspace_layers(model.n_layers, lens.source_layers)
    if not scored_layers:
        raise ValueError("No fitted layers fall inside the workspace range")

    summaries = {
        "jacobian_lens": _summarize_lens(
            jacobian_logits,
            layers=scored_layers,
            positions=scored_positions,
            target_ids=target_ids,
        ),
        "logit_lens": _summarize_lens(
            logit_logits,
            layers=scored_layers,
            positions=scored_positions,
            target_ids=target_ids,
        ),
    }
    checks: dict[str, bool] = {}
    diagnostics: dict[str, bool] = {}
    if case.key == "spider":
        jacobian_rank = summaries["jacobian_lens"]["best_rank"]
        logit_rank = summaries["logit_lens"]["best_rank"]
        diagnostics["paper_top1_hit"] = jacobian_rank == 1
        checks["read_capability"] = jacobian_rank <= 5 and jacobian_rank < logit_rank
    return {
        "key": case.key,
        "prompt": case.prompt,
        "expected_answers": list(case.expected_answers),
        "target_concepts": list(case.target_concepts),
        "input_ids": input_ids[0].tolist(),
        "input_tokens": [
            tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            for token_id in input_ids[0].tolist()
        ],
        "target_variants": [asdict(variant) for variant in target_variants],
        "baseline": {
            "top1_id": baseline_top1_id,
            "top1_token": tokenizer.decode(
                [baseline_top1_id], clean_up_tokenization_spaces=False
            ),
            "top_tokens": top_tokens(model_logits[-1], tokenizer, k=top_k),
        },
        "workspace_layers": scored_layers,
        "scored_positions": scored_positions,
        "summary": summaries,
        "readouts": {
            "jacobian_lens": _readout_payload(
                jacobian_logits, tokenizer, target_ids, top_k=top_k
            ),
            "logit_lens": _readout_payload(
                logit_logits, tokenizer, target_ids, top_k=top_k
            ),
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "passed": all(checks.values()),
    }


def _token_vectors_by_layer(
    *,
    lens: Any,
    unembedding_weight: torch.Tensor,
    layers: Sequence[int],
    source_token_id: int,
    target_token_id: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    return {
        layer: (
            jlens_vector(
                lens,
                unembedding_weight,
                layer=layer,
                token_id=source_token_id,
            ),
            jlens_vector(
                lens,
                unembedding_weight,
                layer=layer,
                token_id=target_token_id,
            ),
        )
        for layer in layers
    }


def _prepare_intervention_context(
    resolved: ResolvedSwapCase,
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> InterventionContext:
    input_ids = model.encode(resolved.read_case.prompt)
    scoring_input, formatting_prefix = prepare_scoring_input(
        input_ids,
        forward_next_token=forward_next_token,
        tokenizer=tokenizer,
    )
    vectors_by_layer = _token_vectors_by_layer(
        lens=lens,
        unembedding_weight=unembedding_weight,
        layers=layers,
        source_token_id=resolved.source.token_id,
        target_token_id=resolved.target.token_id,
    )
    loading = None
    if resolved.read_case.literal_argument is not None:
        with (
            torch.inference_mode(),
            ActivationRecorder(model.layers, at=layers) as recorder,
        ):
            forward_next_token(input_ids)
        loading = workspace_loading(
            recorder.activations,
            {layer: vectors_by_layer[layer][0] for layer in layers},
            positions=positions_from_literal(
                tokenizer,
                input_ids,
                resolved.read_case.literal_argument,
            ),
        )
    target_variants = concept_token_variants(
        tokenizer,
        resolved.case.target_answers,
    )
    with torch.inference_mode():
        clean_logits = forward_next_token(scoring_input)
    return InterventionContext(
        resolved=resolved,
        input_ids=input_ids,
        scoring_input=scoring_input,
        formatting_prefix=formatting_prefix,
        clean_logits=clean_logits,
        target_ids=tuple(variant.token_id for variant in target_variants),
        real_vectors_by_layer=vectors_by_layer,
        workspace_loading=loading,
    )


def _rank_gain_payload(
    context: InterventionContext,
    intervened_logits: torch.Tensor,
) -> dict[str, Any]:
    clean = context.clean_logits.detach().float().cpu()
    intervened = intervened_logits.detach().float().cpu()
    clean_rank = best_target_rank(clean, context.target_ids)
    intervened_rank = best_target_rank(intervened, context.target_ids)
    return {
        "key": context.resolved.case.key,
        "intended_target_ids": list(context.target_ids),
        "clean_rank": clean_rank,
        "intervened_rank": intervened_rank,
        "intervened_top1_id": int(intervened.argmax().item()),
        "log_rank_gain": log_rank_gain(clean_rank, intervened_rank),
    }


def _intervention_payload_at_alpha(
    interventions: Mapping[str, Mapping[str, Any]],
    alpha: float,
) -> Mapping[str, Any]:
    matches = [
        payload for key, payload in interventions.items() if float(key) == float(alpha)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one intervention payload for alpha={alpha}")
    return matches[0]


def analyze_swap_case(
    case: SwapCase,
    *,
    read_case: ReadoutCase,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
    alphas: Sequence[float],
    top_k: int,
    source: TokenVariant | None = None,
    target: TokenVariant | None = None,
    context: InterventionContext | None = None,
) -> dict[str, Any]:
    source = source or single_token_surface(tokenizer, case.source_surface)
    target = target or single_token_surface(tokenizer, case.target_surface)
    if context is None:
        context = _prepare_intervention_context(
            ResolvedSwapCase(
                case=case,
                read_case=read_case,
                source=source,
                target=target,
            ),
            model=model,
            lens=lens,
            tokenizer=tokenizer,
            unembedding_weight=unembedding_weight,
            forward_next_token=forward_next_token,
            layers=layers,
        )
    intervened_logits = {
        alpha: execute_intervention(
            model=model,
            forward_next_token=forward_next_token,
            scoring_input=context.scoring_input,
            vectors_by_layer=context.real_vectors_by_layer,
            alpha=alpha,
        )
        for alpha in alphas
    }

    summary = summarize_swap_logits(
        context.clean_logits,
        intervened_logits,
        clean_answers=read_case.expected_answers,
        target_answers=case.target_answers,
        tokenizer=tokenizer,
        top_k=top_k,
    )
    return {
        "key": case.key,
        "prompt": read_case.prompt,
        "source": asdict(source),
        "target": asdict(target),
        "formatting_prefix": context.formatting_prefix,
        "workspace_loading": context.workspace_loading,
        "workspace_layers": list(layers),
        **summary,
    }


def _run_negative_controls(
    *,
    contexts: Sequence[InterventionContext],
    swap_results: Sequence[Mapping[str, Any]],
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    layers: Sequence[int],
) -> dict[str, Any]:
    require_exact_cases([{"key": context.resolved.case.key} for context in contexts])
    expected_keys = CONTROL_CASE_KEYS
    require_exact_cases(swap_results)
    real_cases = []
    for result in swap_results:
        alpha_one = _intervention_payload_at_alpha(result["interventions"], 1.0)
        real_cases.append(
            {
                "key": result["key"],
                "clean_rank": result["clean"]["target_rank"],
                "intervened_rank": alpha_one["target_rank"],
                "intervened_top1_id": alpha_one["top1_id"],
                "log_rank_gain": log_rank_gain(
                    result["clean"]["target_rank"],
                    alpha_one["target_rank"],
                ),
            }
        )
    require_exact_cases(real_cases)
    real_mean = mean([case["log_rank_gain"] for case in real_cases])

    identity_cases = [
        analyze_identity_case(
            key=context.resolved.case.key,
            clean_logits=context.clean_logits,
            scoring_input=context.scoring_input,
            model=model,
            forward_next_token=forward_next_token,
            real_vectors_by_layer=context.real_vectors_by_layer,
            target_ids=context.target_ids,
        )
        for context in contexts
    ]
    require_exact_cases(identity_cases)
    identity_passed_count = sum(bool(case["passed"]) for case in identity_cases)
    identity = {
        "configuration": {
            "alpha": 1.0,
            "operation": "source concept to the same source concept",
            "workspace_layers": list(layers),
            "activation_positions": "all",
        },
        "cases": identity_cases,
        "passed_case_count": identity_passed_count,
        "required_case_count": len(expected_keys),
        "maximum_absolute_logit_difference": max(
            case["maximum_absolute_logit_difference"] for case in identity_cases
        ),
        "passed": identity_passed_count == len(expected_keys),
    }

    random_vector_seed_results = []
    for seed in CONTROL_SEEDS:
        seed_cases = []
        norms_by_case = {}
        for context in contexts:
            random_vectors, norm_report = matched_random_vectors(
                context.real_vectors_by_layer,
                base_seed=seed,
            )
            norms_by_case[context.resolved.case.key] = norm_report
            intervened_logits = execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=random_vectors,
                alpha=1.0,
            )
            seed_cases.append(_rank_gain_payload(context, intervened_logits))
            del intervened_logits, random_vectors
        require_exact_cases(seed_cases)
        random_vector_seed_results.append(
            {
                "seed": seed,
                "cases": seed_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in seed_cases]
                ),
                "norms_by_case": norms_by_case,
            }
        )
    random_vector_means = [
        result["mean_log_rank_gain"] for result in random_vector_seed_results
    ]
    random_vector_gate = strict_percentile_gate(real_mean, random_vector_means)
    matched_random_vector = {
        "configuration": {
            "alpha": 1.0,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "generation_device": "cpu",
            "generation_dtype": "torch.float32",
            "output_device_dtype": "same as corresponding real vector",
        },
        "real_cases": real_cases,
        "real_mean_log_rank_gain": real_mean,
        "seeds": random_vector_seed_results,
        "control_mean_log_rank_gains": random_vector_means,
        "percentile_95_threshold": random_vector_gate["threshold"],
        "gate": random_vector_gate,
        "passed": random_vector_gate["passed"],
    }

    contexts_by_key = {context.resolved.case.key: context for context in contexts}
    spider_context = contexts_by_key["spider"]
    france_reference = contexts_by_key["france_capital"]
    mismatched_cases = []
    mismatch_config = []
    for context in contexts:
        wrong_reference = (
            france_reference
            if context.resolved.case.key == "spider"
            else spider_context
        )
        intervened_logits = execute_intervention(
            model=model,
            forward_next_token=forward_next_token,
            scoring_input=context.scoring_input,
            vectors_by_layer=wrong_reference.real_vectors_by_layer,
            alpha=1.0,
        )
        mismatched_cases.append(_rank_gain_payload(context, intervened_logits))
        mismatch_config.append(
            {
                "key": context.resolved.case.key,
                "source": asdict(wrong_reference.resolved.source),
                "target": asdict(wrong_reference.resolved.target),
            }
        )
        del intervened_logits
    require_exact_cases(mismatched_cases)
    wrong_summary = summarize_wrong_concept(
        real_cases,
        mismatched_cases,
    )
    wrong_concept = {
        "configuration": {
            "alpha": 1.0,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "mismatches": mismatch_config,
        },
        "matched_cases": real_cases,
        "mismatched_cases": mismatched_cases,
        **wrong_summary,
    }

    source_surfaces = tuple(
        surface
        for case in SWAP_CASES
        for surface in _concept_surfaces(case.source_surface.strip())
    )
    target_surfaces = tuple(
        surface
        for case in SWAP_CASES
        for surface in _concept_surfaces(case.target_surface.strip())
    )
    clean_answer_surfaces = tuple(
        surface
        for case in READOUT_CASES
        for answer in case.expected_answers
        for surface in _concept_surfaces(answer)
    )
    intended_answer_surfaces = tuple(
        surface
        for case in SWAP_CASES
        for answer in case.target_answers
        for surface in _concept_surfaces(answer)
    )
    formatting_token_ids = tuple(
        item["token_id"] for context in contexts for item in context.formatting_prefix
    )
    exclusions = build_random_target_exclusions(
        tokenizer,
        source_surfaces=source_surfaces,
        target_surfaces=target_surfaces,
        clean_answer_surfaces=clean_answer_surfaces,
        intended_answer_surfaces=intended_answer_surfaces,
        formatting_token_ids=formatting_token_ids,
    )
    output_vocab_size = int(unembedding_weight.shape[0])
    selected_targets = select_random_targets(
        tokenizer,
        excluded_ids=exclusions["all"],
        seeds=CONTROL_SEEDS,
        output_vocab_size=output_vocab_size,
    )
    random_target_results = []
    for selected in selected_targets:
        target_cases = []
        for context in contexts:
            vectors_by_layer = _token_vectors_by_layer(
                lens=lens,
                unembedding_weight=unembedding_weight,
                layers=layers,
                source_token_id=context.resolved.source.token_id,
                target_token_id=selected["token_id"],
            )
            intervened_logits = execute_intervention(
                model=model,
                forward_next_token=forward_next_token,
                scoring_input=context.scoring_input,
                vectors_by_layer=vectors_by_layer,
                alpha=1.0,
            )
            target_cases.append(_rank_gain_payload(context, intervened_logits))
            del intervened_logits, vectors_by_layer
        require_exact_cases(target_cases)
        random_target_results.append(
            {
                **selected,
                "cases": target_cases,
                "mean_log_rank_gain": mean(
                    [case["log_rank_gain"] for case in target_cases]
                ),
            }
        )
    random_target_means = [
        result["mean_log_rank_gain"] for result in random_target_results
    ]
    random_target_gate = strict_percentile_gate(real_mean, random_target_means)
    random_target = {
        "configuration": {
            "alpha": 1.0,
            "workspace_layers": list(layers),
            "activation_positions": "all",
            "selection": "SHA-256 index into ascending eligible token IDs",
            "output_vocab_size": output_vocab_size,
        },
        "exclusions": exclusions,
        "real_cases": real_cases,
        "real_mean_log_rank_gain": real_mean,
        "targets": random_target_results,
        "control_mean_log_rank_gains": random_target_means,
        "percentile_95_threshold": random_target_gate["threshold"],
        "gate": random_target_gate,
        "passed": random_target_gate["passed"],
    }

    control_results = {
        "identity": identity,
        "matched_random_vector": matched_random_vector,
        "wrong_concept": wrong_concept,
        "random_target": random_target,
    }
    return {
        "seeds": list(CONTROL_SEEDS),
        "definitions": {
            "log_rank_gain": "log(clean_rank) - log(intervened_rank)",
            "logarithm": "natural",
            "aggregate": "arithmetic mean across exactly five cases",
            "expected_case_keys": list(expected_keys),
            "percentile": ("sort ascending; linear interpolation at (n - 1) * 0.95"),
            "percentile_interpretation": PERCENTILE_INTERPRETATION,
            "comparison": "strictly greater than",
        },
        "thresholds": {
            "percentile_quantile": PERCENTILE_QUANTILE,
            "wrong_concept_required_case_wins": 4,
        },
        "tolerances": {
            "identity_logits": {
                "atol": IDENTITY_ATOL,
                "rtol": IDENTITY_RTOL,
            },
            "random_vector_norm_float32": {
                "atol": NORM_ATOL,
                "rtol": NORM_RTOL,
            },
            "random_vector_norm_low_precision": {
                "atol": 1e-2,
                "rtol": 1e-2,
            },
        },
        **control_results,
        "passed": controls_passed(control_results),
    }


def run_readout_sanity(
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    cases: Sequence[ReadoutCase] = READOUT_CASES,
    swap_cases: Sequence[SwapCase] = SWAP_CASES,
    alphas: Sequence[float] = (1.0, 2.0),
    minimum_improvements: int = 3,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    validate_model_lens(model, lens)
    layers = workspace_layers(model.n_layers, lens.source_layers)
    if not layers:
        raise ValueError("No fitted layers fall inside the workspace range")
    resolved_swaps = resolve_swap_cases(cases, swap_cases, tokenizer)
    if tuple(cases) != READOUT_CASES or tuple(swap_cases) != SWAP_CASES:
        raise ValueError(
            "Negative controls require the exact five configured readout and swap cases"
        )
    if 1.0 not in alphas:
        raise ValueError("Negative controls require the existing alpha=1 intervention")

    read_results = [
        analyze_case(case, model=model, lens=lens, tokenizer=tokenizer, top_k=top_k)
        for case in cases
    ]
    contexts = [
        _prepare_intervention_context(
            resolved,
            model=model,
            lens=lens,
            tokenizer=tokenizer,
            unembedding_weight=unembedding_weight,
            forward_next_token=forward_next_token,
            layers=layers,
        )
        for resolved in resolved_swaps
    ]
    swap_results = [
        analyze_swap_case(
            resolved.case,
            read_case=resolved.read_case,
            model=model,
            lens=lens,
            tokenizer=tokenizer,
            unembedding_weight=unembedding_weight,
            forward_next_token=forward_next_token,
            layers=layers,
            alphas=alphas,
            top_k=top_k,
            source=resolved.source,
            target=resolved.target,
            context=context,
        )
        for resolved, context in zip(resolved_swaps, contexts, strict=True)
    ]
    swaps_by_key = {case["key"]: case for case in swap_results}
    for read_result in read_results:
        swap_result = swaps_by_key[read_result["key"]]
        unformatted_prompt = read_result["baseline"]
        read_result["baseline"] = {
            **swap_result["clean"],
            "formatting_prefix": swap_result["formatting_prefix"],
            "unformatted_prompt": unformatted_prompt,
        }
        read_result["checks"]["baseline_top1"] = read_result["baseline"][
            "expected_top1"
        ]
        read_result["passed"] = all(read_result["checks"].values())

    existing_checks, existing_failures = aggregate_capability_checks(
        read_results,
        swap_results,
        minimum_improvements=minimum_improvements,
    )
    controls = _run_negative_controls(
        contexts=contexts,
        swap_results=swap_results,
        model=model,
        lens=lens,
        tokenizer=tokenizer,
        unembedding_weight=unembedding_weight,
        forward_next_token=forward_next_token,
        layers=layers,
    )
    checks, failures, passed = aggregate_all_checks(
        existing_checks,
        existing_failures,
        controls,
    )
    return {
        "model": MODEL_NAME,
        "lens": {
            "repo": LENS_REPO,
            "revision": LENS_REVISION,
            "file": LENS_FILE,
            "n_prompts": lens.n_prompts,
            "d_model": lens.d_model,
            "source_layers": list(lens.source_layers),
        },
        "n_layers": model.n_layers,
        "d_model": model.d_model,
        "top_k": top_k,
        "intervention_strengths": list(alphas),
        "cases": read_results,
        "swaps": swap_results,
        "controls": controls,
        "checks": checks,
        "failures": failures,
        "passed": passed,
    }
