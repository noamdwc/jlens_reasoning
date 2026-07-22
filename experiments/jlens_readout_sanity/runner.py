"""Execution and result assembly for the J-Lens readout sanity experiment."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch
from jlens.hooks import ActivationRecorder

from experiments.jlens_readout_sanity.constants import (
    CONTROL_ALPHA,
    CONTROL_REQUIRED_CASE_COUNT,
    DEFAULT_INTERVENTION_STRENGTHS,
    DEFAULT_MAX_FORMATTING_TOKENS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    SPIDER_READ_MAX_RANK,
    SWAP_TARGET_TOP1_REQUIRED_COUNT,
    TOP_K,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
)
from experiments.jlens_readout_sanity.controls import (
    aggregate_all_checks,
    run_negative_controls,
)
from experiments.jlens_readout_sanity.types import (
    InterventionContext,
    ReadoutCase,
    ResolvedSwapCase,
    SwapCase,
)
from jlens_reasoning.experiments_utils.interventions import (
    execute_intervention,
    token_vectors_by_layer,
)
from jlens_reasoning.experiments_utils.tokens import (
    TokenVariant,
    best_target_rank,
    concept_token_variants,
    next_token_payload,
    positions_after_literal,
    positions_from_literal,
    prepare_scoring_input,
    single_token_surface,
    top_tokens,
)
from jlens_reasoning.experiments_utils.validation import (
    validate_model_lens,
    workspace_layers,
    workspace_loading,
)


def _validate_case_configuration(
    cases: Sequence[ReadoutCase],
    swap_cases: Sequence[SwapCase],
) -> None:
    read_keys = [case.key for case in cases]
    swap_keys = [case.key for case in swap_cases]
    if len(read_keys) != CONTROL_REQUIRED_CASE_COUNT or len(swap_keys) != (
        CONTROL_REQUIRED_CASE_COUNT
    ):
        raise ValueError(
            "Negative controls require the exact "
            f"five ({CONTROL_REQUIRED_CASE_COUNT}) readout and swap cases"
        )
    if len(set(read_keys)) != len(read_keys):
        raise ValueError("Readout case keys must be unique")
    if len(set(swap_keys)) != len(swap_keys):
        raise ValueError("Swap case keys must be unique")
    if read_keys != swap_keys:
        raise ValueError(
            "Readout and swap cases must have the same keys in the same order"
        )


def resolve_swap_cases(
    cases: Sequence[ReadoutCase],
    swap_cases: Sequence[SwapCase],
    tokenizer: Any,
) -> tuple[ResolvedSwapCase, ...]:
    """Resolve and validate the one-to-one readout/swap case mapping."""
    read_cases_by_key: dict[str, ReadoutCase] = {}
    for case in cases:
        if case.key in read_cases_by_key:
            raise ValueError(f"Duplicate readout case key: {case.key}")
        read_cases_by_key[case.key] = case

    seen_swap_keys: set[str] = set()
    resolved = []
    for case in swap_cases:
        if case.key in seen_swap_keys:
            raise ValueError(f"Duplicate swap case key: {case.key}")
        seen_swap_keys.add(case.key)
        if case.key not in read_cases_by_key:
            raise ValueError(f"Swap case has no matching readout case: {case.key}")
        resolved.append(
            ResolvedSwapCase(
                case=case,
                read_case=read_cases_by_key[case.key],
                source=single_token_surface(tokenizer, case.source_surface),
                target=single_token_surface(tokenizer, case.target_surface),
            )
        )

    missing_swap_keys = set(read_cases_by_key) - seen_swap_keys
    if missing_swap_keys:
        missing = ", ".join(sorted(missing_swap_keys))
        raise ValueError(f"Readout cases have no matching swap case: {missing}")
    return tuple(resolved)


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
    """Analyze one prompt with both Jacobian and logit lenses."""
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
    scored_layers = workspace_layers(
        model.n_layers,
        lens.source_layers,
        lower_fraction=WORKSPACE_LAYER_LOWER_FRACTION,
        upper_fraction=WORKSPACE_LAYER_UPPER_FRACTION,
    )
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
        checks["read_capability"] = (
            jacobian_rank <= SPIDER_READ_MAX_RANK and jacobian_rank < logit_rank
        )
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


def summarize_swap_logits(
    clean_logits: torch.Tensor,
    intervened_logits: Mapping[float, torch.Tensor],
    *,
    clean_answers: Sequence[str],
    target_answers: Sequence[str],
    tokenizer: Any,
    top_k: int,
) -> dict[str, Any]:
    """Summarize clean and intervened next-token logits for one swap."""
    expected_variants = concept_token_variants(tokenizer, clean_answers)
    expected_ids = tuple(variant.token_id for variant in expected_variants)
    target_variants = concept_token_variants(tokenizer, target_answers)
    target_ids = tuple(variant.token_id for variant in target_variants)
    normalized_clean = clean_logits.detach().float().cpu()
    clean = next_token_payload(normalized_clean, target_ids, tokenizer, top_k=top_k)
    clean["expected_rank"] = best_target_rank(normalized_clean, expected_ids)
    clean["expected_top1"] = clean["expected_rank"] == 1
    interventions = {
        str(alpha): next_token_payload(logits, target_ids, tokenizer, top_k=top_k)
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
        max_formatting_tokens=DEFAULT_MAX_FORMATTING_TOKENS,
    )
    vectors_by_layer = token_vectors_by_layer(
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
    """Execute and summarize one configured coordinate swap."""
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


def aggregate_capability_checks(
    read_results: Sequence[Mapping[str, Any]],
    swap_results: Sequence[Mapping[str, Any]],
    *,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
) -> tuple[dict[str, bool], list[str]]:
    """Aggregate experiment-specific read and swap capability gates."""
    clean_baselines = all(
        bool(case["checks"]["baseline_top1"]) for case in read_results
    )
    spider = next((case for case in read_results if case["key"] == "spider"), None)
    spider_read = bool(spider and spider["checks"].get("read_capability", False))
    improved_count = sum(bool(case["improved"]) for case in swap_results)
    top1_count = sum(bool(case["target_top1"]) for case in swap_results)
    checks = {
        "clean_baselines": clean_baselines,
        "spider_read": spider_read,
        "swap_rank_improvements": improved_count >= minimum_improvements,
        "swap_target_top1": top1_count >= SWAP_TARGET_TOP1_REQUIRED_COUNT,
    }
    failures: list[str] = []
    if not clean_baselines:
        failures.append("one or more clean baseline answers are not top-1")
    if not spider_read:
        failures.append("spider readout did not satisfy the Qwen capability gate")
    if not checks["swap_rank_improvements"]:
        failures.append(
            f"coordinate swaps improved {improved_count}/{len(swap_results)} "
            f"target ranks; need at least {minimum_improvements}"
        )
    if not checks["swap_target_top1"]:
        failures.append("no coordinate swap placed its target answer at top-1")
    return checks, failures


def run_readout_sanity(
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    cases: Sequence[ReadoutCase],
    swap_cases: Sequence[SwapCase],
    alphas: Sequence[float] = DEFAULT_INTERVENTION_STRENGTHS,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """Run the complete fixed five-case sanity experiment."""
    validate_model_lens(model, lens)
    _validate_case_configuration(cases, swap_cases)
    resolved_swaps = resolve_swap_cases(cases, swap_cases, tokenizer)
    layers = workspace_layers(
        model.n_layers,
        lens.source_layers,
        lower_fraction=WORKSPACE_LAYER_LOWER_FRACTION,
        upper_fraction=WORKSPACE_LAYER_UPPER_FRACTION,
    )
    if not layers:
        raise ValueError("No fitted layers fall inside the workspace range")
    if CONTROL_ALPHA not in alphas:
        raise ValueError(
            "Negative controls require the existing "
            f"alpha={CONTROL_ALPHA:g} intervention"
        )

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
    controls = run_negative_controls(
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
        "policy": {
            "clean_baselines": {"required_count": len(read_results)},
            "spider_read": {
                "maximum_rank": SPIDER_READ_MAX_RANK,
                "requires_better_than_logit_lens": True,
                "paper_target_rank": 1,
            },
            "swap_rank_improvements": {
                "required_count": minimum_improvements,
                "case_count": len(swap_results),
            },
            "swap_target_top1": {
                "required_count": SWAP_TARGET_TOP1_REQUIRED_COUNT,
                "case_count": len(swap_results),
                "paper_primary_alpha": CONTROL_ALPHA,
                "paper_target_rank": 1,
            },
        },
        "cases": read_results,
        "swaps": swap_results,
        "controls": controls,
        "checks": checks,
        "failures": failures,
        "passed": passed,
    }
