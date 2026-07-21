"""Read-and-change sanity checks for the public Qwen Jacobian lens."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch

from jlens_reasoning.experiments.intervention_utils import (
    InterventionContext as InterventionContext,
)
from jlens_reasoning.experiments.intervention_utils import (
    LensCoordinateSwapper as LensCoordinateSwapper,
)
from jlens_reasoning.experiments.intervention_utils import (
    _prepare_intervention_context,
    analyze_swap_case,
)
from jlens_reasoning.experiments.intervention_utils import (
    _token_vectors_by_layer as _token_vectors_by_layer,
)
from jlens_reasoning.experiments.intervention_utils import (
    analyze_identity_case as analyze_identity_case,
)
from jlens_reasoning.experiments.intervention_utils import (
    coordinate_swap as coordinate_swap,
)
from jlens_reasoning.experiments.intervention_utils import (
    execute_intervention as execute_intervention,
)
from jlens_reasoning.experiments.intervention_utils import (
    jlens_vector as jlens_vector,
)
from jlens_reasoning.experiments.intervention_utils import (
    summarize_swap_logits as summarize_swap_logits,
)
from jlens_reasoning.experiments.readout_cases import (
    READOUT_CASES,
    SWAP_CASES,
    ReadoutCase,
    SwapCase,
    concept_token_variants,
    resolve_swap_cases,
)
from jlens_reasoning.experiments.readout_cases import (
    ResolvedSwapCase as ResolvedSwapCase,
)
from jlens_reasoning.experiments.readout_cases import (
    TokenVariant as TokenVariant,
)
from jlens_reasoning.experiments.readout_cases import (
    single_token_surface as single_token_surface,
)
from jlens_reasoning.experiments.readout_constants import (
    DEFAULT_INTERVENTION_STRENGTHS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    LENS_FILE,
    LENS_REPO,
    LENS_REVISION,
    MODEL_NAME,
    SPIDER_READ_MAX_RANK,
    TOP_K,
)
from jlens_reasoning.experiments.readout_controls import (
    run_negative_controls as _run_negative_controls,
)
from jlens_reasoning.experiments.readout_utils import (
    aggregate_capability_checks,
    best_target_rank,
    positions_after_literal,
    top_tokens,
    validate_model_lens,
    workspace_layers,
)
from jlens_reasoning.experiments.readout_utils import (
    find_last_subsequence as find_last_subsequence,
)
from jlens_reasoning.experiments.readout_utils import (
    positions_from_literal as positions_from_literal,
)
from jlens_reasoning.experiments.readout_utils import (
    prepare_scoring_input as prepare_scoring_input,
)
from jlens_reasoning.experiments.readout_utils import (
    workspace_loading as workspace_loading,
)
from jlens_reasoning.experiments.readout_utils import (
    write_results as write_results,
)
from jlens_reasoning.experiments.sanity_constants import CONTROL_ALPHA
from jlens_reasoning.experiments.sanity_controls import (
    aggregate_all_checks,
)


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


def run_readout_sanity(
    *,
    model: Any,
    lens: Any,
    tokenizer: Any,
    unembedding_weight: torch.Tensor,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    cases: Sequence[ReadoutCase] = READOUT_CASES,
    swap_cases: Sequence[SwapCase] = SWAP_CASES,
    alphas: Sequence[float] = DEFAULT_INTERVENTION_STRENGTHS,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
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
