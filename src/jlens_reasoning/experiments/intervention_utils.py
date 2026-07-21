"""Shared J-Lens intervention mechanics and swap analysis."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from jlens.hooks import ActivationRecorder
from torch import nn

from jlens_reasoning.experiments.readout_cases import (
    ReadoutCase,
    ResolvedSwapCase,
    SwapCase,
    TokenVariant,
    concept_token_variants,
    single_token_surface,
)
from jlens_reasoning.experiments.readout_utils import (
    best_target_rank,
    positions_from_literal,
    prepare_scoring_input,
    top_tokens,
    workspace_loading,
)
from jlens_reasoning.experiments.sanity_controls import (
    IDENTITY_ATOL,
    IDENTITY_RTOL,
    log_rank_gain,
)


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
