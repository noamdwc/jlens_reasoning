"""Deterministic reductions of full-vocabulary lens logits."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from experiments.flenqa_length_drift.anchors import Anchor
from jlens_reasoning.evaluation_utils import best_token_rank


@dataclass(frozen=True, slots=True)
class TokenCandidate:
    surface: str
    token_id: int


@dataclass(frozen=True, slots=True)
class RankedToken:
    rank: int
    token_id: int
    logit: float


@dataclass(frozen=True, slots=True)
class TopKValue:
    prompt_id: str
    layer: int
    position: int
    anchor_label: str
    lens_kind: str
    rank: int
    token_id: int
    logit: float


@dataclass(frozen=True, slots=True)
class TargetValue:
    prompt_id: str
    layer: int
    position: int
    anchor_label: str
    lens_kind: str
    surface: str
    token_id: int
    rank: int
    logit: float


@dataclass(frozen=True, slots=True)
class SummaryValue:
    prompt_id: str
    layer: int
    position: int
    lens_kind: str
    entropy: float
    max_logit: float
    top1_token_id: int


@dataclass(frozen=True, slots=True)
class ReadoutReduction:
    topk: tuple[TopKValue, ...]
    targets: tuple[TargetValue, ...]
    summary: tuple[SummaryValue, ...]


def deterministic_topk(logits: torch.Tensor, *, k: int) -> tuple[RankedToken, ...]:
    """Top-k ordered by descending logit, then ascending token ID."""
    if logits.ndim != 1:
        raise ValueError("deterministic_topk expects one logits vector")
    if type(k) is not int or k < 0:
        raise ValueError("top-k must be a non-negative integer")
    if torch.isnan(logits).any():
        raise ValueError("top-k logits must not contain NaN")
    count = min(k, logits.numel())
    if count == 0:
        return ()

    threshold = torch.topk(logits, k=count, sorted=False).values.min()
    strict_ids = torch.nonzero(logits > threshold, as_tuple=False).flatten()
    remaining = count - strict_ids.numel()
    boundary_ids = torch.nonzero(logits == threshold, as_tuple=False).flatten()[
        :remaining
    ]
    selected_ids = torch.cat((strict_ids, boundary_ids)).sort().values
    selected_logits = logits[selected_ids]
    order = torch.argsort(selected_logits, descending=True, stable=True)
    ordered_ids = selected_ids[order]
    return tuple(
        RankedToken(
            rank=rank,
            token_id=int(token_id),
            logit=float(logits[token_id].item()),
        )
        for rank, token_id in enumerate(ordered_ids.tolist(), start=1)
    )


def _entropy(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits.float(), dim=-1)
    positive = probabilities > 0
    return float(
        -(probabilities[positive] * probabilities[positive].log()).sum().item()
    )


def reduce_readout(
    *,
    prompt_id: str,
    lens_kind: str,
    logits_by_layer: Mapping[int, torch.Tensor],
    positions: Sequence[int],
    anchors: Sequence[Anchor],
    candidates: Sequence[TokenCandidate],
    top_k: int,
) -> ReadoutReduction:
    """Reduce one lens pass without retaining full-vocabulary logits."""
    if len(set(positions)) != len(positions):
        raise ValueError("readout positions must be unique")
    position_index = {position: index for index, position in enumerate(positions)}
    if any(anchor.position not in position_index for anchor in anchors):
        raise ValueError("every anchor must be present in readout positions")

    topk_values: list[TopKValue] = []
    target_values: list[TargetValue] = []
    summary_values: list[SummaryValue] = []
    for layer, layer_logits in sorted(logits_by_layer.items()):
        if layer_logits.ndim != 2 or layer_logits.shape[0] != len(positions):
            raise ValueError(f"Layer {layer} logits rows must match selected positions")
        for position, row in zip(positions, layer_logits, strict=True):
            ranked = deterministic_topk(row, k=1)
            if not ranked:
                raise ValueError("readout logits vocabulary must be nonempty")
            summary_values.append(
                SummaryValue(
                    prompt_id=prompt_id,
                    layer=layer,
                    position=position,
                    lens_kind=lens_kind,
                    entropy=_entropy(row),
                    max_logit=ranked[0].logit,
                    top1_token_id=ranked[0].token_id,
                )
            )
        for anchor in anchors:
            row = layer_logits[position_index[anchor.position]]
            topk_values.extend(
                TopKValue(
                    prompt_id=prompt_id,
                    layer=layer,
                    position=anchor.position,
                    anchor_label=anchor.label,
                    lens_kind=lens_kind,
                    rank=value.rank,
                    token_id=value.token_id,
                    logit=value.logit,
                )
                for value in deterministic_topk(row, k=top_k)
            )
            for candidate in candidates:
                if not 0 <= candidate.token_id < row.numel():
                    raise ValueError(
                        f"Candidate token ID {candidate.token_id} is out of range"
                    )
                target_values.append(
                    TargetValue(
                        prompt_id=prompt_id,
                        layer=layer,
                        position=anchor.position,
                        anchor_label=anchor.label,
                        lens_kind=lens_kind,
                        surface=candidate.surface,
                        token_id=candidate.token_id,
                        rank=best_token_rank(row, (candidate.token_id,)),
                        logit=float(row[candidate.token_id].item()),
                    )
                )
    if any(not math.isfinite(value.entropy) for value in summary_values):
        raise ValueError("readout entropy must be finite")
    return ReadoutReduction(
        topk=tuple(topk_values),
        targets=tuple(target_values),
        summary=tuple(summary_values),
    )
