"""Stateless helpers shared by J-Lens readout experiments."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from jlens_reasoning.experiments.readout_cases import _concept_surfaces
from jlens_reasoning.experiments.readout_constants import (
    DEFAULT_MAX_FORMATTING_TOKENS,
    DEFAULT_MINIMUM_IMPROVEMENTS,
    TOP_K,
    WORKSPACE_LAYER_LOWER_FRACTION,
    WORKSPACE_LAYER_UPPER_FRACTION,
)


def find_last_subsequence(
    sequence: Sequence[int], patterns: Iterable[Sequence[int]]
) -> tuple[int, int]:
    matches: list[tuple[int, int]] = []
    for pattern in patterns:
        width = len(pattern)
        if not width:
            continue
        for start in range(len(sequence) - width + 1):
            if list(sequence[start : start + width]) == list(pattern):
                matches.append((start, start + width))
    if not matches:
        raise ValueError("Literal argument token span was not found in prompt")
    return max(matches, key=lambda span: (span[0], span[1]))


def positions_after_literal(
    tokenizer: Any, input_ids: torch.Tensor, literal: str
) -> list[int]:
    sequence = input_ids[0].tolist()
    patterns = [
        tokenizer.encode(surface, add_special_tokens=False)
        for surface in _concept_surfaces(literal)
    ]
    _, end = find_last_subsequence(sequence, patterns)
    positions = list(range(end, len(sequence)))
    if not positions:
        raise ValueError(f"No positions remain after literal argument {literal!r}")
    return positions


def positions_from_literal(
    tokenizer: Any,
    input_ids: torch.Tensor,
    literal: str,
) -> list[int]:
    sequence = input_ids[0].tolist()
    patterns = [
        tokenizer.encode(surface, add_special_tokens=False)
        for surface in _concept_surfaces(literal)
    ]
    start, _ = find_last_subsequence(sequence, patterns)
    return list(range(start, len(sequence)))


def best_target_rank(logits: torch.Tensor, target_ids: Sequence[int]) -> int:
    if logits.ndim != 1:
        raise ValueError("best_target_rank expects one logits vector")
    if not target_ids:
        raise ValueError("best_target_rank needs at least one target token")
    token_ids = torch.arange(logits.numel(), device=logits.device)
    ranks = []
    for target_id in target_ids:
        target_logit = logits[target_id]
        higher = (logits > target_logit).sum()
        earlier_ties = ((logits == target_logit) & (token_ids < target_id)).sum()
        ranks.append(1 + int(higher.item()) + int(earlier_ties.item()))
    return min(ranks)


def top_tokens(logits: torch.Tensor, tokenizer: Any, *, k: int = TOP_K) -> list[dict]:
    values, indices = torch.topk(logits, k=min(k, logits.numel()))
    return [
        {
            "token_id": int(token_id),
            "token": tokenizer.decode(
                [int(token_id)], clean_up_tokenization_spaces=False
            ),
            "logit": float(value),
        }
        for value, token_id in zip(values.tolist(), indices.tolist(), strict=True)
    ]


def prepare_scoring_input(
    input_ids: torch.Tensor,
    *,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: Any,
    max_formatting_tokens: int = DEFAULT_MAX_FORMATTING_TOKENS,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    scoring_input = input_ids
    prefix: list[dict[str, Any]] = []
    for _ in range(max_formatting_tokens):
        logits = forward_next_token(scoring_input)
        token_id = int(logits.argmax().item())
        surface = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        if surface.strip():
            break
        prefix.append({"token_id": token_id, "token": surface})
        next_id = torch.tensor(
            [[token_id]],
            device=scoring_input.device,
            dtype=scoring_input.dtype,
        )
        scoring_input = torch.cat((scoring_input, next_id), dim=1)
    return scoring_input, prefix


def aggregate_capability_checks(
    read_results: Sequence[Mapping[str, Any]],
    swap_results: Sequence[Mapping[str, Any]],
    *,
    minimum_improvements: int = DEFAULT_MINIMUM_IMPROVEMENTS,
) -> tuple[dict[str, bool], list[str]]:
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
        "swap_target_top1": top1_count >= 1,
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


def workspace_loading(
    activations_by_layer: Mapping[int, torch.Tensor],
    vectors_by_layer: Mapping[int, torch.Tensor],
    *,
    positions: Sequence[int],
) -> float:
    similarities = []
    for layer, vector in vectors_by_layer.items():
        hidden = activations_by_layer[layer][0, list(positions)].float()
        direction = vector.to(hidden.device, dtype=torch.float32).expand_as(hidden)
        similarities.append(
            torch.nn.functional.cosine_similarity(hidden, direction, dim=-1)
        )
    return float(torch.cat(similarities).mean().item())


def workspace_layers(n_layers: int, source_layers: Iterable[int]) -> list[int]:
    lower = math.ceil(WORKSPACE_LAYER_LOWER_FRACTION * n_layers)
    upper = math.floor(WORKSPACE_LAYER_UPPER_FRACTION * n_layers)
    return [layer for layer in source_layers if lower <= layer <= upper]


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.item() if value.ndim == 0 else value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    return value


def write_results(path: Path, result: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_model_lens(model: Any, lens: Any) -> None:
    if model.d_model != lens.d_model:
        raise ValueError(
            f"Model/lens residual width mismatch: {model.d_model} != {lens.d_model}"
        )
    invalid = [layer for layer in lens.source_layers if not 0 <= layer < model.n_layers]
    if invalid:
        raise ValueError(
            f"Lens fitted layers {invalid} are outside model depth {model.n_layers}"
        )
