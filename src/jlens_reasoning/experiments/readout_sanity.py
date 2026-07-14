"""Readout-only sanity checks for the public Qwen Jacobian lens."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = (
    "qwen3.5-4b/jlens/Salesforce-wikitext/"
    "Qwen3.5-4B_jacobian_lens_n1000.pt"
)
TOP_K = 25


@dataclass(frozen=True, slots=True)
class ReadoutCase:
    key: str
    prompt: str
    expected_answers: tuple[str, ...]
    target_concepts: tuple[str, ...]
    literal_argument: str | None = None


READOUT_CASES = (
    ReadoutCase(
        key="spider",
        prompt="The number of legs on the animal that spins webs is",
        expected_answers=("8", "eight"),
        target_concepts=("spider",),
    ),
    ReadoutCase(
        key="france_capital",
        prompt="The capital of France is the city of",
        expected_answers=("Paris",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_language",
        prompt="Most people in France speak",
        expected_answers=("French",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_continent",
        prompt="France is a country on the continent of",
        expected_answers=("Europe",),
        target_concepts=("France",),
        literal_argument="France",
    ),
    ReadoutCase(
        key="france_currency",
        prompt="The single-word name for the currency now used in France is the",
        expected_answers=("Euro",),
        target_concepts=("France",),
        literal_argument="France",
    ),
)


@dataclass(frozen=True, slots=True)
class TokenVariant:
    token_id: int
    surface: str


def _concept_surfaces(concept: str) -> tuple[str, ...]:
    bases = (concept, concept.lower(), concept.capitalize(), concept.upper())
    ordered: list[str] = []
    for base in bases:
        for surface in (base, f" {base}"):
            if surface not in ordered:
                ordered.append(surface)
    return tuple(ordered)


def concept_token_variants(
    tokenizer: Any, concepts: Sequence[str]
) -> tuple[TokenVariant, ...]:
    variants: list[TokenVariant] = []
    seen_ids: set[int] = set()
    for concept in concepts:
        for surface in _concept_surfaces(concept):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in seen_ids:
                seen_ids.add(token_ids[0])
                variants.append(TokenVariant(token_id=token_ids[0], surface=surface))
    if not variants:
        raise ValueError(f"No single-token variants found for {tuple(concepts)!r}")
    return tuple(variants)


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


def workspace_layers(n_layers: int, source_layers: Iterable[int]) -> list[int]:
    lower = math.ceil(0.35 * n_layers)
    upper = math.floor(0.80 * n_layers)
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
