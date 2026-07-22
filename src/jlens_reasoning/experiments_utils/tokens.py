"""Tokenizer-facing helpers shared by model experiments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from jlens_reasoning.evaluation_utils import best_token_rank, top_token_values


@dataclass(frozen=True, slots=True)
class TokenVariant:
    token_id: int
    surface: str


def concept_surfaces(concept: str) -> tuple[str, ...]:
    bases = (concept, concept.lower(), concept.capitalize(), concept.upper())
    ordered: list[str] = []
    for base in bases:
        for surface in (base, f" {base}"):
            if surface not in ordered:
                ordered.append(surface)
    return tuple(ordered)


def single_token_surface(tokenizer: Any, surface: str) -> TokenVariant:
    token_ids = tokenizer.encode(surface, add_special_tokens=False)
    if len(token_ids) != 1:
        raise ValueError(
            f"Configured swap surface {surface!r} must encode as exactly one token"
        )
    return TokenVariant(token_id=token_ids[0], surface=surface)


def concept_token_variants(
    tokenizer: Any,
    concepts: Sequence[str],
) -> tuple[TokenVariant, ...]:
    variants: list[TokenVariant] = []
    seen_ids: set[int] = set()
    for concept in concepts:
        for surface in concept_surfaces(concept):
            token_ids = tokenizer.encode(surface, add_special_tokens=False)
            if len(token_ids) == 1 and token_ids[0] not in seen_ids:
                seen_ids.add(token_ids[0])
                variants.append(TokenVariant(token_id=token_ids[0], surface=surface))
    if not variants:
        raise ValueError(f"No single-token variants found for {tuple(concepts)!r}")
    return tuple(variants)


def find_last_subsequence(
    sequence: Sequence[int],
    patterns: Iterable[Sequence[int]],
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
    tokenizer: Any,
    input_ids: torch.Tensor,
    literal: str,
) -> list[int]:
    sequence = input_ids[0].tolist()
    patterns = [
        tokenizer.encode(surface, add_special_tokens=False)
        for surface in concept_surfaces(literal)
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
        for surface in concept_surfaces(literal)
    ]
    start, _ = find_last_subsequence(sequence, patterns)
    return list(range(start, len(sequence)))


def top_tokens(
    logits: torch.Tensor,
    tokenizer: Any,
    *,
    k: int,
) -> list[dict[str, Any]]:
    return [
        {
            "token_id": token_id,
            "token": token,
            "logit": logit,
        }
        for token_id, token, logit in top_token_values(logits, tokenizer, k=k)
    ]


def next_token_payload(
    logits: torch.Tensor,
    target_ids: Sequence[int],
    tokenizer: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    normalized = logits.detach().float().cpu()
    top1_id = int(normalized.argmax().item())
    return {
        "top1_id": top1_id,
        "top1_token": tokenizer.decode([top1_id], clean_up_tokenization_spaces=False),
        "target_rank": best_token_rank(normalized, target_ids),
        "top_tokens": top_tokens(normalized, tokenizer, k=top_k),
    }


def prepare_scoring_input(
    input_ids: torch.Tensor,
    *,
    forward_next_token: Callable[[torch.Tensor], torch.Tensor],
    tokenizer: Any,
    max_formatting_tokens: int,
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
